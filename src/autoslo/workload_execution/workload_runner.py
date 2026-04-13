import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import Optional

from tqdm.auto import tqdm

import autoslo.utils.config as cfgu
import autoslo.utils.paths as pu
from autoslo.clusters.actions import ScalingAction, SpinUpAction, TearDownAction
from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.capacity_checkpoint import CapacityCheckpoint
from autoslo.clusters.cluster import ClusterState
from autoslo.clusters.managed_cluster_pool import (
    ManagedClusterPool,
    ManagedClusterPoolConfig,
)
from autoslo.clusters.redshift_provisioner import RedshiftServerlessProvisioner
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter, QueryRouterPolicy
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.logging import (
    LOGGER_NAME,
    emit_structured,
    setup_run_logging,
)
from autoslo.utils.yaml_helpers import dump
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.query_text_registry import QueryTextRegistry
from autoslo.workload_definition.schema import Schema
from autoslo.workload_definition.workload import Workload

_has_structured = lambda: bool(
    logging.getLogger(LOGGER_NAME).handlers
)  # noqa: E731


class WorkloadRunner:
    """Execute a workload against live Redshift Serverless clusters.

    The constructor mirrors :class:`~WorkloadSimulator`'s signature so that
    both can be driven from the same YAML configuration (see
    :meth:`from_config`).  Runner-specific settings (``provisioner``,
    ``maxconns``, ``closed_loop``) live in a dedicated ``runner_config``
    section.
    """

    def __init__(self, cfg: dict):
        """
        Initialize the runner with the given configuration.
        """

        # ── basic ────────────────────────────────────────────────────────
        self._run_id = str(int(self._ts()))
        self._cfg = cfgu.copy_and_apply_overrides(
            cfg, {"basic_config.run_id": self._run_id}
        )
        schema_name: str = cfgu.getd(
            self._cfg, "basic_config.schema_name", required=True
        )
        self._schema = Schema.load(schema_name)
        self._iconq_model_id: str = cfgu.getd(
            self._cfg, "basic_config.iconq_model_id", required=True
        )
        self._iconq_model = IconqModel.load(self._iconq_model_id)

        # ── workload ─────────────────────────────────────────────────────
        self._closed_loop: bool = bool(
            cfgu.getd(self._cfg, "workload_config.closed_loop", False)
        )
        self._workload = Workload.from_cfg(self._cfg, self._schema.name)
        self._workload.print_summary()

        # ── SLO ──────────────────────────────────────────────────────────
        self._slo_s: float = cfgu.getd(self._cfg, "slo_config.slo_s", 10.0)
        self._slo_metric = SloMetric(
            cfgu.getd(self._cfg, "slo_config.slo_metric", "relative")
        )
        self._slo_threshold: float = float(
            cfgu.getd(self._cfg, "slo_config.slo_threshold", 0.0)
        )
        self._slo_dict_filename: Optional[str] = cfgu.getd(
            self._cfg, "slo_config.slo_dict_filename"
        )
        self._slo_resolver = SloResolver(self._slo_s, self._slo_dict_filename)
        self._slo_objective = SloObjective(
            slo_metric=self._slo_metric,
            slo_threshold=self._slo_threshold,
        )

        # ── Runner-specific ──────────────────────────────────────────────────
        maxconns = int(cfgu.getd(self._cfg, "runner_config.maxconns", 1000))
        max_threads = cfgu.getd(self._cfg, "runner_config.max_threads", 10)
        self._executor = ThreadPoolExecutor(max_workers=max_threads)
        relative_aws_config_path = cfgu.getd(
            self._cfg,
            "runner_config.aws_config_path",
            "data/__run_configs/aws.yml",
        )
        absolute_aws_config_path = os.path.join(
            pu.AUTOSLO_ROOT, relative_aws_config_path
        )

        # ── Managed Cluster Pool ─────────────────────────────────────────────
        self._managed_cluster_pool_config = (
            ManagedClusterPoolConfig.parse_from_cfg(self._cfg)
        )
        self._allowed_rpu_sizes = list(
            self._managed_cluster_pool_config.allowed_rpu_sizes
        )
        self._provisioner = RedshiftServerlessProvisioner(
            aws_config_path=absolute_aws_config_path,
        )
        self._pool: ManagedClusterPool = ManagedClusterPool(
            provisioner=self._provisioner,
            config=self._managed_cluster_pool_config,
            maxconns=maxconns,
            search_path=self._schema.search_path,
            collect_cluster_stats=True,
            run_id=self._run_id,
        )
        self._capacity_checkpoints = CapacityCheckpoint.parse_from_cfg(
            self._cfg
        )

        # ── QueryRouter ──────────────────────────────────────────────────────
        routing_policy_str: str = cfgu.getd(
            self._cfg, "routing_config.routing_policy", "use_iconq_model"
        )
        self._routing_policy = QueryRouterPolicy(routing_policy_str)
        self._router: QueryRouter = QueryRouter(
            slo_resolver=self._slo_resolver,
            slo_metric=self._slo_metric,
            routing_policy=self._routing_policy,
        )

        # ── Autoscaler ──────────────────────────────────────────────────────
        self._autoscaler = Autoscaler(
            slo_resolver=self._slo_resolver,
            slo_objective=self._slo_objective,
            allowed_rpu_sizes=self._allowed_rpu_sizes,
            iconq_model=self._iconq_model,
            routing_policy=self._routing_policy,
            min_cluster_lifetime_s=cfgu.getd(
                self._cfg,
                "autoscaling_config.min_cluster_lifetime_s",
                1200.0,
            ),
            idle_time_before_tear_down_s=cfgu.getd(
                self._cfg,
                "autoscaling_config.idle_time_before_tear_down_s",
                600.0,
            ),
            observation_window_s=cfgu.getd(
                self._cfg,
                "autoscaling_config.observation_window_s",
                300.0,
            ),
            min_observations_to_act=cfgu.getd(
                self._cfg,
                "autoscaling_config.min_observations_to_act",
                5,
            ),
        )

        # ── Output ───────────────────────────────────────────────────────────
        self._write_text_log: bool = cfgu.getd(
            self._cfg, "output_config.write_text_log", False
        )
        self._out_dir = os.path.join(pu.get_runs_path(), self._run_id)
        os.makedirs(self._out_dir, exist_ok=False)
        dump(self._cfg, os.path.join(self._out_dir, "config.yml"))

        # ── Logging ───────────────────────────────────────────────────────────
        self._structured_handler = setup_run_logging(
            out_dir=self._out_dir,
            write_text_log=self._write_text_log,
        )

        # ── Instance Variables ───────────────────────────────────────────────
        # Futures for in-flight background spin-ups.
        self._pending_spin_ups: list[asyncio.Future] = []

    # ------------------------------------------------------------------
    # Async checkpoint reconciliation (live runner)
    # ------------------------------------------------------------------

    async def _reconcile_checkpoint_async(
        self,
        cp: CapacityCheckpoint,
        async_reference_ts: float,
    ) -> None:
        """Wait until *checkpoint.rel_time_s* elapses, then reconcile."""
        target = async_reference_ts + cp.rel_time_s
        delay = target - self._ts()
        if delay > 0:
            await asyncio.sleep(delay)
        current_time_s = self._ts()
        current_counts_per_rpu = self._pool.ready_and_pending_counts_per_rpu()
        spin_ups_needed = cp.spin_ups_needed(current_counts_per_rpu)

        if _has_structured():
            emit_structured(
                {
                    "timestamp": current_time_s,
                    "event_type": "capacity_checkpoint_reconciliation",
                    "source": "WorkloadRunner",
                    "checkpoint_rel_time_s": cp.rel_time_s,
                    "desired_rpus": list(cp.min_rpus),
                    "current_rpus": dict(current_counts_per_rpu),
                }
            )

        if not spin_ups_needed:
            if self._write_text_log:
                logging.debug(
                    "Checkpoint t=%.1f: already satisfied (current %s).",
                    cp.rel_time_s,
                    dict(current_counts_per_rpu),
                )
            return

        if self._write_text_log:
            logging.debug(
                "Checkpoint t=%.1f — spinning up %d clusters",
                cp.rel_time_s,
                len(spin_ups_needed),
            )
        for action in spin_ups_needed:
            self._on_live_spin_up(action)
            if _has_structured():
                emit_structured(
                    {
                        "timestamp": current_time_s,
                        "event_type": "spin_up",
                        "source": "WorkloadRunner",
                        "rpu": action.rpu,
                        "reason": f"capacity_checkpoint@t={cp.rel_time_s}",
                    }
                )

    # ------------------------------------------------------------------
    # Autoscaler callbacks
    # ------------------------------------------------------------------

    def _on_live_spin_up(self, action: SpinUpAction) -> None:
        """Autoscaler callback: spin up a new cluster (non-blocking).

        Offloads the blocking provisioning to a background thread so
        that the asyncio event loop remains responsive for query
        scheduling.  The autoscaler is notified via
        :meth:`notify_cluster_ready` once the cluster is available.
        """

        loop = asyncio.get_running_loop()
        ts = self._ts()
        future = loop.run_in_executor(
            self._executor, self._pool.request_spin_up, action, ts
        )
        self._pending_spin_ups.append(future)


    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def _ts(self) -> float:
        """
        Return the current UTC wall-clock time.
        """
        return datetime.now(tz=timezone.utc).timestamp()

    def _run_query_sync(
        self, run_id: str, query_id: str, query_text: str, cluster_name: str
    ) -> None:
        """
        Run a single query synchronously.

        Parameters:
            run_id: ID of the current run.
            query_id: ID of the query.
            query_text: SQL text of the query.
            cluster_name: Name of the cluster to run the query on.
        """
        logging.info(f"Starting query {query_id}")
        start_time = self._ts()
        if _has_structured():
            emit_structured(
                {
                    "timestamp": start_time,
                    "source": "WorkloadRunner",
                    "event_type": "query_execution_start",
                    "run_id": run_id,
                    "query_id": query_id,
                    "cluster_name": cluster_name,
                }
            )
        try:
            conn = self._pool.getconn(cluster_name)
        except Exception as e:
            logging.exception(
                f"Query {query_id} failed to acquire connection: {e}"
            )
            return
        try:
            with conn.cursor() as cur:
                edited = f"--{run_id}/{query_id}\n{query_text}"
                cur.execute(edited)
                try:
                    _ = cur.fetchall()
                except Exception as e:
                    pass  # Some queries do not return results.
            conn.commit()
        except Exception as e:
            # Ensure errors don't prevent returning the connection to the pool.
            try:
                conn.rollback()
            except Exception:
                pass
            logging.exception(f"Query {query_id} failed: {e}")
        finally:
            self._pool.putconn(cluster_name, conn)
        end_time = self._ts()
        latency_s = end_time - start_time
        logging.info(f"Query {query_id} finished after t={latency_s:.2f}s")
        if _has_structured():
            emit_structured(
                {
                    "timestamp": end_time,
                    "source": "WorkloadRunner",
                    "event_type": "query_execution_finish",
                    "run_id": run_id,
                    "query_id": query_id,
                    "cluster_name": cluster_name,
                    "client_side_latency_s": latency_s,
                }
            )
        self._pbar.update(1)

    async def _run_query_async(
        self,
        run_id: str,
        async_reference_ts: float,
        query: Query,
        skip_wait: bool = False,
    ) -> None:
        """
        Run a single query asynchronously, waiting until its scheduled start.

        Routing is performed **after** the sleep so that the routing
        decision reflects the live pool state at the query's actual
        arrival time — not the pool state at the start of the run.

        Parameters:
            run_id: ID of the current run.
            async_reference_ts: Reference timestamp for scheduling.
            query: The Query object to execute.
            skip_wait: If True, skip the initial sleep and route immediately.
        """
        now = self._ts()
        scheduled_time = async_reference_ts + query.rel_start_time_s
        delay = scheduled_time - now

        if skip_wait:
            logging.info(
                f"Query {query.query_id} scheduled to start at "
                f"t={scheduled_time:.2f}s "
                f"(in {delay:.2f}s), but skip_wait=True. Starting immediately."
            )
        elif delay > 0:
            logging.info(
                f"Query {query.query_id} scheduled to start at "
                f"t={scheduled_time:.2f}s "
                f"(in {delay:.2f}s). Waiting..."
            )
            await asyncio.sleep(delay)
        else:  # delay <= 0
            logging.warning(
                f"Query {query.query_id} scheduled start time "
                f"t={scheduled_time:.2f}s "
                f"is in the past (delay={delay:.2f}s). Starting immediately."
            )

        # ── Find query text ────────────────────────────────────────────────
        query_text = QueryTextRegistry.get(
            self._schema.name, query.query_text_id
        )
        if query_text is None:
            logging.error(
                f"No text found for schema '{self._schema.name}', "
                f"query_text_id '{query.query_text_id}'. Skipping query "
                f"{query.query_id}."
            )
            return

        # ── Route at arrival time ────────────────────────────────────
        route_start_ts = self._ts()
        snapshot = self._pool.snapshot(only_ready=True)
        old_predicted_latencies = {
            cluster_name: dict(cluster.predicted_latencies)
            for cluster_name, cluster in snapshot.items()
        }
        selected_cluster_name, new_predicted_latencies_on_selected = (
            self._router.route_query(
                query=query,
                clusters=snapshot,
                iconq_model=self._iconq_model,
                current_time_s=route_start_ts,
            )
        )
        self_latency_s = new_predicted_latencies_on_selected[query.query_id]
        route_end_ts = self._ts()

        if _has_structured():
            emit_structured(
                {
                    "timestamp": route_end_ts,
                    "event_type": "query_routed",
                    "query_id": query.query_id,
                    "query_text_id": query.query_text_id.value,
                    "cluster_name": selected_cluster_name,
                    "old_latency_s": None,
                    "raw_model_latency_s": None,
                    "latency_s": self_latency_s,
                    "end_time_s": route_end_ts + self_latency_s,
                    "source": "WorkloadRunner",
                }
            )

        # Update latencies of existing queries as needed (including the new one)
        for qid, latency_s in new_predicted_latencies_on_selected.items():
            old_latency_s = old_predicted_latencies.get(
                selected_cluster_name, {}
            ).get(qid, None)

            if (old_latency_s is not None) and (
                abs(latency_s - old_latency_s) < 1e-3
            ):
                # No change in latency prediction for this query, so skip the
                # update.
                continue

            completion_time_s = route_end_ts + latency_s
            emit_structured(
                {
                    "timestamp": route_end_ts,
                    "event_type": "latency_update",
                    "source": "WorkloadRunner",
                    "query_id": qid,
                    "cluster_name": selected_cluster_name,
                    "old_latency_s": old_latency_s,
                    "latency_s": latency_s,
                    "end_time_s": completion_time_s,
                }
            )

        #  ── Notify pool and autoscaler ────────────────────────────────────
        self._pool.commit_predicted_latencies(
            selected_cluster_name, new_predicted_latencies_on_selected
        )
        self._pool.on_query_start(
            query=query,
            cluster_name=selected_cluster_name,
        )
        post_snapshot = self._pool.snapshot(only_ready=False)
        autoscaler_suggested_actions: list[ScalingAction] = (
            self._autoscaler.inform(
                current_time_s=self._ts(),
                current_query=query,
                pool_snapshot_with_current_query=post_snapshot,
            )
        )
        for action in autoscaler_suggested_actions:
            match type(action):
                case SpinUpAction():
                    self._on_live_spin_up(action)
                case TearDownAction():
                    self._pool.request_tear_down(action, self._ts())
                case _:
                    if self._write_text_log:
                        logging.warning(
                            f"Unknown autoscaling action type: {type(action)}"
                        )

        # ── Execute ──────────────────────────────────────────────────
        try:
            fn = partial(
                self._run_query_sync,
                run_id,
                query.query_id,
                query_text,
                selected_cluster_name,
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, fn)
        finally:
            now = self._ts()
            self._pool.on_query_finish(
                query_id=query.query_id,
                cluster_name=selected_cluster_name,
                current_time_s=now,
            )

    async def run(self) -> None:
        """
        Run the queries from the workload file.
        """

        print(f"Run starting with ID {self._run_id}.")

        if _has_structured():
            emit_structured(
                {
                    "timestamp": self._ts(),
                    "source": "WorkloadRunner",
                    "event_type": "run_start",
                    "run_id": self._run_id,
                    "workload_name": self._workload.workload_name,
                    "num_queries": self._workload.num_queries,
                    "routing_policy": self._router.routing_policy.value,
                    "closed_loop": self._closed_loop,
                }
            )

        # Add a 30-second buffer between the reference timestamp and the first
        # query's scheduled start time for setup.
        async_reference_ts = self._ts() + 30
        logging.info(f"Async reference timestamp: {async_reference_ts:.2f}s")

        tasks: list[asyncio.Task] = []

        # Schedule capacity checkpoint reconciliations.
        for cp in self._capacity_checkpoints:
            tasks.append(
                asyncio.ensure_future(
                    self._reconcile_checkpoint_async(cp, async_reference_ts)
                )
            )

        self._pbar = tqdm(
            total=self._workload.num_queries, desc="Queries", unit="q"
        )

        try:
            for query in self._workload.queries():

                if not self._closed_loop:
                    task = asyncio.ensure_future(
                        self._run_query_async(
                            self._run_id, async_reference_ts, query
                        )
                    )
                    tasks.append(task)
                else:
                    # In closed loop, wait for each query to finish before
                    # starting the next.  Ignore rel_start_time_s.
                    await self._run_query_async(
                        self._run_id, async_reference_ts, query, skip_wait=True
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logging.error("Query task failed: %s", r)

            # Wait for any in-flight background spin-ups to finish
            # before tearing down clusters.
            if self._pending_spin_ups:
                spin_up_results = await asyncio.gather(
                    *self._pending_spin_ups, return_exceptions=True
                )
                for r in spin_up_results:
                    if isinstance(r, Exception):
                        logging.error("Background spin-up failed: %s", r)
        finally:
            self._pbar.close()

            # Graceful cleanup: tear down every remaining READY cluster.
            # request_tear_down → _finalize_removal is synchronous and
            # may block (stats collection + provisioner API call), so
            # we dispatch each call via the default thread-pool executor.
            loop = asyncio.get_running_loop()
            remaining = list(self._pool.clusters_in_state(ClusterState.READY))
            for cn in remaining:
                try:
                    await loop.run_in_executor(
                        self._executor,
                        partial(
                            self._pool.request_tear_down,
                            TearDownAction(
                                reason="run_cleanup",
                                cluster_name=cn,
                            ),
                            self._ts(),
                            force=True,
                        ),
                    )
                except Exception:
                    logging.exception("Failed to tear down cluster %s.", cn)

        logging.info(f"Run finished at {self._ts()}.")

        if _has_structured():
            emit_structured(
                {
                    "timestamp": self._ts(),
                    "source": "WorkloadRunner",
                    "event_type": "run_finish",
                    "run_id": self._run_id,
                    "workload_name": self._workload.workload_name,
                }
            )

        # Finalize structured log (consolidate shards).
        if (
            hasattr(self, "_structured_handler")
            and self._structured_handler is not None
        ):
            self._structured_handler.finalize()


if __name__ == "__main__":
    cfg, config_path = cfgu.load_config_from_cli(
        "Run queries from a workload using a YAML config file.",
    )
    qr = WorkloadRunner(cfg)
    asyncio.run(qr.run())
