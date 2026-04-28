import asyncio
import concurrent.futures
import logging
import os
import threading
from functools import partial
from typing import Optional

from tqdm.auto import tqdm

import autoslo.utils.config as cfgu
from autoslo.clusters.actions import SpinUpAction, TearDownAction
from autoslo.clusters.capacity_checkpoint import CapacityCheckpoint
from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.clusters.redshift_provisioner import RedshiftServerlessProvisioner
from autoslo.config.structured_config import StructuredConfig
from autoslo.routing.wrapper import route_and_update_bookkeeping
from autoslo.utils.logging import emit_structured
from autoslo.utils.structured_events import (
    BaseStructuredEvent,
    EventType,
    QueryRelatedEvent,
    wall_clock_utc,
)
from autoslo.utils.yaml_helpers import dump_yaml
from autoslo.workload_definition.query import Query, QueryTextId


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

        # ── Determine run_id ─────────────────────────────────────────
        self._run_id = str(int(wall_clock_utc()))
        self._cfg = cfgu.copy_and_apply_overrides(
            cfg, {"basic_config.run_id": self._run_id}
        )

        # ── Build, parse and dump structured config ──────────────────────────────
        structured_config = StructuredConfig.build(
            self._cfg, self._run_id, is_runner=True
        )

        self._query_text_registry = structured_config.query_text_registry
        self._iconq_model = structured_config.iconq_model
        self._closed_loop = structured_config.closed_loop
        self._workload = structured_config.workload
        self._slo_objective = structured_config.slo_objective
        self._slo_resolver = structured_config.slo_resolver
        self._executor = structured_config.thread_pool_executor
        self._pool = structured_config.pool
        self._capacity_checkpoints = structured_config.capacity_checkpoints
        self._router = structured_config.router
        self._autoscaler = structured_config.autoscaler
        self._out_dir = structured_config.out_dir
        self._write_text_log = structured_config.write_text_log
        self._structured_handler = structured_config.structured_log_handler

        dump_yaml(self._cfg, os.path.join(self._out_dir, "config.yml"))

        # ── Instance Variables ───────────────────────────────────────────────
        self._routing_lock = threading.Lock()
        self._spin_ups_lock = threading.Lock()
        self._pending_spin_ups: list[concurrent.futures.Future] = []

    # ------------------------------------------------------------------
    # Async checkpoint reconciliation (live runner)
    # ------------------------------------------------------------------

    async def _reconcile_checkpoint_async(
        self,
        checkpoint: CapacityCheckpoint,
        async_reference_ts: float,
    ) -> None:
        """Wait until *checkpoint.rel_time_s* elapses, then reconcile."""
        delay = checkpoint.rel_time_s - self._rel_time_s()
        if delay > 0:
            await asyncio.sleep(delay)
        checkpoint.reconcile(
            pool=self._pool,
            source="WorkloadRunner",
            on_spin_up=self._on_live_spin_up,
            write_text_log=self._write_text_log,
            rel_time_s_getter=self._rel_time_s,
        )

    # ------------------------------------------------------------------
    # Autoscaler callbacks
    # ------------------------------------------------------------------

    def _on_live_spin_up(self, action: SpinUpAction) -> None:
        """Autoscaler callback: spin up a new cluster (non-blocking).

        Submits the blocking provisioning to the thread-pool executor.
        Thread-safe — may be called from any thread (event loop,
        executor thread running routing, etc.).
        """

        future = self._executor.submit(
            self._pool.request_spin_up, action, self._rel_time_s()
        )

        def _on_spin_up_done(fut: concurrent.futures.Future) -> None:
            exc = fut.exception()
            if exc is not None:
                return
            cluster_name = fut.result()
            if cluster_name is None:
                # Spin-up was denied by the budget; SPIN_UP_BLOCKED was
                # already emitted by the pool.  Disable future spin-up
                # considerations in the autoscaler.
                self._autoscaler.disable_spin_up()
                return
            rpu = Cluster.rpu_for_cluster_name(cluster_name)
            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=self._rel_time_s(),
                    event_type=EventType.CLUSTER_READY,
                    source="WorkloadRunner",
                    cluster_name=cluster_name,
                    details={
                        "rpu": rpu,
                        "num_active_clusters": len(
                            self._pool.clusters_in_state(ClusterState.READY)
                        ),
                    },
                )
            )

        future.add_done_callback(_on_spin_up_done)
        with self._spin_ups_lock:
            self._pending_spin_ups.append(future)

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def _rel_time_s(self) -> float:
        """Relative time in seconds since run start."""
        return wall_clock_utc() - self._async_reference_ts

    def _route_locked(self, query: Query) -> str:
        """Route a query under the serialisation lock.

        The lock ensures that routing (snapshot → model inference →
        ``on_query_start`` → autoscaler) is atomic, preventing
        concurrent routing calls from clobbering each other's
        ``predicted_latencies`` updates.

        Called from an executor thread — never from the event loop.
        """
        with self._routing_lock:
            return route_and_update_bookkeeping(
                source="WorkloadRunner",
                rel_time_s_getter=self._rel_time_s,
                pool=self._pool,
                router=self._router,
                query=query,
                iconq_model=self._iconq_model,
                autoscaler=self._autoscaler,
                on_spin_up=self._on_live_spin_up,
                write_text_log=self._write_text_log,
                simulator_pending_events_heap=None,
            )

    def _run_query_sync(
        self,
        query_id: str,
        query_text_id: QueryTextId,
        query_text: str,
        cluster_name: str,
    ) -> Optional[float]:
        """
        Run a single query synchronously.

        Returns the measured client-side latency in seconds on success,
        or ``None`` if the query failed (connection error, SQL error, etc.).
        """
        logging.info(f"Starting query {query_id}")
        start_rel_time_s = self._rel_time_s()
        emit_structured(
            QueryRelatedEvent(
                rel_time_s=start_rel_time_s,
                event_type=EventType.QUERY_EXECUTION_START,
                source="WorkloadRunner",
                cluster_name=cluster_name,
                query_id=query_id,
                query_text_id=query_text_id,
            )
        )
        try:
            conn = self._pool.getconn(cluster_name)
        except Exception as e:
            logging.exception(
                f"Query {query_id} failed to acquire connection: {e}"
            )
            return None
        succeeded = False
        try:
            with conn.cursor() as cur:
                edited = f"--{self._run_id}/{query_id}\n{query_text}"
                cur.execute(edited)
                try:
                    _ = cur.fetchall()
                except Exception as e:
                    pass  # Some queries do not return results.
            conn.commit()
            succeeded = True
        except Exception as e:
            # Ensure errors don't prevent returning the connection to the pool.
            try:
                conn.rollback()
            except Exception:
                pass
            logging.exception(f"Query {query_id} failed: {e}")
        finally:
            self._pool.putconn(cluster_name, conn)
        end_rel_time_s = self._rel_time_s()
        latency_s = end_rel_time_s - start_rel_time_s
        logging.info(f"Query {query_id} finished after t={latency_s:.2f}s")
        emit_structured(
            QueryRelatedEvent(
                rel_time_s=end_rel_time_s,
                event_type=EventType.QUERY_EXECUTION_FINISH,
                source="WorkloadRunner",
                cluster_name=cluster_name,
                details={"latency_s": latency_s},
                query_id=query_id,
                query_text_id=query_text_id,
            )
        )
        return latency_s if succeeded else None

    async def _run_query_async(
        self,
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
            async_reference_ts: Reference timestamp for scheduling.
            query: The Query object to execute.
            skip_wait: If True, skip the initial sleep and route immediately.
        """
        loop = asyncio.get_running_loop()
        selected_cluster_name: Optional[str] = None
        latency_s: Optional[float] = None
        try:
            delay = query.rel_start_time_s - self._rel_time_s()

            if skip_wait:
                logging.info(
                    f"Query {query.query_id} scheduled to start at "
                    f"relative time {query.rel_start_time_s:.2f}s "
                    f"(in {delay:.2f}s), but skip_wait=True. Starting immediately."
                )
            elif delay > 0:
                logging.info(
                    f"Query {query.query_id} scheduled to start at "
                    f"relative time {query.rel_start_time_s:.2f}s "
                    f"(in {delay:.2f}s). Waiting..."
                )
                await asyncio.sleep(delay)
            else:  # delay <= 0
                logging.warning(
                    f"Query {query.query_id} scheduled start time "
                    f"relative time {query.rel_start_time_s:.2f}s "
                    f"is in the past (delay={delay:.2f}s). Starting immediately."
                )

            # ── Find query text ────────────────────────────────────────────
            query_text = self._query_text_registry.get(query.query_text_id)
            if query_text is None:
                logging.error(
                    f"No text found for schema "
                    f"'{self._query_text_registry.schema_name}', "
                    f"query_text_id '{query.query_text_id}'. Skipping query "
                    f"{query.query_id}."
                )
                return

            # ── Emit arrival event (matches simulator) ────────────────
            emit_structured(
                QueryRelatedEvent(
                    rel_time_s=self._rel_time_s(),
                    event_type=EventType.ARRIVAL,
                    source="WorkloadRunner",
                    query_id=query.query_id,
                    query_text_id=query.query_text_id,
                )
            )

            # ── Route at arrival time ────────────────────────────────────
            # Routing involves model inference (CPU-bound) and must be
            # serialised to keep pool state consistent, so it runs in the
            # thread-pool under _routing_lock.
            selected_cluster_name = await loop.run_in_executor(
                self._executor, self._route_locked, query
            )

            # ── Execute ──────────────────────────────────────────────────
            fn = partial(
                self._run_query_sync,
                query.query_id,
                query.query_text_id,
                query_text,
                selected_cluster_name,
            )
            latency_s = await loop.run_in_executor(self._executor, fn)
        finally:
            if selected_cluster_name is not None:
                # on_query_finish may trigger _finalize_removal, which is
                # dispatched to the pool's background executor when one is
                # configured.  The bookkeeping itself is fast but we still
                # run it off the event loop to avoid lock contention.
                emit_structured(
                    QueryRelatedEvent(
                        rel_time_s=self._rel_time_s(),
                        event_type=EventType.COMPLETION,
                        source="WorkloadRunner",
                        cluster_name=selected_cluster_name,
                        details={"success": (latency_s is not None)},
                        query_id=query.query_id,
                        query_text_id=query.query_text_id,
                    )
                )
                loop.run_in_executor(
                    self._executor,
                    partial(
                        self._pool.on_query_finish,
                        query_id=query.query_id,
                        cluster_name=selected_cluster_name,
                        rel_time_s=self._rel_time_s(),
                    ),
                )
            self._pbar.update(1)

    async def run(self) -> None:
        """
        Run the queries from the workload file.
        """

        print(f"Run starting with ID {self._run_id}.")

        # Spin up initial clusters.
        schema = Schema.load(self._workload_runner_config.schema_name)
        self._pool.add_details_and_spin_up_initial_clusters(
            search_path=schema.search_path,
            background_executor=self._executor,
            run_id=self._run_id,
            out_dir=self._out_dir,
            write_text_log=self._write_text_log,
        )

        # Add a 30-second buffer between the reference timestamp and the first
        # query's scheduled start time for setup.
        self._async_reference_ts = wall_clock_utc() + 30

        # Propagate reference time to the provisioner so it can compute
        # relative timestamps for cluster creation_time_s and events.
        prov = self._pool.provisioner
        if isinstance(prov, RedshiftServerlessProvisioner):
            prov.reference_time_s = self._async_reference_ts

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=self._rel_time_s(),
                event_type=EventType.RUN_START,
                source="WorkloadRunner",
                details={
                    "workload_name": self._workload.workload_name,
                    "num_queries": self._workload.num_queries,
                    "routing_policy": self._router.routing_policy.value,
                    "closed_loop": self._closed_loop,
                },
            )
        )

        async_reference_ts = self._async_reference_ts
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
                        self._run_query_async(async_reference_ts, query)
                    )
                    tasks.append(task)
                else:
                    # In closed loop, wait for each query to finish before
                    # starting the next.  Ignore rel_start_time_s.
                    await self._run_query_async(
                        async_reference_ts, query, skip_wait=True
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logging.error("Query task failed: %s", r)

            # Wait for any in-flight background spin-ups to finish
            # before tearing down clusters.
            with self._spin_ups_lock:
                spin_up_snapshot = list(self._pending_spin_ups)
            if spin_up_snapshot:
                async_futs = [asyncio.wrap_future(f) for f in spin_up_snapshot]
                spin_up_results = await asyncio.gather(
                    *async_futs, return_exceptions=True
                )
                for r in spin_up_results:
                    if isinstance(r, Exception):
                        logging.error("Background spin-up failed: %s", r)
        finally:
            self._pbar.close()

            # Graceful cleanup: tear down every remaining READY cluster.
            # request_tear_down transitions to DRAINING; _finalize_removal
            # (stats collection + AWS API calls) runs in the pool's
            # background executor.
            loop = asyncio.get_running_loop()
            remaining = list(self._pool.clusters_in_state(ClusterState.READY))
            for cn in remaining:
                try:
                    emit_structured(
                        BaseStructuredEvent(
                            rel_time_s=self._rel_time_s(),
                            event_type=EventType.TEAR_DOWN_DECISION,
                            source="WorkloadRunner",
                            cluster_name=cn,
                            details={"reason": "run_cleanup"},
                        )
                    )
                    await loop.run_in_executor(
                        self._executor,
                        partial(
                            self._pool.request_tear_down,
                            TearDownAction(
                                reason="run_cleanup",
                                cluster_name=cn,
                            ),
                            self._rel_time_s(),
                            force=True,
                        ),
                    )
                except Exception:
                    logging.exception("Failed to tear down cluster %s.", cn)

            # Wait for all background finalization tasks (stats collection,
            # provisioner tear-down) that the pool dispatched.
            self._pool.wait_for_background_tasks()

        logging.info(f"Run finished at {wall_clock_utc()}.")

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=self._rel_time_s(),
                event_type=EventType.RUN_FINISH,
                source="WorkloadRunner",
                details={
                    "workload_name": self._workload.workload_name,
                },
            )
        )

        # Finalize structured log (consolidate shards).
        if (
            hasattr(self, "_structured_handler")
            and self._structured_handler is not None
        ):
            self._structured_handler.finalize()


if __name__ == "__main__":
    cfg, _ = cfgu.load_config_from_cli(
        "Run queries from a workload using a YAML config file.",
    )
    qr = WorkloadRunner(cfg)
    asyncio.run(qr.run())
