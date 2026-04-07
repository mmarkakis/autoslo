import asyncio
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import psycopg2
import yaml
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from autoslo.capacity.autoscaler import Autoscaler
from autoslo.capacity.autoscaling_policy import (
    AutoscalingPolicy,
    CapacityCheckpoint,
    NoOpPolicy,
)
from autoslo.capacity.cluster_provisioner import (
    ClusterProvisioner,
    SimulatedProvisioner,
)
from autoslo.routing.managed_cluster_pool import (
    ManagedClusterPool,
    ManagedClusterPoolConfig,
)
from autoslo.routing.router import Router
from autoslo.routing.routing_policy import RoutingPolicy
from autoslo.utils.structured_log import (
    LOGGER_NAME,
    emit_structured,
    setup_structured_logging,
)
from autoslo.workload_definition.query import SloMetric
from autoslo.workload_execution.conn_utils import ConnWithSetup
from autoslo.workload_execution.run_stats_collector import (
    SYS_EXTERNAL_QUERY_DETAIL_QUERY,
    SYS_QUERY_DETAIL_QUERY,
    SYS_QUERY_EXPLAIN_QUERY,
    SYS_QUERY_HISTORY_QUERY,
    SYS_SERVERLESS_USAGE_QUERY,
)

_has_structured = lambda: bool(
    logging.getLogger(LOGGER_NAME).handlers
)  # noqa: E731
import autoslo.utils.config as cfgu
from autoslo.workload_definition.query_text_registry import QueryTextRegistry
from autoslo.workload_definition.schema import Schema
from autoslo.workload_definition.workload import Workload

from autoslo.utils.policy_builders import (
    build_routing_policy,
    build_autoscaling_policy,
    build_managed_cluster_pool_config,
    parse_capacity_checkpoints,
)


class WorkloadRunner:
    """Execute a workload against live Redshift Serverless clusters.

    The constructor mirrors :class:`~WorkloadSimulator`'s signature so that
    both can be driven from the same YAML configuration (see
    :meth:`from_config`).  Runner-specific settings (``provisioner``,
    ``maxconns``, ``closed_loop``) live in a dedicated ``runner_config``
    section.
    """

    def __init__(
        self,
        workload_name: str,
        routing_policy: RoutingPolicy,
        schema_name: str,
        managed_cluster_pool_config: Optional[ManagedClusterPoolConfig] = None,
        autoscaling_policy: Optional[AutoscalingPolicy] = None,
        provisioner_config: Optional[dict] = None,
        maxconns: int = 1000,
        closed_loop: bool = False,
        config_path: Optional[str | Path] = None,
        capacity_checkpoints: list[CapacityCheckpoint] | None = None,
        capacity_poll_interval_s: float = 60.0,
        abs_start_time_start: str | None = None,
        abs_start_time_end: str | None = None,
        rescale_factor: float | None = None,
        max_threads: int | None = None,
    ):
        self.config_path = Path(config_path) if config_path else None
        self.workload_name = workload_name
        self.closed_loop = closed_loop
        self.maxconns = maxconns

        # Load workload, then apply optional slicing / time compression.
        self.workload = Workload(
            workload_name=workload_name, schema_name=schema_name
        )
        if abs_start_time_start is not None or abs_start_time_end is not None:
            self.workload.slice_by_abs_time(
                abs_start_time_start, abs_start_time_end
            )
        self.workload.set_rel_start_times_from_zero()
        if rescale_factor is not None:
            self.workload.rescale_rel_start_times(rescale_factor)
        self.workload.print_summary()
        self.workload_df = self.workload.df
        self.schema = Schema.load(
            schema_name or self.workload.schema_name,
        )

        # Build provisioner.
        provisioner: ClusterProvisioner
        if provisioner_config is not None:
            prov_cfg = dict(provisioner_config)
            prov_type = prov_cfg.pop("type")
            if prov_type == "redshift_serverless":
                from autoslo.capacity.redshift_provisioner import (
                    RedshiftServerlessProvisioner,
                )

                provisioner = RedshiftServerlessProvisioner(**prov_cfg)
            else:
                raise ValueError(f"Unknown provisioner type: {prov_type!r}.")
        else:
            provisioner = SimulatedProvisioner(spin_up_delay_s=0.0)

        # Build pool.
        mcp = (
            managed_cluster_pool_config
            if managed_cluster_pool_config is not None
            else ManagedClusterPoolConfig()
        )
        self.pool = ManagedClusterPool(
            provisioner=provisioner,
            config=mcp,
            maxconns=self.maxconns,
            search_path=self.schema.search_path,
        )

        # Build router.
        self.routing_policy = routing_policy
        self.routing_policy_name = type(routing_policy).__name__
        self.routing_policy_config: dict = {}
        self.router = Router(
            policy=self.routing_policy,
            pool=self.pool,
        )

        # Build autoscaler.
        policy = (
            autoscaling_policy
            if autoscaling_policy is not None
            else NoOpPolicy()
        )
        self.autoscaler = Autoscaler(
            policy=policy,
            pool=self.pool,
            on_spin_up=self._on_live_spin_up,
            on_tear_down=self._on_live_tear_down,
            capacity_checkpoints=capacity_checkpoints,
        )
        self._capacity_poll_interval_s = capacity_poll_interval_s

        # Thread-pool executor for query execution.
        self._executor = (
            ThreadPoolExecutor(max_workers=max_threads) if max_threads else None
        )

        # Futures for in-flight background spin-ups (see _on_live_spin_up).
        self._pending_spin_ups: list[asyncio.Future] = []

    # ------------------------------------------------------------------
    # Async checkpoint reconciliation (live runner)
    # ------------------------------------------------------------------

    async def _reconcile_checkpoint_async(
        self,
        checkpoint: CapacityCheckpoint,
        async_reference_ts: float,
    ) -> None:
        """Wait until *checkpoint.rel_time_s* elapses, then reconcile."""
        target = async_reference_ts + checkpoint.rel_time_s
        delay = target - self._ts()
        if delay > 0:
            await asyncio.sleep(delay)
        current_time_s = self._ts()
        self.autoscaler.reconcile_checkpoints_up_to(
            current_time_s=current_time_s, reference_time_s=async_reference_ts
        )

    # ------------------------------------------------------------------
    # Factory: create from YAML config (aligned with WorkloadSimulator)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, config_path: str | Path, **overrides: object
    ) -> "WorkloadRunner":
        """Create a :class:`WorkloadRunner` from a YAML config file.

        Parameters
        ----------
        config_path : str | Path
            Path to the YAML configuration file.
        **overrides
            Dot-delimited keys mapped to override values, e.g.
            ``slo_config.slo_s=5.0``.  Applied on top of the parsed YAML
            before the config dict is interpreted.
        """
        path = Path(config_path)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg = cfgu.copy_and_apply_overrides(cfg, overrides)
        return cls.from_config_dict(cfg, config_path=config_path)

    @classmethod
    def from_config_dict(
        cls,
        cfg: dict,
        config_path: str | Path | None = None,
    ) -> "WorkloadRunner":
        """Create a :class:`WorkloadRunner` from an already-loaded config dict.

        Sections
        --------
        workload_config     : workload_name, abs_start_time_start,
                              abs_start_time_end, rescale_factor,
                              closed_loop
        basic_config        : schema_name, iconq_model_id
        slo_config          : slo_s, slo_metric, slo_threshold,
                              slo_dict_filename
        routing_config      : routing_policy ("model" | "round_robin" | "cache_aware")
        managed_cluster_pool_config : initial_rpus, allowed_rpu_sizes,
                                spin_up_delay_s
        autoscaling_config  : autoscaling_policy ("headroom" | "noop"),
                              eta_crit, idle_periods_before_tear_down,
                              capacity_poll_interval_s,
                              min_cluster_lifetime_s
        runner_config       : provisioner, maxconns
        """

        # ── basic ────────────────────────────────────────────────────────
        schema_name: str = cfgu.getd(
            cfg, "basic_config.schema_name", required=True
        )
        iconq_model_id: Optional[str] = cfgu.getd(
            cfg, "basic_config.iconq_model_id"
        )

        # ── workload ─────────────────────────────────────────────────────
        wl_cfg: dict = cfg.get("workload_config") or {}
        workload_name: str = wl_cfg["workload_name"]
        abs_start_time_start: str | None = wl_cfg.get("abs_start_time_start")
        abs_start_time_end: str | None = wl_cfg.get("abs_start_time_end")
        rescale_factor_raw = wl_cfg.get("rescale_factor")
        rescale_factor: float | None = (
            float(rescale_factor_raw)
            if rescale_factor_raw is not None
            else None
        )
        closed_loop: bool = bool(wl_cfg.get("closed_loop", False))

        # ── SLO ──────────────────────────────────────────────────────────
        slo_s: float = cfgu.getd(cfg, "slo_config.slo_s", 10.0)
        slo_metric = SloMetric(
            cfgu.getd(cfg, "slo_config.slo_metric", "relative")
        )
        slo_dict_filename: Optional[str] = cfgu.getd(
            cfg, "slo_config.slo_dict_filename"
        )
        slo_threshold: float = float(
            cfgu.getd(cfg, "slo_config.slo_threshold", 0.0)
        )
        slo_resolver = SloResolver(slo_s, slo_dict_filename)

        # ── shared policy / pool construction ────────────────────────────
        routing_policy = build_routing_policy(
            cfg,
            iconq_model_id,
            slo_s,
            slo_resolver,
            slo_metric,
        )
        mcp = build_managed_cluster_pool_config(cfg)
        allowed_rpus: list[int] = list(
            mcp.allowed_rpu_sizes
            if mcp is not None
            else ManagedClusterPoolConfig().allowed_rpu_sizes
        )
        autoscaling_policy = build_autoscaling_policy(
            cfg,
            slo_resolver,
            slo_metric,
            slo_threshold,
            iconq_model_id,
            routing_policy,
            allowed_rpus,
        )
        capacity_checkpoints = parse_capacity_checkpoints(cfg)
        poll_s: float = float(
            cfgu.getd(cfg, "autoscaling_config.capacity_poll_interval_s", 60.0)
        )

        # ── runner-specific ──────────────────────────────────────────────
        runner_cfg: dict = cfg.get("runner_config") or {}

        # Load a separate AWS credentials file if referenced.
        aws_cfg: dict = {}
        aws_config_path_str: Optional[str] = runner_cfg.get("aws_config_path")
        if aws_config_path_str is not None:
            aws_path = Path(aws_config_path_str)
            if not aws_path.is_absolute():
                aws_path = Path.cwd() / aws_path
            with open(aws_path) as _f:
                aws_cfg = yaml.safe_load(_f) or {}
            logging.info("Loaded AWS config from %s", aws_path)

        # Build provisioner config by merging aws_cfg (base defaults) with
        # any explicit provisioner fields in runner_config (take precedence).
        provisioner_raw: Optional[dict] = runner_cfg.get("provisioner")
        if provisioner_raw is not None:
            provisioner_config: Optional[dict] = {**aws_cfg, **provisioner_raw}
        elif aws_cfg:
            # No explicit provisioner section — infer redshift_serverless
            # from the aws_config alone.
            provisioner_config = {"type": "redshift_serverless", **aws_cfg}
        else:
            provisioner_config = None

        maxconns: int = int(runner_cfg.get("maxconns", 1000))
        max_threads_raw = runner_cfg.get("max_threads")
        max_threads: int | None = (
            int(max_threads_raw) if max_threads_raw is not None else None
        )

        return cls(
            workload_name=workload_name,
            routing_policy=routing_policy,
            schema_name=schema_name,
            managed_cluster_pool_config=mcp,
            autoscaling_policy=autoscaling_policy,
            provisioner_config=provisioner_config,
            maxconns=maxconns,
            closed_loop=closed_loop,
            config_path=config_path,
            capacity_checkpoints=capacity_checkpoints,
            capacity_poll_interval_s=poll_s,
            abs_start_time_start=abs_start_time_start,
            abs_start_time_end=abs_start_time_end,
            rescale_factor=rescale_factor,
            max_threads=max_threads,
        )

    # ------------------------------------------------------------------
    # Autoscaler callbacks
    # ------------------------------------------------------------------

    def _on_live_spin_up(self, reason: str, rpu: int) -> None:
        """Autoscaler callback: spin up a new cluster (non-blocking).

        Offloads the blocking provisioning to a background thread so
        that the asyncio event loop remains responsive for query
        scheduling.  The autoscaler is notified via
        :meth:`notify_cluster_ready` once the cluster is available.
        """
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor, self._do_spin_up, reason, rpu
        )
        self._pending_spin_ups.append(future)

    def _do_spin_up(self, reason: str, rpu: int) -> None:
        """Blocking spin-up executed in a thread-pool worker."""
        name = self.pool.request_spin_up(rpu, self._ts())
        ready_ts = self._ts()
        self.autoscaler.notify_cluster_ready(name, rpu, ready_ts)
        logging.info(
            "Autoscaler spin-up: %s (rpu=%d, cluster=%s)", reason, rpu, name
        )

    def _on_live_tear_down(self, cluster_name: str) -> None:
        """Autoscaler callback: tear down a cluster.

        Mirrors the simulator's ``_on_sim_tear_down`` guard: refuses to
        tear down the last routable cluster so the run can continue.
        """
        ready_names = self.pool.ready_cluster_names
        if len(ready_names) <= 1:
            logging.debug(
                "Skipping tear-down of %s — it is the last routable "
                "cluster.",
                cluster_name,
            )
            return
        self.pool.request_tear_down(cluster_name, self._ts())
        logging.info("Autoscaler tear-down: %s", cluster_name)

    # ------------------------------------------------------------------
    # End-of-run stats collection
    # ------------------------------------------------------------------

    def _collect_cluster_stats(
        self,
        cluster_name: str,
        conn_info: ClusterConnInfo,
        run_id: str,
    ) -> None:
        """Stats-collector callback invoked during cluster tear-down.

        Opens a fresh connection to *cluster_name*, queries the five
        Redshift system tables used by :class:`RunStatsCollector`, and
        writes each result as a Parquet file in the current run
        directory.

        This method is synchronous and may block for a significant
        amount of time (system tables can take minutes to flush).  It
        is invoked by :meth:`ManagedClusterPool._finalize_removal`
        while the cluster is still alive.
        """
        logging.info(
            "Collecting stats for cluster %s (run %s) ...",
            cluster_name,
            run_id,
        )
        try:
            conn = psycopg2.connect(
                host=conn_info.host,
                port=conn_info.port,
                user=conn_info.user,
                password=conn_info.password,
                dbname=conn_info.dbname,
                connection_factory=lambda dsn, **kw: ConnWithSetup(
                    dsn, search_path="public", **kw
                ),
            )
        except Exception:
            logging.exception(
                "Failed to connect to %s for stats collection.",
                cluster_name,
            )
            return

        try:
            # 1. sys_query_history — anchor table.
            history_df = self._query_to_parquet(
                conn,
                SYS_QUERY_HISTORY_QUERY.format(run_id),
                "sys_query_history",
                cluster_name,
            )
            if history_df is None or history_df.empty:
                logging.warning(
                    "No sys_query_history rows for cluster %s, run %s. "
                    "Skipping remaining system tables.",
                    cluster_name,
                    run_id,
                )
                return

            # Derive query-id and time ranges for the remaining tables.
            min_qid = int(history_df["query_id"].min())
            max_qid = int(history_df["query_id"].max())
            min_time = history_df["start_time"].min() - pd.Timedelta(minutes=1)
            max_time = history_df["end_time"].max() + pd.Timedelta(minutes=3)

            # 2–5. remaining system tables.
            for query_sql, table_name in [
                (
                    SYS_QUERY_EXPLAIN_QUERY.format(min_qid, max_qid),
                    "sys_query_explain",
                ),
                (
                    SYS_QUERY_DETAIL_QUERY.format(min_time, max_time),
                    "sys_query_detail",
                ),
                (
                    SYS_EXTERNAL_QUERY_DETAIL_QUERY.format(min_time, max_time),
                    "sys_external_query_detail",
                ),
                (
                    SYS_SERVERLESS_USAGE_QUERY.format(min_time, max_time),
                    "sys_serverless_usage",
                ),
            ]:
                self._query_to_parquet(
                    conn, query_sql, table_name, cluster_name
                )
        finally:
            try:
                conn.close()
            except Exception:
                pass

        logging.info("Stats collection complete for cluster %s.", cluster_name)

    def _query_to_parquet(
        self,
        conn,
        query: str,
        table_name: str,
        cluster_name: str,
    ) -> Optional[pd.DataFrame]:
        """Execute *query*, write result as Parquet, return the DataFrame."""
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                cols = (
                    [desc[0] for desc in cur.description]
                    if cur.description
                    else []
                )
            df = pd.DataFrame(rows, columns=cols)
            out_path = os.path.join(
                self._run_dir,
                f"{table_name}+{cluster_name}.parquet",
            )
            df.to_parquet(out_path, index=False)
            logging.info("Wrote %d rows to %s", len(df), out_path)
            return df
        except Exception:
            logging.exception(
                "Failed to query %s for cluster %s.",
                table_name,
                cluster_name,
            )
            return None

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def _ts(self, cast_to_int: bool = False) -> Union[int, float]:
        """
        Return the current UTC wall-clock time.

        Parameters:
            cast_to_int: If True, return the timestamp as an integer (seconds
                since epoch). If False, return as a float (with fractional
                seconds).
        """
        base = datetime.now(tz=timezone.utc).timestamp()
        if cast_to_int:
            return int(base)
        return base

    def _setup_run_directory(self):
        """
        Set up the run directory for storing results and other run metadata.
        """

        # Create a unique run directory based on the current timestamp.
        while True:
            run_id = str(self._ts(cast_to_int=True))
            run_dir = os.path.join(pu.get_runs_path(), f"{run_id}")
            if not os.path.exists(run_dir):
                break
        os.makedirs(run_dir, exist_ok=False)

        # Set up a log file inside the run directory.
        log_file_path = os.path.join(run_dir, "run.log")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # Remove all existing handlers (console and file alike) so that
        # log records are emitted only to the run-specific file and the
        # structured log — never to the caller's console.
        for h in list(logger.handlers):
            logger.removeHandler(h)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Prevent propagation to ancestor loggers (which might print to console).
        logger.propagate = False
        logging.info(f"Run directory created at {run_dir}")

        # Set up structured logging for this run.
        self._structured_handler = setup_structured_logging(out_dir=run_dir)

        # Dump the parameters of the run into a YAML file.
        d = {
            "run_id": run_id,
            "workload_name": self.workload_name,
            "num_queries": len(self.workload_df),
            "schema_name": self.schema.name,
            "search_path": self.schema.search_path,
            "initial_rpus": [
                self.pool.get_rpu(cn) for cn in self.pool.cluster_names
            ],
            "routing_policy": self.routing_policy_config,
            "maxconns": self.maxconns,
            "closed_loop": self.closed_loop,
        }

        with open(os.path.join(run_dir, "run_params.yml"), "w") as f:
            yaml.dump(d, f, sort_keys=False)
        logging.info(
            f"Run parameters saved to {os.path.join(run_dir, 'run_params.yml')}"
        )

        # Keep a verbatim copy of the config file used for this run.
        if self.config_path is not None:
            shutil.copy2(
                self.config_path,
                os.path.join(run_dir, "runner_config.yml"),
            )
            logging.info(
                f"Config file copied to "
                f"{os.path.join(run_dir, 'runner_config.yml')}"
            )

        return run_id, run_dir

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
            conn = self.pool.getconn(cluster_name)
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
            self.pool.putconn(cluster_name, conn)
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
                    "latency_s": latency_s,
                }
            )
        self._pbar.update(1)

    async def _run_query_async(
        self,
        run_id: str,
        async_reference_ts: float,
        rel_start_time_s: float,
        query_id: str,
        query_text: str,
        query_text_id: str,
    ) -> None:
        """
        Run a single query asynchronously, waiting until its scheduled start.

        Routing is performed **after** the sleep so that the routing
        decision reflects the live pool state at the query's actual
        arrival time — not the pool state at the start of the run.

        Parameters:
            run_id: ID of the current run.
            async_reference_ts: Reference timestamp for scheduling.
            rel_start_time_s: Relative start time in seconds from the
                reference timestamp.  Ignored in closed-loop mode
                (pass 0).
            query_id: ID of the query.
            query_text: SQL text of the query.
            query_text_id: The query_text_id for this query (used for
                routing).
        """
        now = self._ts()
        scheduled_time = async_reference_ts + rel_start_time_s
        delay = scheduled_time - now
        logging.info(
            f"Query {query_id} scheduled to start at t={scheduled_time:.2f}s "
            f"(in {delay:.2f}s)"
        )
        if delay > 0:
            await asyncio.sleep(delay)

        # ── Route at arrival time ────────────────────────────────────
        route_start_ts = self._ts()
        result = self.router.route_query_with_predictions(
            query_id=query_id,
            query_text_id=query_text_id,
        )
        cluster_name = result.cluster_name
        route_end_ts = self._ts()

        if _has_structured():
            emit_structured(
                {
                    "timestamp": self._ts(),
                    "source": "WorkloadRunner",
                    "event_type": "query_routed",
                    "run_id": run_id,
                    "query_id": query_id,
                    "query_text_id": query_text_id,
                    "cluster_name": cluster_name,
                    "routing_time_s": route_end_ts - route_start_ts,
                }
            )

        # ── Register immediately (before autoscaler) ─────────────────
        # Registering the query in the pool *before* notifying the
        # autoscaler guarantees the cluster cannot be fully torn down
        # while this query is active.  If the autoscaler decides to
        # drain the cluster in response to the routing result, the
        # active-query count prevents premature removal.
        now = self._ts()
        self.router.on_query_start(
            query_id=query_id,
            cluster_name=cluster_name,
            query_text_id=query_text_id,
            start_time_s=now,
        )

        # Feed routing result to the autoscaler (sees updated pool state).
        self.autoscaler.on_routing_result(result, self._ts())

        # ── Execute ──────────────────────────────────────────────────
        try:
            fn = partial(
                self._run_query_sync, run_id, query_id, query_text, cluster_name
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, fn)
        finally:
            now = self._ts()
            self.router.on_query_finish(
                query_id=query_id,
                cluster_name=cluster_name,
                current_time_s=now,
            )
            self.autoscaler.on_query_complete(query_id, cluster_name, now)

    # ------------------------------------------------------------------
    # Periodic autoscaler tick (mirrors simulator's capacity_poll_interval)
    # ------------------------------------------------------------------

    async def _autoscaler_tick_loop(self, async_reference_ts: float) -> None:
        """Periodically call ``autoscaler.on_time_advance``.

        Ensures that idle-period tear-downs and other time-based policies
        fire even when no queries are arriving or completing.

        The loop runs until cancelled (via ``asyncio.Task.cancel``).
        """
        next_tick_rel = self._capacity_poll_interval_s
        while True:
            target = async_reference_ts + next_tick_rel
            delay = target - self._ts()
            if delay > 0:
                await asyncio.sleep(delay)
            self.autoscaler.on_time_advance(self._ts())
            next_tick_rel += self._capacity_poll_interval_s

    async def run(self) -> str:
        """
        Run the queries from the workload file.

        Returns:
            The ID of the run.
        """
        run_id, run_dir = self._setup_run_directory()
        self._run_dir = run_dir
        print(f"Run started with ID {run_id}.")

        # Register the stats-collection callback so that each cluster's
        # system tables are captured before the provisioner tears it down.
        self.pool.set_stats_collector(self._collect_cluster_stats, run_id)

        if _has_structured():
            emit_structured(
                {
                    "timestamp": self._ts(),
                    "source": "WorkloadRunner",
                    "event_type": "run_start",
                    "run_id": run_id,
                    "workload_name": self.workload_name,
                    "num_queries": len(self.workload_df),
                    "routing_policy": self.routing_policy_name,
                    "closed_loop": self.closed_loop,
                }
            )

        # Add a 30-second buffer between the reference timestamp and the first
        # query's scheduled start time for setup.
        async_reference_ts = self._ts() + 30
        logging.info(f"Async reference timestamp: {async_reference_ts:.2f}s")

        tasks: list[asyncio.Task] = []

        # Schedule capacity checkpoint reconciliations.
        for cp in self.autoscaler.checkpoints:
            tasks.append(
                asyncio.ensure_future(
                    self._reconcile_checkpoint_async(cp, async_reference_ts)
                )
            )

        # Start a periodic autoscaler tick so that time-based policies
        # (e.g. idle-period tear-down) fire even during quiet periods.
        tick_task = asyncio.ensure_future(
            self._autoscaler_tick_loop(async_reference_ts)
        )

        self._pbar = tqdm(total=len(self.workload_df), desc="Queries", unit="q")

        try:
            for _, row in self.workload_df.iterrows():
                query_id = str(row["query_id"])
                query_text_id = str(row["query_text_id"])
                rel_start_time_s = row["rel_start_time_s"]

                # Resolve the SQL text from the registry (not timing-sensitive).
                query_text = QueryTextRegistry.get(
                    self.schema.name, query_text_id
                )
                if query_text is None:
                    logging.warning(
                        f"No query text found for schema '{self.schema.name}', "
                        f"query_text_id '{query_text_id}'. Skipping query {query_id}."
                    )
                    continue

                if not self.closed_loop:
                    task = asyncio.ensure_future(
                        self._run_query_async(
                            run_id,
                            async_reference_ts,
                            rel_start_time_s,
                            query_id,
                            query_text,
                            query_text_id=query_text_id,
                        )
                    )
                    tasks.append(task)
                else:
                    # In closed loop, wait for each query to finish before
                    # starting the next.  Ignore rel_start_time_s.
                    await self._run_query_async(
                        run_id,
                        async_reference_ts,
                        0,
                        query_id,
                        query_text,
                        query_text_id=query_text_id,
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
            # Stop the periodic tick.
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass

            self._pbar.close()

            # Graceful cleanup: tear down every remaining READY cluster.
            # request_tear_down → _finalize_removal is synchronous and
            # may block (stats collection + provisioner API call), so
            # we dispatch each call via the default thread-pool executor.
            loop = asyncio.get_running_loop()
            remaining = list(self.pool.ready_cluster_names)
            for cn in remaining:
                try:
                    await loop.run_in_executor(
                        self._executor,
                        partial(
                            self.pool.request_tear_down,
                            cn,
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
                    "run_id": run_id,
                    "workload_name": self.workload_name,
                }
            )

        # Finalize structured log (consolidate shards).
        if (
            hasattr(self, "_structured_handler")
            and self._structured_handler is not None
        ):
            self._structured_handler.finalize()

        return run_id


if __name__ == "__main__":
    cfg, config_path = cfgu.load_config_from_cli(
        "Run queries from a workload using a YAML config file.",
    )
    qr = WorkloadRunner.from_config_dict(cfg, config_path=config_path)
    asyncio.run(qr.run())
