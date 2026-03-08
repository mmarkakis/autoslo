import argparse
import asyncio
import logging
import os
import shutil
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional, Union

import yaml
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.capacity.autoscaler import Autoscaler
from autoslo.capacity.autoscaling_policy import (
    AutoscalingPolicy,
    CapacityCheckpoint,
    NoOpPolicy,
)
from autoslo.capacity.cluster_provisioner import SimulatedProvisioner
from autoslo.capacity.headroom_policy import HeadroomPolicy
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.managed_cluster_pool import (
    ManagedClusterPool,
    ManagedClusterPoolConfig,
)
from autoslo.routing.model_policy import ModelPolicy
from autoslo.routing.router import Router
from autoslo.routing.routing_policy import RoundRobinPolicy, RoutingPolicy
from autoslo.utils.structured_log import (
    LOGGER_NAME,
    emit_structured,
    setup_structured_logging,
)
from autoslo.workload_definition.query import QueryTextId, SloMetric

_has_structured = lambda: bool(
    logging.getLogger(LOGGER_NAME).handlers
)  # noqa: E731
from autoslo.workload_definition.workload import Workload
from autoslo.workload_definition.query_text_registry import QueryTextRegistry
from autoslo.workload_definition.schema import Schema


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
    ):
        self.config_path = Path(config_path) if config_path else None
        self.workload_name = workload_name
        self.closed_loop = closed_loop
        self.maxconns = maxconns

        # Load workload.
        if workload_name.startswith("benchmarking_workload_"):
            workload_path = os.path.join(
                pu.get_data_path(),
                "benchmarking_workloads",
                f"{workload_name}.parquet",
            )
        elif workload_name.startswith("interference_"):
            workload_path = os.path.join(
                pu.get_data_path(),
                "interference_workloads",
                f"{workload_name}.parquet",
            )
        else:
            workload_path = os.path.join(
                pu.get_data_path(),
                "chunks",
                f"{workload_name}",
                "chunk_workload.parquet",
            )
        if not os.path.exists(workload_path):
            raise FileNotFoundError(
                f"Workload file {workload_path} does not exist."
            )
        self.workload = Workload.load(workload_path)
        self.workload.set_rel_start_times_from_zero()
        self.workload_df = self.workload.df
        self.schema = Schema.load(
            schema_name or self.workload.schema_name,
        )

        # Build provisioner.
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

    # ------------------------------------------------------------------
    # Async checkpoint reconciliation (live runner)
    # ------------------------------------------------------------------

    async def _reconcile_checkpoint_async(
        self,
        checkpoint: CapacityCheckpoint,
        async_reference_ts: float,
    ) -> None:
        """Wait until *checkpoint.time_s* elapses, then reconcile."""
        target = async_reference_ts + checkpoint.time_s
        delay = target - self._async_ts()
        if delay > 0:
            await asyncio.sleep(delay)
        self.autoscaler.reconcile_checkpoints_up_to(checkpoint.time_s)

    # ------------------------------------------------------------------
    # Factory: create from YAML config (aligned with WorkloadSimulator)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "WorkloadRunner":
        """Create a :class:`WorkloadRunner` from a YAML config file.

        Reads the same nested-section format as
        :meth:`WorkloadSimulator.from_config` with an additional optional
        ``runner_config`` section for live-execution settings.

        Sections
        --------
        basic_config        : workload_name, schema_name, iconq_model_id
        slo_config          : slo_s, slo_metric, slo_threshold,
                              slo_dict_filename
        routing_config      : routing_policy ("model" | "round_robin")
        managed_cluster_pool_config : initial_rpus, allowed_rpu_sizes,
                                spin_up_delay_s
        autoscaling_config  : autoscaling_policy ("headroom" | "noop"),
                              eta_crit, idle_periods_before_tear_down,
                              capacity_poll_interval_s,
                              min_cluster_lifetime_s
        runner_config       : provisioner, maxconns, closed_loop
        """
        path = Path(config_path)
        with open(path) as f:
            cfg = yaml.safe_load(f)

        # Helper: read from named section first, fall back to root.
        def _s(section_key: str, key: str, default=None):
            section = cfg.get(section_key)
            if section and key in section:
                return section[key]
            return cfg.get(key, default)

        # ── basic ────────────────────────────────────────────────────────
        workload_name: str = _s("basic_config", "workload_name")
        schema_name: Optional[str] = _s("basic_config", "schema_name")
        iconq_model_id: Optional[str] = _s("basic_config", "iconq_model_id")

        # ── SLO ──────────────────────────────────────────────────────────
        slo_s: float = _s("slo_config", "slo_s", 10.0)
        raw_metric: str = _s("slo_config", "slo_metric", "relative")
        slo_metric = SloMetric(raw_metric)
        slo_dict_filename: Optional[str] = _s("slo_config", "slo_dict_filename")
        slo_resolver = SloResolver(slo_s, slo_dict_filename)

        # ── routing policy ───────────────────────────────────────────────
        routing_cfg: dict = cfg.get("routing_config") or {}
        policy_type: str = routing_cfg.get(
            "routing_policy", cfg.get("routing_policy", "model")
        )

        if policy_type == "model":
            routing_policy: RoutingPolicy = ModelPolicy(
                iconq_model_id=iconq_model_id,
                default_slo_s=slo_s,
                slo_overrides=slo_resolver.slo_dict,
                slo_metric=slo_metric,
            )
        elif policy_type == "round_robin":
            routing_policy = RoundRobinPolicy()
        else:
            raise ValueError(
                f"Unknown routing_policy {policy_type!r}. "
                "Expected one of: 'model', 'round_robin'."
            )

        # ── cluster pool ─────────────────────────────────────────────────
        mcp_raw: Optional[dict] = cfg.get("managed_cluster_pool_config")
        mcp: Optional[ManagedClusterPoolConfig] = None
        if mcp_raw is not None:
            mcp_raw = dict(mcp_raw)  # shallow copy
            if "initial_rpus" in mcp_raw and isinstance(
                mcp_raw["initial_rpus"], list
            ):
                mcp_raw["initial_rpus"] = tuple(mcp_raw["initial_rpus"])
            if "allowed_rpu_sizes" in mcp_raw and isinstance(
                mcp_raw["allowed_rpu_sizes"], list
            ):
                mcp_raw["allowed_rpu_sizes"] = tuple(
                    mcp_raw["allowed_rpu_sizes"]
                )
            mcp = ManagedClusterPoolConfig(**mcp_raw)

        # ── autoscaling ──────────────────────────────────────────────────
        autoscaling_policy_type: str = _s(
            "autoscaling_config", "autoscaling_policy", "headroom"
        )
        allowed_rpus: list[int] = list(
            mcp.allowed_rpu_sizes
            if mcp is not None
            else ManagedClusterPoolConfig().allowed_rpu_sizes
        )

        if autoscaling_policy_type == "headroom":
            autoscaling_policy: AutoscalingPolicy = HeadroomPolicy(
                slo_resolver=slo_resolver,
                slo_metric=slo_metric,
                eta_crit=float(_s("autoscaling_config", "eta_crit", 0.1)),
                idle_periods_before_tear_down=int(
                    _s(
                        "autoscaling_config",
                        "idle_periods_before_tear_down",
                        5,
                    )
                ),
                min_cluster_lifetime_s=float(
                    _s(
                        "autoscaling_config",
                        "min_cluster_lifetime_s",
                        1200.0,
                    )
                ),
                allowed_rpu_sizes=allowed_rpus,
                iconq_model=(
                    IconqModel.load(iconq_model_id) if iconq_model_id else None
                ),
            )
        elif autoscaling_policy_type == "noop":
            autoscaling_policy = NoOpPolicy()
        else:
            raise ValueError(
                f"Unknown autoscaling_policy {autoscaling_policy_type!r}. "
                "Expected one of: 'headroom', 'noop'."
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
        closed_loop: bool = bool(runner_cfg.get("closed_loop", False))

        # ── capacity checkpoints ─────────────────────────────────────────
        raw_checkpoints: list[dict] = (
            _s("autoscaling_config", "capacity_checkpoints", []) or []
        )
        capacity_checkpoints = [
            CapacityCheckpoint(
                time_s=float(cp["time_s"]),
                min_rpus=tuple(cp["min_rpus"]),
            )
            for cp in raw_checkpoints
        ]

        poll_s: float = float(
            _s("autoscaling_config", "capacity_poll_interval_s", 60.0)
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
        )

    # ------------------------------------------------------------------
    # Autoscaler callbacks
    # ------------------------------------------------------------------

    def _on_live_spin_up(self, reason: str, rpu: int) -> None:
        """Autoscaler callback: spin up a new cluster."""
        name = self.pool.request_spin_up(rpu, self._ts())
        logging.info(
            "Autoscaler spin-up: %s (rpu=%d, cluster=%s)", reason, rpu, name
        )

    def _on_live_tear_down(self, cluster_name: str) -> None:
        """Autoscaler callback: tear down a cluster."""
        self.pool.request_tear_down(cluster_name, self._ts())
        logging.info("Autoscaler tear-down: %s", cluster_name)

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def _ts(self, cast_to_int: bool = False) -> Union[int, float]:
        """
        Return the current UTC wall-clock time.

        All timing in the runner uses wall-clock time for consistency.

        Parameters:
            cast_to_int: If True, return the timestamp as an integer (seconds
                since epoch). If False, return as a float (with fractional
                seconds).
        """
        base = datetime.now(tz=timezone.utc).timestamp()
        if cast_to_int:
            return int(base)
        return base

    def _async_ts(self) -> float:
        """Return the current wall-clock time (UTC, fractional seconds).

        This deliberately uses the **same clock** as :meth:`_ts` (wall-clock
        UTC) rather than the event-loop monotonic clock.  Mixing two clocks
        (``loop.time()`` vs ``datetime.now()``) leads to meaningless deltas
        when NTP adjustments occur.  The trade-off is that a large clock jump
        could distort scheduling, but in practice NTP steps are sub-second.
        """
        return datetime.now(tz=timezone.utc).timestamp()

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
        # Remove file handlers from previous runs but keep any console
        # handlers that were set up by the caller (e.g. basicConfig in
        # __main__).  This avoids silently swallowing log output when
        # the runner is embedded in a larger script.
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler):
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
        conn = self.pool.conn_pool(cluster_name).getconn()
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
            try:
                self.pool.conn_pool(cluster_name).putconn(conn)
            except Exception:
                pass
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
        now = self._async_ts()
        scheduled_time = async_reference_ts + rel_start_time_s
        delay = scheduled_time - now
        logging.info(
            f"Query {query_id} scheduled to start at t={scheduled_time:.2f}s "
            f"(in {delay:.2f}s)"
        )
        if delay > 0:
            await asyncio.sleep(delay)

        # ── Route at arrival time ────────────────────────────────────
        route_start_ts = self._async_ts()
        result = self.router.route_query_with_predictions(
            query_id=query_id,
            query_text_id=query_text_id,
        )
        cluster_name = result.cluster_name
        route_end_ts = self._async_ts()

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
        now = self._async_ts()
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
            await loop.run_in_executor(None, fn)
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

    async def _autoscaler_tick_loop(
        self, async_reference_ts: float
    ) -> None:
        """Periodically call ``autoscaler.on_time_advance``.

        Ensures that idle-period tear-downs and other time-based policies
        fire even when no queries are arriving or completing.

        The loop runs until cancelled (via ``asyncio.Task.cancel``).
        """
        next_tick_rel = self._capacity_poll_interval_s
        while True:
            target = async_reference_ts + next_tick_rel
            delay = target - self._async_ts()
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
        print(f"Run started with ID {run_id}.")
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

        async_reference_ts = self._async_ts()
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
                query_id = row["query_id"]
                query_text_id = str(row["query_text_id"])
                schema_name = str(row.get("schema_name", ""))
                rel_start_time_s = row["rel_start_time_s"]

                # Resolve the SQL text from the registry (not timing-sensitive).
                query_text = QueryTextRegistry.get(schema_name, query_text_id)
                if query_text is None:
                    logging.warning(
                        f"No query text found for schema '{schema_name}', "
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

            await asyncio.gather(*tasks)
        finally:
            # Stop the periodic tick.
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass

            self._pbar.close()

            # Graceful cleanup: close all connection pools so that
            # in-progress Redshift connections are released.
            try:
                self.pool.destroy_all_conn_pools()
            except Exception:
                logging.exception("Error destroying connection pools.")

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
    parser = argparse.ArgumentParser(
        description="Run queries from a workload using a YAML config file."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML config file (e.g. data/__run_configs/test.yml).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    qr = WorkloadRunner.from_config(args.config)
    asyncio.run(qr.run())
