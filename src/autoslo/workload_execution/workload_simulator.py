import heapq
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import yaml
from filelock import FileLock
from tqdm import tqdm

import autoslo.utils.config as cfgu
import autoslo.utils.paths as pu
from autoslo.utils.yaml_helpers import dump_config
from autoslo.blueprint_selection import log_timeline_builder
from autoslo.blueprint_selection.query_timeline_visualizer_2 import (
    export_gantt_video,
    render_gantt_scrubber,
)
from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.capacity.autoscaler import Autoscaler
from autoslo.capacity.autoscaling_policy import (
    AutoscalingPolicy,
    CapacityCheckpoint,
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
from autoslo.routing.routing_core import (
    PlacementScore,
    RoutingCore,
    RoutingResult,
)
from autoslo.routing.routing_policy import RoutingPolicy
from autoslo.utils.billing import Billing
from autoslo.utils.structured_log import (
    StructuredLogHandler,
    emit_structured,
    setup_structured_logging,
)
from autoslo.workload_definition.query import Query, QueryTextId, SloMetric
from autoslo.workload_definition.workload import Workload

if TYPE_CHECKING:
    from autoslo.workload_definition.redset_workload import (
        RedsetWorkloadSamplingSpec,
    )

logger = logging.getLogger(__name__)


class WorkloadSimulator:
    """
    Overall strategy:
    Phase 1: As each query comes in, route it to some endpoint to minimize the
        number of SLO violations. Prefer active endpoints rather than starting
        new ones.

    Phase 2: At the end of the workload, we can now trade some (bounded) amount
        of SLO violations for a lower execution cost. We can do this by
        re-routing some queries to different endpoints and replying from that
        point on.

    """

    TOLERANCE_FOR_SLO_VIOLATION_AMOUNT_OPTIMIZATION_S = 1e-4

    def __init__(
        self,
        workload_name: str,
        routing_policy: RoutingPolicy,
        slo_s: float,
        schema_name: str,
        iconq_model_id: str,
        slo_metric: SloMetric = SloMetric.RELATIVE,
        slo_threshold: float = 0.0,
        slo_dict_filename: Optional[str] = None,
        verbose: bool = True,
        export_video: bool = False,
        video_frame_duration: float = 1.0,
        simulator_run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        overwrite_experiment: bool = False,
        managed_cluster_pool_config: Optional[ManagedClusterPoolConfig] = None,
        autoscaling_policy: Optional[AutoscalingPolicy] = None,
        capacity_poll_interval_s: float = 60.0,
        capacity_checkpoints: list[CapacityCheckpoint] | None = None,
        abs_start_time_start: str | None = None,
        abs_start_time_end: str | None = None,
        rescale_factor: float | None = None,
        closed_loop: bool = False,
        out_dir: str | Path | None = None,
        workload: "Workload | None" = None,
        iconq_model: "IconqModel | None" = None,
    ):
        self._workload_name = workload_name
        self._iconq_model_id = iconq_model_id
        self._iconq_model = (
            iconq_model
            if iconq_model is not None
            else IconqModel.load(iconq_model_id)
        )
        self._slo_s = slo_s
        self._slo_dict_filename = slo_dict_filename
        self._slo_resolver = SloResolver(slo_s, slo_dict_filename)
        self._closed_loop = closed_loop

        # Store the routing policy (caller is responsible for constructing it).
        self._routing_policy = routing_policy

        # Dynamic provisioning parameters.
        self._dynamic_cluster_config = (
            managed_cluster_pool_config
            if managed_cluster_pool_config is not None
            else ManagedClusterPoolConfig()
        )
        self._autoscaling_policy = autoscaling_policy
        self._cc_poll_interval_s = capacity_poll_interval_s
        self._capacity_checkpoints = capacity_checkpoints or []

        self._slo_metric = slo_metric
        self._slo_threshold = slo_threshold
        self._verbose = verbose
        self._export_video = export_video
        self._video_frame_duration = video_frame_duration
        self._simulator_run_id = simulator_run_id
        self._experiment_name = experiment_name
        self._overwrite_experiment = overwrite_experiment
        self._schema_name = schema_name
        self._workload: Workload
        if workload is not None:
            # Caller-provided workload (e.g. policy tuner) — use as-is.
            self._workload = workload
        else:
            if workload_name.startswith("redset"):
                from autoslo.workload_definition.redset_workload import (
                    RedsetWorkload,
                )  # noqa: PLC0415

                self._workload = RedsetWorkload.load(workload_name)
            else:
                self._workload = Workload(
                    workload_name=workload_name,
                    schema_name=schema_name,
                )
            if abs_start_time_start is not None or abs_start_time_end is not None:
                self._workload.slice_by_abs_time(
                    abs_start_time_start, abs_start_time_end
                )
            self._workload.set_rel_start_times_from_zero()
            if rescale_factor is not None:
                self._workload.rescale_rel_start_times(rescale_factor)

        self._run_id = simulator_run_id or str(
            int(datetime.now().timestamp() * 1000)
        )

        self._seed: Optional[int] = None  # populated in simulate_one

        # Optional caller-provided output directory override.
        self._out_dir_override: str | Path | None = out_dir

        # Setup the outputs directory.
        self._out_dir = self._make_out_dir(self._run_id)
        self._write_config_yml()

        # Set up structured logging handler.
        self._structured_handler: Optional[StructuredLogHandler] = None
        if self._verbose:
            self._structured_handler = setup_structured_logging(
                out_dir=self._out_dir,
            )

        # Dynamic provisioning state (always initialised by _init_dynamic_clusters).
        self._pending_events: list[tuple[float, int, str]] = []
        self._event_counter: int = 0
        self._current_sim_time_s: float = 0.0
        self._next_tick_time_s: float = 0.0

        # External latency tracking (replaces mutation of Query.latency_s).
        self._predicted_latencies: dict[str, float] = {}
        """query_id → current best latency estimate (monotonically increasing)."""

        self._query_to_cluster_name: dict[str, str] = {}
        """query_id → cluster_name for queries currently tracked."""

        self._init_dynamic_clusters()

    # ------------------------------------------------------------------
    # Factory: create from YAML config (aligned with WorkloadRunner)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, config_path: str | Path, **overrides: object
    ) -> "WorkloadSimulator":
        """Create a :class:`WorkloadSimulator` from a YAML config file.

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
        cfg = cfgu.apply_overrides(cfg, overrides)
        return cls.from_config_dict(cfg)

    @classmethod
    def from_config_dict(
        cls, cfg: dict, workload: "Workload | None" = None,
    ) -> "WorkloadSimulator":
        """Create a :class:`WorkloadSimulator` from an already-loaded config dict.

        Parameters
        ----------
        cfg :
            Parsed YAML configuration dictionary.
        workload :
            When provided, this pre-built :class:`Workload` is used
            directly instead of loading one from disk.  Time-slicing,
            rescaling, and ``set_rel_start_times_from_zero`` are
            **skipped** — the caller is responsible for preparing the
            workload beforehand.

        Nested sections
        ---------------
        workload_config     : workload_name, abs_start_time_start,
                              abs_start_time_end, rescale_factor,
                              closed_loop
        basic_config        : schema_name, experiment_name,
                              simulator_run_id, overwrite_experiment,
                              iconq_model_id
        slo_config          : slo_s, slo_metric, slo_threshold, slo_dict_filename
        routing_config      : routing_policy  ("model" | "round_robin" | "cache_aware"),
        managed_cluster_pool_config : initial_rpus, allowed_rpu_sizes,
                                spin_up_delay_s
        autoscaling_config  : autoscaling_policy ("headroom" | "noop"),
                              eta_crit, idle_periods_before_tear_down,
                              capacity_poll_interval_s, min_cluster_lifetime_s
        output_config       : verbose, export_video, video_frame_duration
        """

        # ── basic ────────────────────────────────────────────────────────────
        schema_name: str = cfgu.cfg_get(
            cfg, "basic_config", "schema_name", required=True
        )
        experiment_name: Optional[str] = cfgu.cfg_get(
            cfg, "basic_config", "experiment_name"
        )
        simulator_run_id: Optional[str] = cfgu.cfg_get(
            cfg, "basic_config", "simulator_run_id"
        )
        overwrite_experiment: bool = cfgu.cfg_get(
            cfg, "basic_config", "overwrite_experiment", False
        )
        iconq_model_id: str = cfgu.cfg_get(
            cfg, "basic_config", "iconq_model_id", required=True
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

        # ── SLO ──────────────────────────────────────────────────────────────
        slo_s: float = cfgu.cfg_get(cfg, "slo_config", "slo_s", 10.0)
        slo_metric = SloMetric(
            cfgu.cfg_get(cfg, "slo_config", "slo_metric", "relative")
        )
        slo_threshold: float = float(
            cfgu.cfg_get(cfg, "slo_config", "slo_threshold", 0.0)
        )
        slo_dict_filename: Optional[str] = cfgu.cfg_get(
            cfg, "slo_config", "slo_dict_filename"
        )
        slo_resolver = SloResolver(slo_s, slo_dict_filename)

        # ── Load the IconqModel once and share across all consumers ──────────
        _iconq_model: IconqModel | None = (
            IconqModel.load(iconq_model_id) if iconq_model_id else None
        )

        # ── shared policy / pool construction ────────────────────────────────
        routing_policy = cfgu.build_routing_policy(
            cfg,
            iconq_model_id,
            slo_s,
            slo_resolver,
            slo_metric,
            iconq_model=_iconq_model,
        )
        mcp = cfgu.build_managed_cluster_pool_config(cfg)
        allowed_rpus: list[int] = list(
            mcp.allowed_rpu_sizes
            if mcp is not None
            else ManagedClusterPoolConfig().allowed_rpu_sizes
        )
        autoscaling_policy = cfgu.build_autoscaling_policy(
            cfg,
            slo_resolver,
            slo_metric,
            slo_threshold,
            iconq_model_id,
            routing_policy,
            allowed_rpus,
            iconq_model=_iconq_model,
        )
        poll_s: float = float(
            cfgu.cfg_get(
                cfg, "autoscaling_config", "capacity_poll_interval_s", 60.0
            )
        )
        capacity_checkpoints = cfgu.parse_capacity_checkpoints(cfg)

        # ── output (simulator-specific) ──────────────────────────────────────
        verbose: bool = cfgu.cfg_get(cfg, "output_config", "verbose", False)
        export_video: bool = cfgu.cfg_get(
            cfg, "output_config", "export_video", False
        )
        video_frame_duration: float = cfgu.cfg_get(
            cfg, "output_config", "video_frame_duration", 1.0
        )
        out_dir: str | None = cfgu.cfg_get(cfg, "output_config", "out_dir")

        return cls(
            workload_name=workload_name,
            routing_policy=routing_policy,
            slo_s=slo_s,
            schema_name=schema_name,
            iconq_model_id=iconq_model_id,
            slo_metric=slo_metric,
            slo_threshold=slo_threshold,
            slo_dict_filename=slo_dict_filename,
            verbose=verbose,
            export_video=export_video,
            video_frame_duration=video_frame_duration,
            simulator_run_id=simulator_run_id,
            experiment_name=experiment_name,
            overwrite_experiment=overwrite_experiment,
            managed_cluster_pool_config=mcp,
            autoscaling_policy=autoscaling_policy,
            capacity_poll_interval_s=poll_s,
            capacity_checkpoints=capacity_checkpoints,
            abs_start_time_start=abs_start_time_start,
            abs_start_time_end=abs_start_time_end,
            rescale_factor=rescale_factor,
            closed_loop=closed_loop,
            out_dir=out_dir,
            workload=workload,
            iconq_model=_iconq_model,
        )

    # ------------------------------------------------------------------
    # Post-routing scoring (for non-model policies)
    # ------------------------------------------------------------------

    def _score_with_model(
        self,
        query_id: str,
        query_text_id: QueryTextId,
        cluster_name: str,
        start_time_s: float,
    ) -> tuple[Optional[PlacementScore], Optional[Query]]:
        """Score a routing decision post-hoc using the scoring model.

        Called when the routing policy returned a :class:`RoutingResult` with
        ``score=None`` (e.g. :class:`RoundRobinPolicy`).
        Delegates to :meth:`RoutingCore.score_query_on_clusters` for the single
        chosen cluster.

        Returns ``(PlacementScore, featurised_tracking_query)`` on success, or
        ``(None, None)`` if any step fails (cluster not in pool, no predictions).
        """
        scores, incoming, _ = RoutingCore.score_query_on_clusters(
            iconq_model=self._iconq_model,
            pool=self._pool,
            query_id=query_id,
            query_text_id=query_text_id,
            start_time_s=start_time_s,
            slo_resolver=self._slo_resolver,
            slo_metric=self._slo_metric,
            current_latencies=self._predicted_latencies,
            cluster_names=[cluster_name],
        )
        score = scores.get(cluster_name)
        if score is None:
            logger.warning(
                "_score_with_model: no score produced for cluster %s; skipping.",
                cluster_name,
            )
            return None, None
        return score, incoming

    # ------------------------------------------------------------------
    # helper: build/return the output directory path
    # ------------------------------------------------------------------
    def _make_out_dir(self, run_id: str) -> str:
        if self._out_dir_override is not None:
            out_dir = os.path.join(str(self._out_dir_override), run_id)
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        if self._experiment_name:
            experiment_dir = os.path.join(
                pu.get_data_path(), "simulator_runs", self._experiment_name
            )
            if os.path.exists(experiment_dir) and self._overwrite_experiment:
                shutil.rmtree(experiment_dir)
            out_dir = os.path.join(
                experiment_dir,
                run_id,
            )
        else:
            out_dir = os.path.join(pu.get_data_path(), "simulator_runs", run_id)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _write_config_yml(self) -> None:
        config_out_path = os.path.join(self._out_dir, "config.yml")
        with open(config_out_path, "w") as f:
            d = {
                "run_id": self._run_id,
                "experiment_name": self._experiment_name,
                "workload_name": self._workload_name,
                "routing_policy_type": type(self._routing_policy).__name__,
                "iconq_model_id": self._iconq_model_id,
                "slo_s": self._slo_s,
                "slo_dict_filename": self._slo_dict_filename,
                "slo_dict": self._slo_resolver.slo_dict,
                "slo_metric": self._slo_metric.value,
                "slo_threshold": self._slo_threshold,
                "verbose": self._verbose,
                "export_video": self._export_video,
                "video_frame_duration": self._video_frame_duration,
                "seed": self._seed,
                "closed_loop": self._closed_loop,
            }
            dump_config(d, f, sort_keys=False)

    def reset(self, simulator_run_id: Optional[str] = None) -> None:
        """
        Reset the simulator state for a new run, reusing the model and workload.
        This allows multiple samples to be run without reloading heavy objects.

        Parameters:
            simulator_run_id: Optional run ID for the new run. If None, generates
                a new timestamp-based ID.
        """
        self._run_id = simulator_run_id or str(
            int(datetime.now().timestamp() * 1000)
        )
        self._seed = None
        self._out_dir = self._make_out_dir(self._run_id)
        self._write_config_yml()

        # Reset structured logging.
        if self._structured_handler is not None:
            self._structured_handler.reset(out_dir=self._out_dir)
        elif self._verbose:
            self._structured_handler = setup_structured_logging(
                out_dir=self._out_dir,
            )

        # Reset per-run bookkeeping.
        self._init_dynamic_clusters()

    def _log_if_verbose(self, d: dict) -> None:
        """Emit a structured log record (if verbose logging is enabled).

        Adds ``source='simulator'`` automatically if not already set.

        Parameters:
            d: A dictionary containing the log entry data. Must include at
                least ``timestamp`` and ``event_type``.
        """
        if not self._verbose:
            return
        d.setdefault("source", "simulator")
        emit_structured(d)

    # ------------------------------------------------------------------
    # Dynamic provisioning helpers
    # ------------------------------------------------------------------

    def _init_dynamic_clusters(self) -> None:
        """Set up provisioner, pool, router, controller, and initial clusters."""
        config = self._dynamic_cluster_config

        # Provisioner
        self._provisioner: SimulatedProvisioner = SimulatedProvisioner(
            spin_up_delay_s=config.spin_up_delay_s
        )

        # Pool (no initial clusters via constructor — we add them below
        # with instant on_cluster_ready to bypass spin-up delay).
        self._pool: ManagedClusterPool = ManagedClusterPool(
            provisioner=self._provisioner,
            config=ManagedClusterPoolConfig(
                initial_rpus=(),
                allowed_rpu_sizes=config.allowed_rpu_sizes,
            ),
        )

        # Activate initial clusters immediately (no spin-up delay).
        for rpu in config.initial_rpus:
            name = self._pool.request_spin_up(rpu, 0.0)
            self._pool.on_cluster_ready(name, 0.0)

        # Create the Router (invokes policy.on_attach to wire up RPU lookup).
        self._router: Router = Router(
            policy=self._routing_policy,
            pool=self._pool,
        )

        # If we have a standalone scoring model (not attached via the policy),
        # wire up the RPU lookup so interaction featurisation works correctly.
        if self._iconq_model is not None and not isinstance(
            self._routing_policy, ModelPolicy
        ):
            self._iconq_model.iconq_interaction_featurizer.set_rpu_lookup(
                self._pool.get_rpu
            )

        # Autoscaler (replaces CapacityController).
        # If no policy was provided, default to HeadroomPolicy.
        policy: AutoscalingPolicy
        if self._autoscaling_policy is None:
            policy = HeadroomPolicy(
                slo_resolver=self._slo_resolver,
                slo_metric=self._slo_metric,
                allowed_rpu_sizes=list(config.allowed_rpu_sizes),
            )
        else:
            policy = self._autoscaling_policy

        self._autoscaler: Autoscaler = Autoscaler(
            policy=policy,
            pool=self._pool,
            on_spin_up=self._on_sim_spin_up,
            on_tear_down=self._on_sim_tear_down,
            capacity_checkpoints=self._capacity_checkpoints,
        )

        # Register all initial clusters with the autoscaler so
        # they are protected by the minimum lifetime.
        for cn in self._pool.cluster_names:
            self._autoscaler.notify_cluster_ready(
                cn, self._pool.get_rpu(cn), 0.0
            )

        # Event queue
        self._pending_events = []
        self._event_counter = 0
        self._current_sim_time_s = 0.0
        self._next_tick_time_s = self._cc_poll_interval_s

    def _on_sim_spin_up(self, reason: str, rpu: int) -> None:
        """Capacity-controller callback: schedule a new cluster."""
        cluster_name = self._pool.request_spin_up(rpu, self._current_sim_time_s)
        ready_time = (
            self._current_sim_time_s + self._provisioner.spin_up_delay_s
        )
        self._event_counter += 1
        heapq.heappush(
            self._pending_events,
            (ready_time, self._event_counter, cluster_name),
        )
        logger.debug(
            "Scheduled cluster %s (%d RPU) ready at t=%.1f",
            cluster_name,
            rpu,
            ready_time,
        )
        self._log_if_verbose(
            {
                "timestamp": self._current_sim_time_s,
                "event_type": "spin_up_scheduled",
                "cluster_name": cluster_name,
                "rpu": rpu,
                "reason": reason,
                "end_time_s": ready_time,
            }
        )

    def _on_sim_tear_down(self, cluster_name: str) -> None:
        """Capacity-controller callback: begin graceful tear-down.

        Delegates to the pool, which marks the cluster as DRAINING.
        When the last active query finishes (via ``on_query_finish``),
        the pool automatically finalises removal.
        """
        # Pre-check for logging: pool will guard against last-cluster
        # removal, but we want a specific log entry.
        ready_names = self._pool.ready_cluster_names
        if len(ready_names) <= 1:
            logger.debug(
                "Skipping tear-down of %s — it is the last routable "
                "cluster.",
                cluster_name,
            )
            self._log_if_verbose(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "tear_down_blocked",
                    "cluster_name": cluster_name,
                    "reason": "last_routable_cluster",
                    "num_active_clusters": len(ready_names),
                }
            )
            return

        active = self._pool.get_active_queries(cluster_name)
        self._pool.request_tear_down(cluster_name, self._current_sim_time_s)

        if active:
            logger.debug(
                "Cluster %s marked as draining with %d active queries.",
                cluster_name,
                len(active),
            )
            self._log_if_verbose(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "tear_down_requested",
                    "cluster_name": cluster_name,
                    "reason": "draining",
                    "num_active_queries": len(active),
                }
            )
        else:
            self._log_if_verbose(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "tear_down_requested",
                    "cluster_name": cluster_name,
                    "reason": "immediate",
                    "num_active_queries": 0,
                }
            )

    def _process_pending_events_up_to(self, time_s: float) -> None:
        """Drain cluster-ready events that fire at or before *time_s*."""
        while self._pending_events and self._pending_events[0][0] <= time_s:
            _, _, cluster_name = heapq.heappop(self._pending_events)
            self._pool.on_cluster_ready(cluster_name, time_s)
            rpu = self._pool.get_rpu(cluster_name)
            # Notify the autoscaler that the cluster is now ready
            # (decrements pending count and records ready time for
            # minimum-lifetime enforcement).
            if self._autoscaler is not None:
                self._autoscaler.notify_cluster_ready(cluster_name, rpu, time_s)
            logger.debug(
                "Cluster %s (%d RPU) became ready at t=%.1f",
                cluster_name,
                rpu,
                time_s,
            )
            self._log_if_verbose(
                {
                    "timestamp": time_s,
                    "event_type": "cluster_ready",
                    "cluster_name": cluster_name,
                    "rpu": rpu,
                    "num_active_clusters": len(self._pool.cluster_names),
                }
            )

    def _advance_simulated_time(self, target_time_s: float) -> None:
        """Process capacity-controller ticks and cluster-ready events
        chronologically up to *target_time_s*.

        Must be called before routing each query so that the routing
        logic sees an up-to-date cluster set.
        """
        # Process ticks (and any events that fire before/at each tick).
        while self._next_tick_time_s <= target_time_s:
            tick_time = self._next_tick_time_s
            self._process_pending_events_up_to(tick_time)
            self._cleanup_completed_queries_up_to(tick_time)
            self._current_sim_time_s = tick_time

            # Process capacity checkpoints that fire at or before this tick.
            self._autoscaler.reconcile_checkpoints_up_to(tick_time)

            # Compute headroom for logging before the tick.
            all_aq = self._pool.get_all_active_queries()
            draining = self._pool.draining_cluster_names
            all_active = [
                q for cn, qs in all_aq.items() if cn not in draining for q in qs
            ]
            headroom = RoutingCore.compute_slo_headroom(
                all_active, self._slo_resolver, self._predicted_latencies
            )
            self._log_if_verbose(
                {
                    "timestamp": tick_time,
                    "event_type": "capacity_tick",
                    "headroom": headroom,
                    "num_active_queries": len(all_active),
                    "num_active_clusters": len(self._pool.cluster_names),
                }
            )

            self._autoscaler.on_time_advance(tick_time)
            self._next_tick_time_s += self._cc_poll_interval_s
            # Process events spawned by this tick (e.g. delay=0 spin-ups).
            self._process_pending_events_up_to(tick_time)

        # Remaining events up to target.
        self._process_pending_events_up_to(target_time_s)
        # Process any remaining checkpoints between last tick and target.
        self._autoscaler.reconcile_checkpoints_up_to(target_time_s)
        self._current_sim_time_s = target_time_s

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_one(
        self,
        sampling_spec: "Optional[RedsetWorkloadSamplingSpec]" = None,
        progress_callback: "Optional[Callable[[int, int], None]]" = None,
    ) -> None:
        """
        First pass: route queries as they come in, preferring active endpoints
        and minimizing SLO violations.
        """

        seq_num_to_cluster_name: dict[int, str] = {}

        # Store seed so it ends up in config.yml and experiment_meta.json
        self._seed = getattr(sampling_spec, "seed", None)
        self._write_config_yml()

        queries = (
            self._workload.queries(sampling_spec=sampling_spec)  # type: ignore[call-arg]
            if type(self._workload).__name__ == "RedsetWorkload"
            else self._workload.queries()
        )
        print(
            f"Simulating routing of {len(queries)} queries from workload "
            f"{self._workload_name} using {type(self._routing_policy).__name__}"
            + (
                f" (model {self._iconq_model_id})"
                if self._iconq_model_id
                else ""
            )
            + "..."
        )
        print(
            f"The first and last relative query start times are {queries[0].rel_start_time_s} and {queries[-1].rel_start_time_s}"
        )

        total_queries = len(queries)

        closed_loop_clock = 0.0  # tracks effective wall-clock in closed-loop

        # When a progress_callback is supplied (tuner workers), report
        # progress every _PROGRESS_INTERVAL queries instead of using tqdm.
        _PROGRESS_INTERVAL = 25
        _use_callback = progress_callback is not None
        iterator = (
            enumerate(queries)
            if _use_callback
            else tqdm(enumerate(queries), total=total_queries)
        )

        if _use_callback:
            progress_callback(0, total_queries)

        for i, query in iterator:

            # In closed-loop mode the next query starts only after the
            # previous one finishes, ignoring the original inter-arrival times.
            start_s = (
                closed_loop_clock
                if self._closed_loop
                else query.rel_start_time_s
            )

            self._advance_simulated_time(start_s)

            self._cleanup_completed_queries_up_to(start_s)

            self._log_if_verbose(
                {
                    "timestamp": start_s,
                    "event_type": "arrival",
                    "query_id": query.query_id,
                    "query_text_id": query.query_text_id.value,
                }
            )

            # Route the query via the Router (delegates to policy).
            # DRAINING clusters are automatically excluded by the pool.
            result = self._router.route_query_with_predictions(
                query_id=query.query_id,
                query_text_id=str(query.query_text_id),
                start_time_s=start_s,
            )

            # If the policy did not produce a PlacementScore (e.g. RoundRobinPolicy),
            # score the chosen cluster post-hoc using the scoring
            # model so that latency predictions and co-runner updates are still
            # tracked correctly.
            if result.score is None and self._iconq_model is not None:
                computed_score, enriched_tq = self._score_with_model(
                    query_id=query.query_id,
                    query_text_id=query.query_text_id,
                    cluster_name=result.cluster_name,
                    start_time_s=start_s,
                )
                if (computed_score is not None) and (enriched_tq is not None):
                    result = RoutingResult(
                        cluster_name=result.cluster_name,
                        score=computed_score,
                        query=enriched_tq,
                    )

            # Feed routing result to the autoscaler.
            if self._autoscaler is not None:
                self._autoscaler.on_routing_result(
                    result, start_s, self._predicted_latencies
                )

            selected_cluster_name = result.cluster_name
            tq = result.query
            assert (
                result.score is not None
            ), "RoutingResult must have a score at this point."
            self_latency_s = result.score.latencies[query.query_id]

            # Store predicted latency and cluster mapping.
            self._predicted_latencies[query.query_id] = self_latency_s
            self._query_to_cluster_name[query.query_id] = selected_cluster_name

            self._log_if_verbose(
                {
                    "timestamp": start_s,
                    "event_type": "routing",
                    "query_id": query.query_id,
                    "query_text_id": query.query_text_id.value,
                    "cluster_name": selected_cluster_name,
                    "old_latency_s": None,
                    "raw_model_latency_s": None,
                    "latency_s": self_latency_s,
                    "end_time_s": start_s + self_latency_s,
                }
            )

            # Register the tracking query with the pool.
            # (Handles neighbour bookkeeping and billing-window start.)
            self._pool.on_query_start(tq, selected_cluster_name)

            seq_num_to_cluster_name[i] = selected_cluster_name

            # Update co-runner latencies on the chosen cluster using the
            # model predictions.
            for q in self._pool.get_active_queries(selected_cluster_name):
                if q.query_id == query.query_id:
                    continue
                old_latency_s = self._predicted_latencies.get(q.query_id, -1.0)
                predicted_latency_s = result.score.latencies.get(
                    q.query_id, old_latency_s
                )
                updated_latency_s = max(old_latency_s, predicted_latency_s)
                self._log_if_verbose(
                    {
                        "timestamp": start_s,
                        "event_type": "latency_update",
                        "query_id": q.query_id,
                        "query_text_id": q.query_text_id.value,
                        "cluster_name": selected_cluster_name,
                        "old_latency_s": old_latency_s,
                        "raw_model_latency_s": predicted_latency_s,
                        "latency_s": updated_latency_s,
                        "end_time_s": q.rel_start_time_s + updated_latency_s,
                    }
                )
                self._predicted_latencies[q.query_id] = updated_latency_s

            # Advance the closed-loop clock past this query's completion.
            if self._closed_loop:
                closed_loop_clock = start_s + self_latency_s

            # Report progress to the parent process.
            if _use_callback and (i + 1) % _PROGRESS_INTERVAL == 0:
                progress_callback(i + 1, total_queries)

        # Final progress report.
        if _use_callback:
            progress_callback(total_queries, total_queries)

        all_completed = [
            q
            for qs in self._pool.get_all_completed_queries().values()
            for q in qs
        ]
        if all_completed:
            workload_end_time_s = max(
                q.rel_start_time_s
                + self._predicted_latencies.get(q.query_id, 0.0)
                for q in all_completed
            )

        self._cleanup_completed_queries_up_to()
        self.write_out_billing_interval_analysis()
        if self._structured_handler is not None:
            self._structured_handler.finalize()

        mapping_out_path = os.path.join(self._out_dir, "mapping.yml")
        with open(mapping_out_path, "w") as f:
            dump_config(seq_num_to_cluster_name, f, sort_keys=False)

        if self._experiment_name:
            self._write_experiment_meta()

    def _cleanup_completed_queries_up_to(
        self, current_time_s: Optional[float] = None
    ) -> None:
        """
        Move queries that have completed by current_time_s from active to
        completed.

        Parameters:
            current_time_s: The current time in seconds since the start of the
                workload. If None, all active queries are considered completed.
        """
        # Snapshot cluster names before processing (pool may auto-remove
        # draining clusters as their last queries finish).
        draining_before = set(self._pool.draining_cluster_names)
        cluster_names = list(self._pool.ready_cluster_names) + list(
            draining_before
        )

        for cluster_name in cluster_names:
            active_queries = self._pool.get_active_queries(cluster_name)
            completed: list[tuple[Query, float]] = []
            for query in active_queries:
                end_time_s = (
                    query.rel_start_time_s
                    + self._predicted_latencies.get(query.query_id, 0.0)
                )
                if (current_time_s is None) or (end_time_s <= current_time_s):
                    completed.append((query, end_time_s))

            for query, end_time_s in completed:
                # Pool handles: active → completed, billing window,
                # auto-finalization of draining clusters.
                self._pool.on_query_finish(
                    query_id=query.query_id,
                    cluster_name=cluster_name,
                    current_time_s=end_time_s,
                )
                # Feed query-completion event to the autoscaler.
                if self._autoscaler is not None:
                    self._autoscaler.on_query_complete(
                        query.query_id, cluster_name, end_time_s
                    )
                self._log_if_verbose(
                    {
                        "timestamp": end_time_s,
                        "event_type": "completion",
                        "query_id": query.query_id,
                        "query_text_id": query.query_text_id.value,
                        "cluster_name": cluster_name,
                        "old_latency_s": None,
                        "raw_model_latency_s": None,
                        "latency_s": self._predicted_latencies.get(
                            query.query_id, -1.0
                        ),
                        "end_time_s": end_time_s,
                    }
                )

        # Detect draining clusters that were auto-removed by the pool.
        draining_after = set(self._pool.draining_cluster_names)
        for cn in draining_before - draining_after:
            logger.debug("Draining cluster %s is now empty — deactivated.", cn)
            self._log_if_verbose(
                {
                    "timestamp": current_time_s,
                    "event_type": "cluster_deactivated",
                    "cluster_name": cn,
                    "reason": "drain_complete",
                    "num_active_clusters": len(self._pool.cluster_names),
                }
            )

    def write_out_visualization(self) -> None:
        """
        Write out an HTML visualization of the query assignment, built from the
        structured log (no longer driven by in-memory snapshots).
        Optionally also exports a video if export_video flag is set.
        """
        log_path = os.path.join(self._out_dir, "structured_log.parquet")
        with open(os.path.join(self._out_dir, "config.yml")) as f:
            config = yaml.safe_load(f)

        snapshot = log_timeline_builder.build_final_snapshot_from_log(
            log_path=log_path, config=config
        )
        fig = render_gantt_scrubber(
            [snapshot],
            slo_s=self._slo_s,
            slo_metric=self._slo_metric,
            slo_threshold=self._slo_threshold,
            workload_name=self._workload_name,
        )

        out_path = os.path.join(self._out_dir, "visualization.html")
        fig.write_html(out_path, auto_play=False, include_plotlyjs="cdn")

        # Export video if requested
        if self._export_video:
            video_out_path = os.path.join(self._out_dir, "visualization.mp4")
            export_gantt_video(
                snapshots=[snapshot],
                slo_s=self._slo_s,
                output_path=video_out_path,
                frame_duration=self._video_frame_duration,
                constant_layout=True,
                slo_metric=self._slo_metric,
                slo_threshold=self._slo_threshold,
                workload_name=self._workload_name,
            )

    def write_out_billing_interval_analysis(self) -> None:
        """
        Write out a yaml file analyzing the billing intervals per cluster.
        """

        d = {}

        all_completed = self._pool.get_all_completed_queries()
        cost_map = self._pool.cost_per_second_map
        cluster_names = sorted(all_completed.keys())
        for cluster_name in cluster_names:
            completed_queries = all_completed[cluster_name]
            if len(completed_queries) == 0:
                continue

            from autoslo.blueprint_selection.slo_resolver import (
                query_interval as _qi,
            )  # noqa: PLC0415

            billed_intervals = Billing.billed_intervals(
                [
                    _qi(
                        q.rel_start_time_s,
                        self._predicted_latencies.get(q.query_id, 0.0),
                        q.query_id,
                    )
                    for q in completed_queries
                ],
            )
            total_duration_s = sum(iv.end - iv.begin for iv in billed_intervals)
            cost_per_second = cost_map.get(cluster_name, 0.0)
            d[cluster_name] = {
                "num_completed_queries": len(completed_queries),
                "num_billed_intervals": len(billed_intervals),
                "total_billed_time_s": float(total_duration_s),
                "cluster_cost_per_second": cost_per_second,
                "total_billed_cost": float(total_duration_s * cost_per_second),
                "billed_intervals": [
                    {
                        "begin_s": float(iv.begin),
                        "end_s": float(iv.end),
                        "query_ids": sorted(list(iv.data["query_ids"])),
                    }
                    for iv in billed_intervals
                ],
            }

        out_path = os.path.join(self._out_dir, "billing_interval_analysis.yml")
        with open(out_path, "w") as f:
            dump_config(d, f, sort_keys=False)

    def _write_experiment_meta(self) -> None:
        """
        Append this run's summary stats to the shared experiment_meta.json,
        creating it if it does not exist.  Uses a file lock for safety when
        multiple simulator processes share the same experiment directory.
        """
        if not self._experiment_name:
            return

        experiment_dir = os.path.join(
            pu.get_data_path(), "simulator_runs", self._experiment_name
        )
        meta_path = os.path.join(experiment_dir, "experiment_meta.json")
        lock_path = meta_path + ".lock"

        # Compute summary stats from the billing analysis file.
        billing_path = os.path.join(
            self._out_dir, "billing_interval_analysis.yml"
        )
        total_cost = 0.0
        if os.path.exists(billing_path):
            with open(billing_path) as f:
                billing = yaml.safe_load(f) or {}
            for cluster_data in billing.values():
                total_cost += cluster_data.get("total_billed_cost", 0.0)

        # Compute violation stats from the solve log.
        violation_rate = 0.0
        violation_amount_s = 0.0
        violation_relative_mean = 0.0
        num_queries = 0
        log_path = os.path.join(self._out_dir, "structured_log.parquet")
        if os.path.exists(log_path):
            import pandas as _pd

            log = _pd.read_parquet(log_path)
            completions = log[log["event_type"] == "completion"].copy()
            num_queries = len(completions)
            if num_queries > 0 and self._slo_s:
                durations = completions["latency_s"].fillna(0.0)
                # Compute per-row SLO using the resolver so that
                # per-template overrides are reflected in violation stats.
                per_row_slo = (
                    completions["query_text_id"]
                    .map(self._slo_resolver.resolve)
                    .fillna(self._slo_s)
                )
                violations = durations > per_row_slo
                violation_rate = float(violations.mean())
                violation_amount_s = float(
                    (durations - per_row_slo).clip(lower=0.0).sum()
                )
                # Relative violation: max(0, (latency - slo) / slo) per query.
                relative_violations = (
                    (durations - per_row_slo) / per_row_slo
                ).clip(lower=0.0)
                violation_relative_mean = float(relative_violations.mean())

        run_entry = {
            "run_id": self._run_id,
            "seed": self._seed,
            "slo_s": self._slo_s,
            "slo_metric": self._slo_metric.value,
            "slo_threshold": self._slo_threshold,
            "slo_dict_filename": self._slo_dict_filename,
            "slo_dict": self._slo_resolver.slo_dict,
            "violation_rate": round(violation_rate, 6),
            "violation_amount_s": round(violation_amount_s, 4),
            "violation_relative_mean": round(violation_relative_mean, 6),
            "total_cost": round(total_cost, 4),
            "num_queries": num_queries,
            "completed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with FileLock(lock_path):
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {
                    "experiment_name": self._experiment_name,
                    "runs": [],
                }
            # Replace entry if run_id already present (idempotent re-runs)
            meta["runs"] = [
                r for r in meta["runs"] if r.get("run_id") != self._run_id
            ]
            meta["runs"].append(run_entry)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)


if __name__ == "__main__":
    cfg, _ = cfgu.load_config_from_cli(
        "Run the WorkloadSimulator from a YAML config file.",
    )
    sim = WorkloadSimulator.from_config_dict(cfg)
    sim.simulate_one()
