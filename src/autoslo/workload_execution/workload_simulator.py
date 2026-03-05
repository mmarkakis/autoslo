import heapq
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd
import yaml
from filelock import FileLock
from tqdm import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection import log_timeline_builder
from autoslo.blueprint_selection.query_timeline_visualizer_2 import (
    export_gantt_video,
    render_gantt_scrubber,
)
from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.capacity.capacity_controller import CapacityController
from autoslo.capacity.cluster_provisioner import SimulatedProvisioner
from autoslo.capacity.policy_tuner import DynamicClusterConfig
from autoslo.routing.managed_cluster_pool import ManagedClusterPool
from autoslo.routing.model_policy import ModelPolicy
from autoslo.routing.router import Router
from autoslo.routing.routing_core import RoutingCore
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.workload import Workload

if TYPE_CHECKING:
    from autoslo.workload_definition.redset_workload import (
        RedsetWorkload,
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
        iconq_model_id: str,
        slo_s: float,
        schema_name: Optional[str] = None,
        optimize_based_on_slo_violation_amount: bool = False,
        slo_violation_rate_threshold: float = 0,
        slo_violation_amount_threshold_s: float = 0,
        slo_dict_filename: Optional[str] = None,
        verbose: bool = True,
        export_video: bool = False,
        video_frame_duration: float = 1.0,
        simulator_run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
        overwrite_experiment: bool = False,
        dynamic_cluster_config: Optional[DynamicClusterConfig] = None,
        eta_crit: float = 0.1,
        idle_periods_before_tear_down: int = 5,
        capacity_poll_interval_s: float = 60.0,
        min_cluster_lifetime_s: float = 1200.0,
    ):
        self._workload_name = workload_name
        self._iconq_model_id = iconq_model_id
        self._slo_s = slo_s
        self._slo_dict_filename = slo_dict_filename
        self._slo_resolver = SloResolver(slo_s, slo_dict_filename)

        # Create the routing policy (loads the IconQ model internally).
        self._model_policy = ModelPolicy(
            iconq_model_id=iconq_model_id,
            default_slo_s=slo_s,
            slo_overrides=self._slo_resolver.slo_dict,
            optimize_by_amount=optimize_based_on_slo_violation_amount,
        )
        self._iconq_model = self._model_policy.iconq_model  # convenience alias

        # Dynamic provisioning parameters.
        self._dynamic_cluster_config = (
            dynamic_cluster_config
            if dynamic_cluster_config is not None
            else DynamicClusterConfig()
        )
        self._cc_eta_crit = eta_crit
        self._cc_idle_periods = idle_periods_before_tear_down
        self._cc_poll_interval_s = capacity_poll_interval_s
        # Minimum cluster lifetime
        self._cc_min_cluster_lifetime_s = min_cluster_lifetime_s

        self._optimize_based_on_slo_violation_amount = (
            optimize_based_on_slo_violation_amount
        )
        self._slo_violation_rate_threshold = slo_violation_rate_threshold
        self._slo_violation_amount_threshold_s = (
            slo_violation_amount_threshold_s
        )
        self._verbose = verbose
        self._export_video = export_video
        self._video_frame_duration = video_frame_duration
        self._simulator_run_id = simulator_run_id
        self._experiment_name = experiment_name
        self._overwrite_experiment = overwrite_experiment
        self._schema_name = schema_name
        self._workload: Workload
        if workload_name.startswith("redset"):
            from autoslo.workload_definition.redset_workload import RedsetWorkload  # noqa: PLC0415
            self._workload = RedsetWorkload.load(workload_name)
        else:
            workload_path = os.path.join(
                pu.get_workloads_dir(),
                schema_name or "",
                f"{workload_name}.parquet",
            )
            self._workload = Workload(
                workload_name,
                pd.read_parquet(workload_path),
            )
        self._workload.set_rel_start_times_from_zero()

        self._run_id = simulator_run_id or str(
            int(datetime.now().timestamp() * 1000)
        )

        self._seed: Optional[int] = None  # populated in simulate_one

        # Setup the outputs directory.
        self._out_dir = self._make_out_dir(self._run_id)
        self._write_config_yml()

        # Set up logging if verbose is enabled.
        self._log_idx = 0
        self._log_rows: list[dict[str, Any]] = []
        self._log_columns = [
            "timestamp",
            "event_type",
            "query_id",
            "query_text_id",
            "cluster_name",
            "old_latency_s",
            "raw_model_latency_s",
            "latency_s",
            "end_time_s",
            "marginal_slo_violation",
            "marginal_cost",
            "rpu",
            "reason",
            "num_active_queries",
            "num_active_clusters",
            "headroom",
        ]
        self._log_threshold = 10000

        # Dynamic provisioning state.
        self._provisioner: Optional[SimulatedProvisioner] = None
        self._pool: Optional[ManagedClusterPool] = None
        self._capacity_controller: Optional[CapacityController] = None
        self._router: Optional[Router] = None
        self._pending_events: list[tuple[float, int, str]] = []
        self._event_counter: int = 0
        self._current_sim_time_s: float = 0.0
        self._next_tick_time_s: float = 0.0

        self._init_dynamic_clusters()

    # ------------------------------------------------------------------
    # Factory: create from YAML config (aligned with WorkloadRunner)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "WorkloadSimulator":
        """Create a :class:`WorkloadSimulator` from a YAML config file.

        The configuration uses the same field names as
        :class:`~autoslo.workload_execution.workload_runner.WorkloadRunner`
        where the concepts overlap and adds simulator-specific keys.

        Required keys
        -------------
        workload_name : str
        iconq_model_id : str
        slo_s : float

        Optional keys (with defaults)
        -----------------------------
        slo_dict_filename : str | None
        optimize_based_on_slo_violation_amount : bool  (False)
        slo_violation_rate_threshold : float  (0)
        slo_violation_amount_threshold_s : float  (0)
        verbose : bool  (False)
        export_video : bool  (False)
        video_frame_duration : float  (1.0)
        simulator_run_id : str | None
        experiment_name : str | None
        overwrite_experiment : bool  (False)
        dynamic_cluster_config : dict with ``initial_rpus``,
            ``allowed_rpu_sizes``, ``spin_up_delay_s``
        eta_crit : float  (0.1)
        idle_periods_before_tear_down : int  (5)
        capacity_poll_interval_s : float  (60.0)
        min_cluster_lifetime_s : float  (1200.0)
        """
        path = Path(config_path)
        with open(path) as f:
            cfg = yaml.safe_load(f)

        # Handle dynamic_cluster_config sub-dict.
        dcc_raw = cfg.get("dynamic_cluster_config")
        dcc: Optional[DynamicClusterConfig] = None
        if dcc_raw is not None:
            # Convert list RPUs to tuples for the frozen dataclass.
            if "initial_rpus" in dcc_raw and isinstance(
                dcc_raw["initial_rpus"], list
            ):
                dcc_raw["initial_rpus"] = tuple(dcc_raw["initial_rpus"])
            if "allowed_rpu_sizes" in dcc_raw and isinstance(
                dcc_raw["allowed_rpu_sizes"], list
            ):
                dcc_raw["allowed_rpu_sizes"] = tuple(
                    dcc_raw["allowed_rpu_sizes"]
                )
            dcc = DynamicClusterConfig(**dcc_raw)

        return cls(
            workload_name=cfg["workload_name"],
            iconq_model_id=cfg["iconq_model_id"],
            slo_s=cfg["slo_s"],
            schema_name=cfg.get("schema_name"),
            optimize_based_on_slo_violation_amount=cfg.get(
                "optimize_based_on_slo_violation_amount", False
            ),
            slo_violation_rate_threshold=cfg.get(
                "slo_violation_rate_threshold", 0
            ),
            slo_violation_amount_threshold_s=cfg.get(
                "slo_violation_amount_threshold_s", 0
            ),
            slo_dict_filename=cfg.get("slo_dict_filename"),
            verbose=cfg.get("verbose", False),
            export_video=cfg.get("export_video", False),
            video_frame_duration=cfg.get("video_frame_duration", 1.0),
            simulator_run_id=cfg.get("simulator_run_id"),
            experiment_name=cfg.get("experiment_name"),
            overwrite_experiment=cfg.get("overwrite_experiment", False),
            dynamic_cluster_config=dcc,
            eta_crit=cfg.get("eta_crit", 0.1),
            idle_periods_before_tear_down=cfg.get(
                "idle_periods_before_tear_down", 5
            ),
            capacity_poll_interval_s=cfg.get(
                "capacity_poll_interval_s", 60.0
            ),
            min_cluster_lifetime_s=cfg.get(
                "min_cluster_lifetime_s", 1200.0
            ),
        )

    # ------------------------------------------------------------------
    # helper: build/return the output directory path
    # ------------------------------------------------------------------
    def _make_out_dir(self, run_id: str) -> str:
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
                "iconq_model_id": self._iconq_model_id,
                "slo_s": self._slo_s,
                "slo_dict_filename": self._slo_dict_filename,
                "slo_dict": self._slo_resolver.slo_dict,
                "optimize_based_on_slo_violation_amount": (
                    self._optimize_based_on_slo_violation_amount
                ),
                "slo_violation_rate_threshold": (
                    self._slo_violation_rate_threshold
                ),
                "slo_violation_amount_threshold_s": (
                    self._slo_violation_amount_threshold_s
                ),
                "verbose": self._verbose,
                "export_video": self._export_video,
                "video_frame_duration": self._video_frame_duration,
                "seed": self._seed,
            }
            yaml.safe_dump(d, f, sort_keys=False)

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

        # Reset logging.
        self._log_idx = 0
        self._log_rows = []

        # Reset per-run bookkeeping.
        self._init_dynamic_clusters()

    def _log_if_verbose(self, d: dict) -> None:
        """
        Create the specified log entry if verbose is enabled,
        and write out to a parquet file if the number of log entries reaches
        the threshold.

        Parameters:
            d: A dictionary containing the log entry data. The keys should match
                the columns specified in self._log_columns.
        """

        if not self._verbose:
            return

        self._log_rows.append(d)

        if len(self._log_rows) >= self._log_threshold:
            self._log_df = pd.DataFrame(
                self._log_rows, columns=self._log_columns
            )
            log_filename = os.path.join(
                self._out_dir, f"solve_log_{self._log_idx}.parquet"
            )
            self._log_df.to_parquet(log_filename)
            self._log_idx += 1
            self._log_rows = []

    def finalize_log(self) -> None:
        """
        Write out any remaining log entries and consolidate all log files into
        one, deleting the individual log files afterwards to save space.
        """

        if not self._verbose:
            return

        # Write out remaining log rows if any.
        if len(self._log_rows) > 0:
            self._log_df = pd.DataFrame(
                self._log_rows, columns=self._log_columns
            )
            log_filename = os.path.join(
                self._out_dir, f"solve_log_{self._log_idx}.parquet"
            )
            self._log_df.to_parquet(log_filename)
            self._log_idx += 1
            self._log_rows = []

        # Consolidate log files into one.
        all_log_dfs = []
        for idx in range(self._log_idx):
            log_filename = os.path.join(
                self._out_dir, f"solve_log_{idx}.parquet"
            )
            df = pd.read_parquet(log_filename)
            all_log_dfs.append(df)
            os.remove(log_filename)
        if len(all_log_dfs) > 0:
            full_log_df = pd.concat(all_log_dfs, ignore_index=True)
            full_log_out_path = os.path.join(self._out_dir, "solve_log.parquet")
            full_log_df.to_parquet(full_log_out_path, index=False)

    # ------------------------------------------------------------------
    # Dynamic provisioning helpers
    # ------------------------------------------------------------------

    def _init_dynamic_clusters(self) -> None:
        """Set up provisioner, pool, router, controller, and initial clusters."""
        config = self._dynamic_cluster_config

        # Provisioner
        self._provisioner = SimulatedProvisioner(
            spin_up_delay_s=config.spin_up_delay_s
        )

        # Pool (no initial clusters via constructor — we add them below
        # with instant on_cluster_ready to bypass spin-up delay).
        self._pool = ManagedClusterPool(
            provisioner=self._provisioner,
            initial_rpus=None,
            allowed_rpu_sizes=list(config.allowed_rpu_sizes),
        )

        # Activate initial clusters immediately (no spin-up delay).
        for rpu in config.initial_rpus:
            name = self._pool.request_spin_up(rpu, 0.0)
            self._pool.on_cluster_ready(name, 0.0)

        # Create the Router (invokes policy.on_attach to wire up RPU lookup).
        self._router = Router(
            policy=self._model_policy,
            pool=self._pool,
        )

        # Capacity controller (pool excludes draining clusters from routing;
        # we also exclude them from capacity-controller polling).
        self._capacity_controller = CapacityController(
            get_active_queries=lambda: {
                cn: qs
                for cn, qs in self._pool.get_all_active_queries().items()
                if cn not in self._pool.draining_cluster_names
            },
            slo_resolver=self._slo_resolver,
            on_spin_up=self._on_sim_spin_up,
            on_tear_down=self._on_sim_tear_down,
            poll_interval_s=self._cc_poll_interval_s,
            eta_crit=self._cc_eta_crit,
            idle_periods_before_tear_down=self._cc_idle_periods,
            allowed_rpu_sizes=list(config.allowed_rpu_sizes),
            min_cluster_lifetime_s=self._cc_min_cluster_lifetime_s,
            get_current_time_s=lambda: self._current_sim_time_s,
        )

        # Register all initial clusters with the controller so
        # they are protected by the minimum lifetime.
        for cn in self._pool.cluster_names:
            self._capacity_controller.notify_cluster_ready(cn, 0.0)

        # Event queue
        self._pending_events = []
        self._event_counter = 0
        self._current_sim_time_s = 0.0
        self._next_tick_time_s = self._cc_poll_interval_s

    def _on_sim_spin_up(self, reason: str, rpu: int) -> None:
        """Capacity-controller callback: schedule a new cluster."""
        cluster_name = self._pool.request_spin_up(
            rpu, self._current_sim_time_s
        )
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
        self._pool.request_tear_down(
            cluster_name, self._current_sim_time_s
        )

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
            # Notify the capacity controller that the cluster
            # is now ready (decrements pending count and records ready
            # time for minimum-lifetime enforcement).
            if self._capacity_controller is not None:
                self._capacity_controller.notify_cluster_ready(
                    cluster_name, time_s
                )
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
                    "num_active_clusters": len(
                        self._pool.cluster_names
                    ),
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

            # Compute headroom for logging before the tick.
            all_aq = self._pool.get_all_active_queries()
            draining = self._pool.draining_cluster_names
            all_active = [
                q
                for cn, qs in all_aq.items()
                if cn not in draining
                for q in qs
            ]
            headroom = RoutingCore.compute_slo_headroom(
                all_active, self._slo_resolver
            )
            self._log_if_verbose(
                {
                    "timestamp": tick_time,
                    "event_type": "capacity_tick",
                    "headroom": headroom,
                    "num_active_queries": len(all_active),
                    "num_active_clusters": len(
                        self._pool.cluster_names
                    ),
                }
            )

            self._capacity_controller.tick_once()
            self._next_tick_time_s += self._cc_poll_interval_s
            # Process events spawned by this tick (e.g. delay=0 spin-ups).
            self._process_pending_events_up_to(tick_time)

        # Remaining events up to target.
        self._process_pending_events_up_to(target_time_s)
        self._current_sim_time_s = target_time_s

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_one(
        self,
        sampling_spec: "Optional[RedsetWorkloadSamplingSpec]" = None,
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
            f"{self._workload_name} using Iconq model {self._iconq_model_id}..."
        )
        print(
            f"The first and last relative query start times are {queries[0].rel_start_time_s} and {queries[-1].rel_start_time_s}"
        )

        total_queries = len(queries)

        for i, query in tqdm(enumerate(queries), total=total_queries):

            self._advance_simulated_time(query.rel_start_time_s)

            self._cleanup_completed_queries_up_to(query.rel_start_time_s)

            self._log_if_verbose(
                {
                    "timestamp": query.rel_start_time_s,
                    "event_type": "arrival",
                    "query_id": query.query_id,
                    "query_text_id": query.query_text_id.value,
                }
            )

            # Route the query via the Router (delegates to ModelPolicy).
            # DRAINING clusters are automatically excluded by the pool.
            result = self._router.route_query_with_predictions(
                query_id=query.query_id,
                query_text_id=str(query.query_text_id),
                start_time_s=query.rel_start_time_s,
            )
            assert result.score is not None, (
                "ModelPolicy must always produce a PlacementScore"
            )
            selected_cluster_name = result.cluster_name
            tq = result.tracking_query
            self_latency_s = result.score.latencies[query.query_id]

            self._log_if_verbose(
                {
                    "timestamp": query.rel_start_time_s,
                    "event_type": "routing",
                    "query_id": query.query_id,
                    "query_text_id": query.query_text_id.value,
                    "cluster_name": selected_cluster_name,
                    "old_latency_s": None,
                    "raw_model_latency_s": None,
                    "latency_s": self_latency_s,
                    "end_time_s": query.rel_start_time_s + self_latency_s,
                }
            )

            # Register the tracking query with the pool.
            # (Handles neighbour bookkeeping and billing-window start.)
            self._pool.on_query_start(tq)

            seq_num_to_cluster_name[i] = selected_cluster_name

            # Update co-runner latencies on the chosen cluster using the
            # model predictions returned by the router.
            for q in self._pool.get_active_queries(
                selected_cluster_name
            ):
                if q.query_id == query.query_id:
                    continue
                old_latency_s = q.latency_s
                predicted_latency_s = result.score.latencies.get(
                    q.query_id, old_latency_s
                )
                updated_latency_s = max(old_latency_s, predicted_latency_s)
                self._log_if_verbose(
                    {
                        "timestamp": query.rel_start_time_s,
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
                q.latency_s = updated_latency_s

        all_completed = [
            q
            for qs in self._pool.get_all_completed_queries().values()
            for q in qs
        ]
        if all_completed:
            workload_end_time_s = max(
                q.rel_start_time_s + q.latency_s for q in all_completed
            )

        self._cleanup_completed_queries_up_to()
        self.write_out_billing_interval_analysis()
        self.finalize_log()

        mapping_out_path = os.path.join(self._out_dir, "mapping.yml")
        with open(mapping_out_path, "w") as f:
            yaml.safe_dump(seq_num_to_cluster_name, f, sort_keys=False)

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
        cluster_names = (
            list(self._pool.ready_cluster_names) + list(draining_before)
        )

        for cluster_name in cluster_names:
            active_queries = self._pool.get_active_queries(cluster_name)
            completed: list[tuple[Query, float]] = []
            for query in active_queries:
                end_time_s = query.rel_start_time_s + query.latency_s
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
                self._log_if_verbose(
                    {
                        "timestamp": end_time_s,
                        "event_type": "completion",
                        "query_id": query.query_id,
                        "query_text_id": query.query_text_id.value,
                        "cluster_name": cluster_name,
                        "old_latency_s": None,
                        "raw_model_latency_s": None,
                        "latency_s": query.latency_s,
                        "end_time_s": end_time_s,
                    }
                )

        # Detect draining clusters that were auto-removed by the pool.
        draining_after = set(self._pool.draining_cluster_names)
        for cn in draining_before - draining_after:
            logger.debug(
                "Draining cluster %s is now empty — deactivated.", cn
            )
            self._log_if_verbose(
                {
                    "timestamp": current_time_s,
                    "event_type": "cluster_deactivated",
                    "cluster_name": cn,
                    "reason": "drain_complete",
                    "num_active_clusters": len(
                        self._pool.cluster_names
                    ),
                }
            )

    def write_out_visualization(self) -> None:
        """
        Write out an HTML visualization of the query assignment, built from the
        solve log (no longer driven by in-memory snapshots).
        Optionally also exports a video if export_video flag is set.
        """
        log_path = os.path.join(self._out_dir, "solve_log.parquet")
        with open(os.path.join(self._out_dir, "config.yml")) as f:
            config = yaml.safe_load(f)

        snapshot = log_timeline_builder.build_final_snapshot_from_log(
            log_path=log_path, config=config
        )
        fig = render_gantt_scrubber(
            [snapshot],
            slo_s=self._slo_s,
            violation_rate_threshold=self._slo_violation_rate_threshold,
            violation_amount_threshold=self._slo_violation_amount_threshold_s,
            optimize_cumulative_slo_violation_time=(
                self._optimize_based_on_slo_violation_amount
            ),
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
                violation_rate_threshold=self._slo_violation_rate_threshold,
                violation_amount_threshold=self._slo_violation_amount_threshold_s,
                optimize_cumulative_slo_violation_time=(
                    self._optimize_based_on_slo_violation_amount
                ),
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

            billed_intervals = Billing.billed_intervals(
                [q.as_interval() for q in completed_queries],
            )
            total_duration_s = sum(iv.end - iv.begin for iv in billed_intervals)
            cost_per_second = cost_map.get(
                cluster_name, 0.0
            )
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
            yaml.safe_dump(d, f, sort_keys=False)

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
        num_queries = 0
        log_path = os.path.join(self._out_dir, "solve_log.parquet")
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

        run_entry = {
            "run_id": self._run_id,
            "seed": self._seed,
            "slo_s": self._slo_s,
            "slo_dict_filename": self._slo_dict_filename,
            "slo_dict": self._slo_resolver.slo_dict,
            "violation_rate": round(violation_rate, 6),
            "total_cost": round(total_cost, 4),
            "violation_amount_s": round(violation_amount_s, 4),
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the WorkloadSimulator from a YAML config file.",
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (e.g. data/__run_configs/test.yml).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    sim = WorkloadSimulator.from_config(args.config)
    sim.simulate_one()
