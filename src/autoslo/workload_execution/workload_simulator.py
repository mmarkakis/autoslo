import copy
import heapq
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import yaml
from filelock import FileLock
from tqdm import tqdm

import autoslo.utils.config as cfgu
import autoslo.utils.paths as pu
from autoslo.blueprint_selection import log_timeline_builder
from autoslo.blueprint_selection.query_timeline_visualizer_2 import (
    export_gantt_video,
    render_gantt_scrubber,
)
from autoslo.clusters.autoscaler import (
    Autoscaler,
    AutoscalingAction,
    AutoscalingActionType,
)

from autoslo.clusters.capacity_checkpoint import (
    CapacityCheckpoint,
)
from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.clusters.managed_cluster_pool import (
    ManagedClusterPool,
    ManagedClusterPoolConfig,
)
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter, QueryRouterPolicy
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.billing import Billing
from autoslo.utils.paralellism import inner_level_num_cpus

from autoslo.utils.structured_log import (
    LOGGER_NAME,
    emit_structured,
    setup_structured_logging,
)
from autoslo.utils.yaml_helpers import dump
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.workload import Workload

_has_structured = lambda: bool(logging.getLogger(LOGGER_NAME).handlers)


@dataclass
class SimulatorEventType:
    QUERY_ARRIVAL: str = "query_arrival"
    QUERY_COMPLETION: str = "query_completion"
    CAPACITY_CHECKPOINT: str = "capacity_checkpoint"
    CLUSTER_READY: str = "cluster_ready"


@dataclass
class SimulatorEvent:
    rel_time_s: float
    event_type: str
    details: dict[str, Any]

    def __lt__(self, other: "SimulatorEvent") -> bool:
        """Order by rel_time_s, then by event_type for tie-breaking."""
        if self.rel_time_s == other.rel_time_s:
            return self.event_type < other.event_type
        return self.rel_time_s < other.rel_time_s


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

    def __init__(
        self,
        cfg: dict,
        workload: Optional[Workload] = None,
    ):
        """
        Initialize the simulator with the given config.
        """

        # ── basic ────────────────────────────────────────────────────────────
        self._run_id: Optional[str] = cfgu.getd(
            cfg, "basic_config.run_id"
        ) or str(int(datetime.now().timestamp() * 1000))
        self._cfg = cfgu.copy_and_apply_overrides(
            cfg, {"basic_config.run_id": self._run_id}
        )
        self._schema_name: str = cfgu.getd(
            self._cfg, "basic_config.schema_name", required=True
        )
        self._experiment_name: Optional[str] = cfgu.getd(
            self._cfg, "basic_config.experiment_name"
        )
        self._overwrite_experiment: bool = cfgu.getd(
            self._cfg, "basic_config.overwrite_experiment", False
        )
        self._iconq_model_id: str = cfgu.getd(
            self._cfg, "basic_config.iconq_model_id", required=True
        )
        self._iconq_model = IconqModel.load(self._iconq_model_id)

        # ── workload ─────────────────────────────────────────────────────
        self._closed_loop: bool = bool(
            cfgu.getd(self._cfg, "workload_config.closed_loop", False)
        )
        if workload is not None:
            self._workload = workload
        else:
            workload_name: str = cfgu.getd(
                self._cfg, "workload_config.workload_name", required=True
            )
            abs_start_time_start: str | None = cfgu.getd(
                self._cfg, "workload_config.abs_start_time_start"
            )
            abs_start_time_end: str | None = cfgu.getd(
                self._cfg, "workload_config.abs_start_time_end"
            )
            rescale_factor: float | None = cfgu.getd(
                self._cfg, "workload_config.rescale_factor", None
            )
            self._workload = Workload(
                workload_name=workload_name,
                schema_name=self._schema_name,
            )
            self._workload.slice_by_abs_time(
                abs_start_time_start, abs_start_time_end
            )
            self._workload.set_rel_start_times_from_zero()
            self._workload.rescale_rel_start_times(rescale_factor)

        # ── SLO ──────────────────────────────────────────────────────────────
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

        # ── Managed Cluster Pool ─────────────────────────────────────────────
        self._managed_cluster_pool_config = (
            ManagedClusterPoolConfig.parse_from_cfg(self._cfg)
        )
        self._allowed_rpu_sizes = list(
            self._managed_cluster_pool_config.allowed_rpu_sizes
        )
        self._provisioner: SimulatedProvisioner = SimulatedProvisioner(
            spin_up_delay_s=self._managed_cluster_pool_config.spin_up_delay_s
        )
        self._pool: ManagedClusterPool = ManagedClusterPool(
            provisioner=self._provisioner,
            config=self._managed_cluster_pool_config,
        )
        self._capacity_checkpoints = CapacityCheckpoint.parse_from_cfg(
            self._cfg
        )

        # Activate initial clusters immediately (no spin-up delay).
        pending_cluster_names = self._pool.clusters_in_state(
            ClusterState.PENDING
        )
        for name in pending_cluster_names:
            self._pool.on_cluster_ready(name, 0.0)

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
        self._export_video: bool = cfgu.getd(
            self._cfg, "output_config.export_video", False
        )
        self._video_frame_duration: float = cfgu.getd(
            self._cfg, "output_config.video_frame_duration", 1.0
        )
        out_dir_override = cfgu.getd(self._cfg, "output_config.out_dir", None)
        self._out_dir = WorkloadSimulator._make_out_dir(
            run_id=self._run_id,
            out_dir_override=out_dir_override,
            experiment_name=self._experiment_name,
            overwrite_experiment=self._overwrite_experiment,
        )
        dump(self._cfg, os.path.join(self._out_dir, "config.yml"))

        # ── Logging ───────────────────────────────────────────────────────────
        self._structured_handler = setup_structured_logging(
            out_dir=self._out_dir
        )
        if self._write_text_log:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            # Remove all existing handlers (console and file alike) so that
            # log records are emitted only to the run-specific file and the
            # structured log — never to the caller's console.
            for h in list(logger.handlers):
                logger.removeHandler(h)

            log_file_path = os.path.join(self._out_dir, "run.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # Prevent propagation to other loggers (which may print to console).
            logger.propagate = False
            logging.info(f"Run directory created at {self._out_dir}")

        # ── Instance Variables ───────────────────────────────────────────────
        self._pending_events: list[SimulatorEvent] = []
        self._current_sim_time_s = 0.0
        self._predicted_latencies: dict[str, dict[str, float]] = defaultdict(
            dict
        )
        """cluster_name → {query_id → current best latency estimate}."""

    @property
    def out_dir(self) -> str:
        """Path to the output directory for the current run."""
        return self._out_dir

    # ------------------------------------------------------------------
    # helper: build/return the output directory path
    # ------------------------------------------------------------------
    @staticmethod
    def _make_out_dir(
        run_id: str,
        out_dir_override: str | None = None,
        experiment_name: str | None = None,
        overwrite_experiment: bool = False,
    ) -> str:
        if out_dir_override is not None:
            out_dir = os.path.join(str(out_dir_override), run_id)
            os.makedirs(out_dir, exist_ok=True)
            return out_dir
        if experiment_name:
            experiment_dir = os.path.join(
                pu.get_data_path(), "simulator_runs", experiment_name
            )
            if os.path.exists(experiment_dir) and overwrite_experiment:
                shutil.rmtree(experiment_dir)
            out_dir = os.path.join(
                experiment_dir,
                run_id,
            )
        else:
            out_dir = os.path.join(pu.get_data_path(), "simulator_runs", run_id)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    # ------------------------------------------------------------------
    # Dynamic provisioning helpers
    # ------------------------------------------------------------------

    def _on_sim_spin_up(self, action: AutoscalingAction) -> None:
        """Capacity-controller callback: schedule a new cluster."""
        rpu = action.rpu
        assert rpu is not None, "RPU must be specified for spin-up actions."
        cluster_name = self._pool.request_spin_up(rpu, self._current_sim_time_s)
        ready_time = (
            self._current_sim_time_s + self._provisioner.spin_up_delay_s
        )
        heapq.heappush(
            self._pending_events,
            SimulatorEvent(
                rel_time_s=ready_time,
                event_type=SimulatorEventType.CLUSTER_READY,
                details={"cluster_name": cluster_name},
            ),
        )
        if self._write_text_log:
            logging.debug(
                "Scheduled cluster %s (%d RPU) ready at t=%.1f",
                cluster_name,
                rpu,
                ready_time,
            )
        emit_structured(
            {
                "timestamp": self._current_sim_time_s,
                "event_type": "spin_up_scheduled",
                "cluster_name": cluster_name,
                "rpu": rpu,
                "reason": action.reason,
                "end_time_s": ready_time,
                "source": "WorkloadSimulator",
            }
        )

    def _on_sim_tear_down(self, action: AutoscalingAction) -> None:
        """Capacity-controller callback: begin graceful tear-down.

        Delegates to the pool, which marks the cluster as DRAINING.
        When the last active query finishes (via ``on_query_finish``),
        the pool automatically finalises removal.
        """
        cluster_name = action.cluster_name
        assert cluster_name is not None
        # Pre-check for logging: pool will guard against last-cluster
        # removal, but we want a specific log entry.
        ready_names = self._pool.clusters_in_state(ClusterState.READY)
        if len(ready_names) <= 1:
            if self._write_text_log:
                logging.debug(
                    "Skipping tear-down of %s — it is the last routable "
                    "cluster.",
                    cluster_name,
                )
            if _has_structured():
                emit_structured(
                    {
                        "timestamp": self._current_sim_time_s,
                        "source": "WorkloadSimulator",
                        "event_type": "tear_down_skipped",
                        "cluster_name": cluster_name,
                        "reason": "last_cluster",
                    }
                )

            return

        snapshot = self._pool.snapshot(only_ready=False)
        active = snapshot[cluster_name].active_queries
        self._pool.request_tear_down(cluster_name, self._current_sim_time_s)

        if active:
            if self._write_text_log:
                logging.debug(
                    "Cluster %s marked as draining with %d active queries.",
                    cluster_name,
                    len(active),
                )
            emit_structured(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "tear_down_requested",
                    "cluster_name": cluster_name,
                    "reason": "draining",
                    "num_active_queries": len(active),
                    "source": "WorkloadSimulator",
                }
            )
        else:
            emit_structured(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "tear_down_requested",
                    "cluster_name": cluster_name,
                    "reason": "immediate",
                    "num_active_queries": 0,
                    "source": "WorkloadSimulator",
                }
            )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    _PROGRESS_INTERVAL = 100

    def default_progress_callback(self, completed: int, total: int) -> None:
        """Default progress callback that prints progress to a tqdm progress bar."""
        if completed == 0:
            self._progress_bar = tqdm(total=total)
        else:
            self._progress_bar.n = completed
            self._progress_bar.refresh()

    def run(
        self,
        progress_callback: "Optional[Callable[[int, int], None]]" = None,
    ) -> None:
        """
        First pass: route queries as they come in, preferring active endpoints
        and minimizing SLO violations.
        """

        seq_num_to_cluster_name: dict[int, str] = {}
        queries = self._workload.queries()

        print(
            f"Simulating routing of {len(queries)} queries from workload "
            f"{self._workload.workload_name} using "
            + (
                f" (model {self._iconq_model_id})"
                if self._iconq_model_id
                else ""
            )
            + "..."
        )
        print(
            f"The first and last relative query start times are "
            f"{queries[0].rel_start_time_s} and {queries[-1].rel_start_time_s}"
        )

        self._total_queries = len(queries)
        self._completed_queries = 0

        if progress_callback is None:
            progress_callback = self.default_progress_callback
        progress_callback(0, self._total_queries)

        # Add all the checkpoint creation events to the heap.
        for checkpoint in self._capacity_checkpoints:
            self._pending_events.append(
                SimulatorEvent(
                    rel_time_s=checkpoint.rel_time_s,
                    event_type=SimulatorEventType.CAPACITY_CHECKPOINT,
                    details={"checkpoint": checkpoint},
                )
            )
        # Add all the query arrival events to the heap, except in closed-loop
        # mode where we only add the first one.
        if self._closed_loop:
            first_query = queries[0]
            self._pending_events.append(
                SimulatorEvent(
                    rel_time_s=first_query.rel_start_time_s,
                    event_type=SimulatorEventType.QUERY_ARRIVAL,
                    details={"query": first_query, "index": 0},
                )
            )
        else:
            for i, query in enumerate(queries):
                self._pending_events.append(
                    SimulatorEvent(
                        rel_time_s=query.rel_start_time_s,
                        event_type=SimulatorEventType.QUERY_ARRIVAL,
                        details={"query": query, "index": i},
                    )
                )

        heapq.heapify(self._pending_events)

        while self._pending_events:

            event = heapq.heappop(self._pending_events)
            self._current_sim_time_s = event.rel_time_s

            match event.event_type:
                case SimulatorEventType.QUERY_ARRIVAL:
                    self._handle_query_arrival(event, seq_num_to_cluster_name)
                case SimulatorEventType.QUERY_COMPLETION:
                    self._handle_query_completion(event, progress_callback)
                case SimulatorEventType.CAPACITY_CHECKPOINT:
                    self._handle_capacity_checkpoint(event)
                case SimulatorEventType.CLUSTER_READY:
                    self._handle_cluster_ready(event)

                case _:
                    if self._write_text_log:
                        logging.warning(
                            f"Unknown event type: {event.event_type}"
                        )

        self.write_out_billing_interval_analysis()
        if self._structured_handler is not None:
            self._structured_handler.finalize()

        mapping_out_path = os.path.join(self._out_dir, "mapping.yml")
        dump(seq_num_to_cluster_name, mapping_out_path)

        if self._experiment_name:
            self._write_experiment_meta()

    def _handle_query_arrival(
        self, event: SimulatorEvent, seq_num_to_cluster_name: dict[int, str]
    ) -> None:
        """
        Handle the arrival of a new query: route it, update the pool and
        autoscaler state, and schedule its completion event.
        """
        query = event.details["query"]
        index = event.details["index"]

        emit_structured(
            {
                "timestamp": self._current_sim_time_s,
                "event_type": "arrival",
                "source": "WorkloadSimulator",
                "query_id": query.query_id,
                "query_text_id": query.query_text_id.value,
            }
        )

        # Route the query.
        snapshot = self._pool.snapshot(only_ready=True)
        selected_cluster_name, new_predicted_latencies_on_selected = (
            self._router.route_query(
                query=query,
                clusters=snapshot,
                initial_predicted_latencies=self._predicted_latencies,
                iconq_model=self._iconq_model,
                current_time_s=self._current_sim_time_s,
            )
        )
        self_latency_s = new_predicted_latencies_on_selected[query.query_id]
        seq_num_to_cluster_name[index] = selected_cluster_name
        emit_structured(
            {
                "timestamp": self._current_sim_time_s,
                "event_type": "routing",
                "query_id": query.query_id,
                "query_text_id": query.query_text_id.value,
                "cluster_name": selected_cluster_name,
                "old_latency_s": None,
                "raw_model_latency_s": None,
                "latency_s": self_latency_s,
                "end_time_s": self._current_sim_time_s + self_latency_s,
                "source": "WorkloadSimulator",
            }
        )

        # Update latencies of existing queries as needed (including the new one)
        for qid, latency_s in new_predicted_latencies_on_selected.items():
            old_latency_s = self._predicted_latencies[
                selected_cluster_name
            ].get(qid, None)

            if (old_latency_s is not None) and (
                abs(latency_s - old_latency_s) < 1e-3
            ):
                # No change in latency prediction for this query, so skip the update.
                continue

            self._predicted_latencies[selected_cluster_name][qid] = latency_s
            completion_time_s = self._current_sim_time_s + latency_s
            emit_structured(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "latency_update",
                    "source": "WorkloadSimulator",
                    "query_id": qid,
                    "cluster_name": selected_cluster_name,
                    "old_latency_s": old_latency_s,
                    "latency_s": latency_s,
                    "end_time_s": completion_time_s,
                }
            )
            heapq.heappush(
                self._pending_events,
                SimulatorEvent(
                    rel_time_s=completion_time_s,
                    event_type=SimulatorEventType.QUERY_COMPLETION,
                    details={
                        "query_id": qid,
                        "cluster_name": selected_cluster_name,
                        "latency_s": latency_s,
                    },
                ),
            )

        # Update pool and notify autoscaler of the new query.
        self._pool.on_query_start(
            query=query,
            cluster_name=selected_cluster_name,
        )
        post_snapshot = self._pool.snapshot(only_ready=False)
        autoscaler_suggested_actions: list[AutoscalingAction] = (
            self._autoscaler.inform(
                current_time_s=self._current_sim_time_s,
                current_query=query,
                pool_snapshot_with_current_query=post_snapshot,
                predicted_latencies=self._predicted_latencies,
            )
        )
        for action in autoscaler_suggested_actions:
            match action.action_type:
                case AutoscalingActionType.SPIN_UP:
                    self._on_sim_spin_up(action)
                case AutoscalingActionType.TEAR_DOWN:
                    self._on_sim_tear_down(action)
                case _:
                    if self._write_text_log:
                        logging.warning(
                            f"Unknown autoscaling action type: {action.action_type}"
                        )

    def _handle_query_completion(
        self,
        event: SimulatorEvent,
        progress_callback: "Callable[[int, int], None]",
    ) -> None:
        """
        Handle the completion of a query: update the pool and autoscaler state,
        and report progress.

        Note that the pool's on_query_finish will trigger auto-finalization of
        draining clusters, which we detect and log here.
        """
        query_id = event.details["query_id"]
        cluster_name = event.details["cluster_name"]
        latency_s_from_event = event.details["latency_s"]

        # Verify that this is a valid completion event for an active query.
        currently_predicted_latency_s = self._predicted_latencies.get(
            cluster_name, {}
        ).get(query_id)
        if (currently_predicted_latency_s is None) or (
            abs(currently_predicted_latency_s - latency_s_from_event) > 1e-3
        ):
            # This was an older completion event, but the latency prediction has
            # changed since.
            emit_structured(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "completion_ignored",
                    "source": "WorkloadSimulator",
                    "query_id": query_id,
                }
            )
            return

        # Clean up this query's state.
        del self._predicted_latencies[cluster_name][query_id]
        self._pool.on_query_finish(
            query_id=query_id,
            cluster_name=cluster_name,
            current_time_s=self._current_sim_time_s,
        )
        emit_structured(
            {
                "timestamp": self._current_sim_time_s,
                "event_type": "completion",
                "source": "WorkloadSimulator",
                "query_id": query_id,
                "cluster_name": cluster_name,
                "latency_s": latency_s_from_event,
            }
        )

        # If we are in closed-loop mode, schedule the next query arrival now that this one has completed.
        self._completed_queries += 1
        if (
            self._completed_queries % self._PROGRESS_INTERVAL == 0
            or self._completed_queries == self._total_queries
        ):
            progress_callback(self._completed_queries, self._total_queries)
        if self._closed_loop:
            next_query = self._workload.queries()[self._completed_queries]
            heapq.heappush(
                self._pending_events,
                SimulatorEvent(
                    rel_time_s=next_query.rel_start_time_s,
                    event_type=SimulatorEventType.QUERY_ARRIVAL,
                    details={
                        "query": next_query,
                        "index": self._completed_queries,
                    },
                ),
            )

    def _handle_capacity_checkpoint(self, event: SimulatorEvent) -> None:
        """
        Handle a capacity checkpoint event: check the current provisioned
        capacity against the checkpoint's requirements, and if there is a gap,
        trigger the necessary spin-ups to meet the checkpoint.
        """
        cp: CapacityCheckpoint = event.details["checkpoint"]
        current = Counter(self._pool.ready_and_pending_cluster_rpu_multiset)
        desired = Counter(cp.min_rpus)
        gap = desired - current  # keeps only positive differences

        if _has_structured():
            emit_structured(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "capacity_checkpoint_reconciliation",
                    "source": "WorkloadSimulator",
                    "checkpoint_rel_time_s": cp.rel_time_s,
                    "desired_rpus": list(cp.min_rpus),
                    "current_rpus": sorted(current.elements()),
                    "gap_spin_ups": str(dict(gap)) if gap else "",
                }
            )

        if not gap:
            if self._write_text_log:
                logging.debug(
                    "Checkpoint t=%.1f: already satisfied (current %s).",
                    cp.rel_time_s,
                    dict(current),
                )
            return

        if self._write_text_log:
            logging.debug(
                "Checkpoint t=%.1f: gap %s — spinning up.",
                cp.rel_time_s,
                dict(gap),
            )
        for rpu, count in sorted(gap.items()):
            for _ in range(count):
                action = AutoscalingAction(
                    action_type=AutoscalingActionType.SPIN_UP,
                    rpu=rpu,
                    reason=f"capacity_checkpoint@t={cp.rel_time_s}",
                )
                self._on_sim_spin_up(action)
                if _has_structured():
                    emit_structured(
                        {
                            "timestamp": self._current_sim_time_s,
                            "event_type": "spin_up",
                            "source": "WorkloadSimulator",
                            "rpu": rpu,
                            "reason": f"capacity_checkpoint@t={cp.rel_time_s}",
                        }
                    )

    def _handle_cluster_ready(self, event: SimulatorEvent) -> None:
        """
        Handle the event of a cluster becoming ready: update the pool and
        autoscaler state.
        """

        cluster_name = event.details["cluster_name"]

        self._pool.on_cluster_ready(cluster_name, self._current_sim_time_s)
        rpu = Cluster.rpu_for_cluster_name(cluster_name)

        snapshot = self._pool.snapshot(only_ready=False)
        if _has_structured():
            emit_structured(
                {
                    "timestamp": self._current_sim_time_s,
                    "event_type": "cluster_ready",
                    "source": "WorkloadSimulator",
                    "cluster_name": cluster_name,
                    "rpu": rpu,
                    "num_active_clusters": len(
                        self._pool.clusters_in_state(ClusterState.READY)
                    ),
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
            workload_name=self._workload.workload_name,
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
                workload_name=self._workload.workload_name,
            )

    def write_out_billing_interval_analysis(self) -> None:
        """
        Write out a yaml file analyzing the billing intervals per cluster.
        """

        d = {}

        all_completed = self._pool.get_all_completed_queries()
        cluster_names = sorted(all_completed.keys())
        for cluster_name in cluster_names:
            completed_queries = all_completed[cluster_name]
            if len(completed_queries) == 0:
                continue
            billed_intervals = Billing.billed_intervals(
                [
                    Query.query_interval(
                        q.rel_start_time_s,
                        latency_s,
                        q.query_id,
                    )
                    for latency_s, q in completed_queries
                ],
            )

            total_duration_s = sum(iv.end - iv.begin for iv in billed_intervals)
            rpu = Cluster.rpu_for_cluster_name(cluster_name)
            cost_per_second = Cluster.cost_per_second_for_rpu(rpu)
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
        dump(d, out_path)

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

                lat_and_slos = [
                    (lat, slo) for lat, slo in zip(durations, per_row_slo)
                ]

                violation_rate = SloMetric.BINARY.aggregate_batch(lat_and_slos)
                violation_amount_s = SloMetric.ABSOLUTE_S.aggregate_batch(
                    lat_and_slos
                )
                violation_relative_mean = SloMetric.RELATIVE.aggregate_batch(
                    lat_and_slos
                )

        run_entry = {
            "run_id": self._run_id,
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
    ncpus = str(inner_level_num_cpus())
    os.environ["OMP_NUM_THREADS"] = ncpus
    os.environ["MKL_NUM_THREADS"] = ncpus
    os.environ["OPENBLAS_NUM_THREADS"] = ncpus
    torch.set_num_threads(int(ncpus))
    sim = WorkloadSimulator(cfg)
    sim.run()
