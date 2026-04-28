import heapq
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm

import autoslo.utils.config as cfgu
from autoslo.clusters.actions import SpinUpAction, TearDownAction
from autoslo.clusters.capacity_checkpoint import CapacityCheckpoint
from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.config.structured_config import StructuredConfig
from autoslo.routing.wrapper import route_and_update_bookkeeping
from autoslo.utils.billing import Billing
from autoslo.utils.logging import emit_structured
from autoslo.utils.paralellism import inner_level_num_cpus
from autoslo.utils.structured_events import (
    BaseStructuredEvent,
    EventType,
    QueryRelatedEvent,
)
from autoslo.utils.yaml_helpers import dump_yaml
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.workload import Workload
from autoslo.workload_execution.simulator_event import (
    SimulatorEvent,
    SimulatorEventType,
)


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
        out_dir: Optional[str | Path] = None,
        write_text_log: bool = False,
    ):
        """
        Initialize the simulator with the given config.
        """
        # ── Build, parse and dump structured config ──────────────────────────
        structured_config = StructuredConfig.build(
            cfg=cfg,
            out_dir=out_dir,
            write_text_log=write_text_log,
            is_runner=False,
        )
        self._run_id = structured_config.run_id
        self._out_dir = structured_config.out_dir
        self._write_text_log = structured_config.write_text_log
        self._iconq_model = structured_config.iconq_model
        self._workload = structured_config.workload
        self._slo_objective = structured_config.slo_objective
        self._slo_resolver = structured_config.slo_resolver
        self._pool = structured_config.pool
        self._capacity_checkpoints = structured_config.capacity_checkpoints
        self._router = structured_config.router
        self._autoscaler = structured_config.autoscaler
        self._structured_handler = structured_config.structured_log_handler

        dump_yaml(cfg, os.path.join(self._out_dir, "config.yml"))

        # ── Activate initial clusters immediately (no spin-up delay) ──────
        pending_cluster_names = self._pool.clusters_in_state(
            ClusterState.PENDING
        )
        for name in pending_cluster_names:
            self._pool.on_cluster_ready(name, 0.0)

        # ── Instance Variables ───────────────────────────────────────────────
        self._pending_events: list[SimulatorEvent] = []
        self._current_sim_time_s = 0.0

    @property
    def out_dir(self) -> str:
        """Path to the output directory for the current run."""
        return self._out_dir

    # ------------------------------------------------------------------
    # Dynamic provisioning helpers
    # ------------------------------------------------------------------

    def _on_sim_spin_up(self, action: SpinUpAction) -> None:
        """Capacity-controller callback: schedule a new cluster."""
        cluster_name = self._pool.request_spin_up(
            action, self._current_sim_time_s
        )
        if cluster_name is None:
            # Spin-up was denied by the budget; SPIN_UP_BLOCKED was already
            # emitted by the pool.  Disable future spin-up considerations in the
            # autoscaler.
            self._autoscaler.disable_spin_up()
            return
        provisioner = self._pool.provisioner
        assert type(provisioner) is SimulatedProvisioner
        ready_time = self._current_sim_time_s + provisioner.spin_up_delay_s
        heapq.heappush(
            self._pending_events,
            SimulatorEvent(
                rel_time_s=ready_time,
                event_type=SimulatorEventType.CLUSTER_READY,
                details={"cluster_name": cluster_name},
            ),
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

        print("Spinning up initial clusters...")
        self._pool.add_details_and_spin_up_initial_clusters(
            run_id=self._run_id,
            out_dir=self._out_dir,
            write_text_log=self._write_text_log,
        )

        seq_num_to_cluster_name: dict[int, str] = {}
        queries = self._workload.queries()

        print(
            f"Simulating routing of {len(queries)} queries from workload "
            f"{self._workload.workload_name} using  IconqModel "
            f"{self._iconq_model.id})..."
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

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=self._current_sim_time_s,
                event_type=EventType.RUN_START,
                source="WorkloadSimulator",
                details={
                    "workload_name": self._workload.workload_name,
                    "num_queries": self._total_queries,
                    "routing_policy": self._router.routing_policy.value,
                },
            )
        )

        # Add all the checkpoint creation events to the heap.
        for checkpoint in self._capacity_checkpoints:
            self._pending_events.append(
                SimulatorEvent(
                    rel_time_s=checkpoint.rel_time_s,
                    event_type=SimulatorEventType.CAPACITY_CHECKPOINT,
                    details={"checkpoint": checkpoint},
                )
            )
        # Add all the query arrival events to the heap.
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

        # Flush final progress so the bar reflects terminal state even when
        # the last callback happened at a coarse interval.
        progress_callback(self._completed_queries, self._total_queries)
        if self._completed_queries != self._total_queries:
            logging.warning(
                "Simulation ended with %d/%d completed queries.",
                self._completed_queries,
                self._total_queries,
            )
        if hasattr(self, "_progress_bar"):
            self._progress_bar.close()

        # Tear down all remaining READY/DRAINING clusters so the log viewer
        # shows cluster lifetimes correctly (mirrors WorkloadRunner cleanup).
        remaining = list(self._pool.clusters_in_state(ClusterState.READY))
        for cn in remaining:
            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=self._current_sim_time_s,
                    event_type=EventType.TEAR_DOWN_DECISION,
                    source="WorkloadSimulator",
                    cluster_name=cn,
                    details={"reason": "run_cleanup"},
                )
            )
            try:
                self._pool.request_tear_down(
                    TearDownAction(reason="run_cleanup", cluster_name=cn),
                    self._current_sim_time_s,
                    force=True,
                )
            except Exception:
                logging.exception("Failed to tear down cluster %s.", cn)

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=self._current_sim_time_s,
                event_type=EventType.RUN_FINISH,
                source="WorkloadSimulator",
                details={
                    "workload_name": self._workload.workload_name,
                },
            )
        )

        self.write_out_billing_interval_analysis()
        if self._structured_handler is not None:
            self._structured_handler.finalize()

        mapping_out_path = os.path.join(self._out_dir, "mapping.yml")
        dump_yaml(seq_num_to_cluster_name, mapping_out_path)

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
            QueryRelatedEvent(
                rel_time_s=self._current_sim_time_s,
                event_type=EventType.ARRIVAL,
                source="WorkloadSimulator",
                query_id=query.query_id,
                query_text_id=query.query_text_id,
            )
        )

        selected_cluster_name = route_and_update_bookkeeping(
            source="WorkloadSimulator",
            rel_time_s_getter=lambda: self._current_sim_time_s,
            pool=self._pool,
            router=self._router,
            query=query,
            iconq_model=self._iconq_model,
            autoscaler=self._autoscaler,
            on_spin_up=self._on_sim_spin_up,
            write_text_log=self._write_text_log,
            simulator_pending_events_heap=self._pending_events,
        )
        seq_num_to_cluster_name[index] = selected_cluster_name

        emit_structured(
            QueryRelatedEvent(
                rel_time_s=self._current_sim_time_s,
                event_type=EventType.QUERY_EXECUTION_START,
                source="WorkloadSimulator",
                cluster_name=selected_cluster_name,
                query_id=query.query_id,
                query_text_id=query.query_text_id,
            )
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
        query_text_id = event.details["query_text_id"]
        cluster_name = event.details["cluster_name"]
        latency_s_from_event = event.details["latency_s"]

        # Verify that this is a valid completion event for an active query.
        maybe_currently_predicted_latency_s = self._pool.get_predicted_latency(
            cluster_name, query_id
        )
        if (maybe_currently_predicted_latency_s is None) or (
            abs(maybe_currently_predicted_latency_s - latency_s_from_event)
            > 1e-3
        ):
            # This was an older completion event, but the latency prediction has
            # changed since.
            if self._write_text_log:
                logging.debug(
                    "Ignoring stale completion event for query %s on %s "
                    "(predicted=%.3f, event=%.3f).",
                    query_id,
                    cluster_name,
                    (
                        maybe_currently_predicted_latency_s
                        if maybe_currently_predicted_latency_s is not None
                        else float("nan")
                    ),
                    latency_s_from_event,
                )
            return

        emit_structured(
            QueryRelatedEvent(
                rel_time_s=self._current_sim_time_s,
                event_type=EventType.QUERY_EXECUTION_FINISH,
                source="WorkloadSimulator",
                cluster_name=cluster_name,
                query_id=query_id,
                query_text_id=query_text_id,
            )
        )

        emit_structured(
            QueryRelatedEvent(
                rel_time_s=self._current_sim_time_s,
                event_type=EventType.COMPLETION,
                source="WorkloadSimulator",
                cluster_name=cluster_name,
                details={
                    "success": True,
                },
                query_id=query_id,
                query_text_id=query_text_id,
            )
        )
        self._pool.on_query_finish(
            query_id=query_id,
            cluster_name=cluster_name,
            rel_time_s=self._current_sim_time_s,
        )

    def _handle_capacity_checkpoint(self, event: SimulatorEvent) -> None:
        """
        Handle a capacity checkpoint event: check the current provisioned
        capacity against the checkpoint's requirements, and if there is a gap,
        trigger the necessary spin-ups to meet the checkpoint.
        """
        checkpoint: CapacityCheckpoint = event.details["checkpoint"]
        checkpoint.reconcile(
            pool=self._pool,
            source="WorkloadSimulator",
            on_spin_up=self._on_sim_spin_up,
            write_text_log=self._write_text_log,
            rel_time_s_getter=lambda: self._current_sim_time_s,
        )

    def _handle_cluster_ready(self, event: SimulatorEvent) -> None:
        """
        Handle the event of a cluster becoming ready: update the pool and
        autoscaler state.
        """

        cluster_name = event.details["cluster_name"]
        self._pool.on_cluster_ready(cluster_name, self._current_sim_time_s)
        rpu = Cluster.rpu_for_cluster_name(cluster_name)

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=self._current_sim_time_s,
                event_type=EventType.CLUSTER_READY,
                source="WorkloadSimulator",
                cluster_name=cluster_name,
                details={
                    "rpu": rpu,
                    "num_active_clusters": len(
                        self._pool.clusters_in_state(ClusterState.READY)
                    ),
                },
            )
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
                    )
                    for latency_s, q in completed_queries
                ],
            )

            total_duration_s = sum(iv.end - iv.start for iv in billed_intervals)
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
                        "start_s": float(iv.start),
                        "end_s": float(iv.end),
                    }
                    for iv in billed_intervals
                ],
            }

        out_path = os.path.join(self._out_dir, "billing_interval_analysis.yml")
        dump_yaml(d, out_path)


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
