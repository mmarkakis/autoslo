import heapq
import json
import logging
import os
from datetime import datetime
from typing import Callable, Optional

import torch
import yaml
from filelock import FileLock
from tqdm import tqdm

import autoslo.utils.config as cfgu
import autoslo.utils.paths as pu
from autoslo.clusters.actions import SpinUpAction
from autoslo.clusters.capacity_checkpoint import CapacityCheckpoint
from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.routing.wrapper import route_and_update_bookkeeping
from autoslo.slo.slo_metric import LatencySlo, SloMetric
from autoslo.utils.billing import Billing
from autoslo.utils.logging import emit_structured
from autoslo.utils.paralellism import inner_level_num_cpus
from autoslo.utils.structured_events import (
    BaseStructuredEvent,
    EventType,
    QueryRelatedEvent,
)
from autoslo.utils.yaml_helpers import dump
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.workload import Workload
from autoslo.workload_execution.simulator_event import (
    SimulatorEvent,
    SimulatorEventType,
)
from autoslo.workload_execution.structured_config import StructuredConfig


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

        # ── Determine run_id ─────────────────────────────────────────
        self._run_id: Optional[str] = cfgu.getd(
            cfg, "basic_config.run_id"
        ) or str(int(datetime.now().timestamp() * 1000))
        self._cfg = cfgu.copy_and_apply_overrides(
            cfg, {"basic_config.run_id": self._run_id}
        )

        # ── Build, parse and dump structured config ──────────────────────────────
        structured_config = StructuredConfig.build(
            self._cfg, self._run_id, workload=workload, is_runner=False
        )

        self._iconq_model = structured_config.iconq_model
        self._closed_loop = structured_config.closed_loop
        self._workload = structured_config.workload
        self._slo_objective = structured_config.slo_objective
        self._slo_resolver = structured_config.slo_resolver
        self._pool = structured_config.pool
        self._capacity_checkpoints = structured_config.capacity_checkpoints
        self._router = structured_config.router
        self._autoscaler = structured_config.autoscaler
        self._out_dir = structured_config.out_dir
        self._experiment_name = structured_config.experiment_name
        self._write_text_log = structured_config.write_text_log
        self._structured_handler = structured_config.structured_log_handler

        dump(self._cfg, os.path.join(self._out_dir, "config.yml"))

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
                    "closed_loop": self._closed_loop,
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
            if num_queries > 0 and self._slo_resolver.default_slo_s:
                durations = completions["latency_s"].fillna(0.0)
                # Compute per-row SLO using the resolver so that
                # per-template overrides are reflected in violation stats.
                per_row_slo = (
                    completions["query_text_id"]
                    .map(self._slo_resolver.resolve)
                    .fillna(self._slo_resolver.default_slo_s)
                )

                lat_and_slos = [
                    LatencySlo(lat, slo)
                    for lat, slo in zip(durations, per_row_slo)
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
            "slo_s": self._slo_resolver.default_slo_s,
            "slo_metric": self._slo_objective.slo_metric.value,
            "slo_threshold": self._slo_objective.slo_threshold,
            "slo_dict_filename": self._slo_resolver.slo_dict_filename,
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
