import json
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from scipy import datasets
import yaml
from filelock import FileLock
from intervaltree import Interval  # type: ignore[import]
from tqdm import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection import log_timeline_builder
from autoslo.blueprint_selection.query_timeline_visualizer_2 import (
    export_gantt_video,
    render_gantt_scrubber,
)
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.utils.billing import Billing
from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.redset_workload import (
    RedsetWorkload,
    RedsetWorkloadSamplingSpec,
)
from autoslo.workload_definition.workload import Workload


class WorkloadRoutingSimulator:
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
        blueprint_name: str,
        slo_s: float,
        optimize_based_on_slo_violation_amount: bool = False,
        slo_violation_rate_threshold: float = 0,
        slo_violation_amount_threshold_s: float = 0,
        verbose: bool = False,
        export_video: bool = False,
        video_frame_duration: float = 1.0,
        simulator_run_id: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        self._workload_name = workload_name
        self._iconq_model_id = iconq_model_id
        self._blueprint_name = blueprint_name
        self._blueprint = Blueprint.from_config(blueprint_name)
        self._slo_s = slo_s
        self._iconq_model = IconqModel.load(model_id=iconq_model_id)

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

        self._workload: Workload
        if workload_name.startswith("redset"):
            self._workload = RedsetWorkload.load(workload_name)
        else:
            self._workload = Chunk.load(workload_name)

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
            "tpcds_temp_and_q_idx",
            "cluster_name",
            "old_latency_s",
            "raw_model_latency_s",
            "latency_s",
            "end_time_s",
            "marginal_slo_violation",
            "marginal_cost",
        ]
        self._log_threshold = 10000

        # Set up bookkeeping etc.
        self._cost_per_second_per_cluster: dict[str, float] = {}
        self._active_queries_per_cluster: dict[str, list[Query]] = {}
        self._completed_queries_per_cluster: dict[str, list[Query]] = {}
        self._most_recent_billing_window_start_time_per_cluster_s: dict[
            str, Optional[float]
        ] = {}

        for cluster_name in self._blueprint.cluster_names:
            cluster = Cluster.from_config(cluster_name)
            self._cost_per_second_per_cluster[cluster_name] = (
                cluster.cost_per_second
            )
            self._active_queries_per_cluster[cluster_name] = []
            self._completed_queries_per_cluster[cluster_name] = []
            self._most_recent_billing_window_start_time_per_cluster_s[
                cluster_name
            ] = None

        # Cache before-state per cluster to avoid redundant interval merges.
        self._cached_before_cost: dict[str, float] = {
            name: 0.0 for name in self._blueprint.cluster_names
        }
        self._cached_before_slo_violation: dict[str, float] = {
            name: 0.0 for name in self._blueprint.cluster_names
        }
        self._before_cache_valid: dict[str, bool] = {
            name: False for name in self._blueprint.cluster_names
        }

        # Incremental neighbor tracking: maps query_id → full list of all queries
        # that have ever been concurrent with it (active co-runners at assignment
        # time + any since-completed neighbors). Maintained incrementally so
        # routing needs no scans — just a lookup + the single incoming query.
        self._neighbors_per_active_query: dict[str, list[Query]] = {}

    # ------------------------------------------------------------------
    # helper: build/return the output directory path
    # ------------------------------------------------------------------
    def _make_out_dir(self, run_id: str) -> str:
        if self._experiment_name:
            out_dir = os.path.join(
                pu.get_data_path(),
                "simulator_runs",
                self._experiment_name,
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
                "blueprint_name": self._blueprint_name,
                "slo_s": self._slo_s,
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

        # Reset per-run bookkeeping for all clusters.
        for cluster_name in self._blueprint.cluster_names:
            self._active_queries_per_cluster[cluster_name] = []
            self._completed_queries_per_cluster[cluster_name] = []
            self._most_recent_billing_window_start_time_per_cluster_s[
                cluster_name
            ] = None
            # Invalidate before-cache for the new run.
            self._before_cache_valid[cluster_name] = False

        self._neighbors_per_active_query = {}

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

    def simulate_one(self, sampling_spec: RedsetWorkloadSamplingSpec) -> None:
        """
        First pass: route queries as they come in, preferring active endpoints
        and minimizing SLO violations.
        """

        seq_num_to_cluster_name: dict[int, str] = {}

        # Store seed so it ends up in config.yml and experiment_meta.json
        self._seed = getattr(sampling_spec, "seed", None)
        self._write_config_yml()

        queries = self._workload.queries(sampling_spec=sampling_spec)
        print(
            f"Simulating routing of {len(queries)} queries from workload "
            f"{self._workload_name} using Iconq model {self._iconq_model_id} "
            f"and blueprint {self._blueprint_name}..."
        )
        print(
            f"The first and last relative query start times are {queries[0].rel_start_time_s} and {queries[-1].rel_start_time_s}"
        )

        total_queries = len(queries)

        for i, query in tqdm(enumerate(queries), total=total_queries):

            self._cleanup_completed_queries_up_to(query.rel_start_time_s)

            self._log_if_verbose(
                {
                    "timestamp": query.rel_start_time_s,
                    "event_type": "arrival",
                    "query_id": query.query_id,
                    "tpcds_temp_and_q_idx": query.tpcds_temp_and_q_idx,
                }
            )

            # Add query to the right cluster.
            (
                selected_cluster_name,
                updated_query,
                latencies_on_best_cluster_s,
            ) = self._find_best_cluster_for_query(
                query,
            )
            self_latency_s = latencies_on_best_cluster_s[query.query_id]
            self._log_if_verbose(
                {
                    "timestamp": query.rel_start_time_s,
                    "event_type": "routing",
                    "query_id": query.query_id,
                    "tpcds_temp_and_q_idx": query.tpcds_temp_and_q_idx,
                    "cluster_name": selected_cluster_name,
                    "old_latency_s": None,
                    "raw_model_latency_s": None,
                    "latency_s": self_latency_s,
                    "end_time_s": query.rel_start_time_s + self_latency_s,
                }
            )
            current_actives = self._active_queries_per_cluster[
                selected_cluster_name
            ]
            # Initialize this query's neighbor list with all currently active
            # co-runners, and add it to each of their lists in turn.
            self._neighbors_per_active_query[updated_query.query_id] = list(
                current_actives
            )
            for active_q in current_actives:
                self._neighbors_per_active_query[active_q.query_id].append(
                    updated_query
                )
            self._active_queries_per_cluster[selected_cluster_name].append(
                updated_query
            )
            # Invalidate the before-cache for this cluster since its state changed.
            self._before_cache_valid[selected_cluster_name] = False
            seq_num_to_cluster_name[i] = selected_cluster_name

            # Go through the active queries on the best cluster and update their
            # latencies based on the prediction results.
            for q in self._active_queries_per_cluster[selected_cluster_name]:
                if q.query_id == query.query_id:
                    continue
                old_latency_s = q.latency_s
                predicted_latency_s = latencies_on_best_cluster_s[q.query_id]
                updated_latency_s = max(old_latency_s, predicted_latency_s)
                self._log_if_verbose(
                    {
                        "timestamp": query.rel_start_time_s,
                        "event_type": "latency_update",
                        "query_id": q.query_id,
                        "tpcds_temp_and_q_idx": q.tpcds_temp_and_q_idx,
                        "cluster_name": selected_cluster_name,
                        "old_latency_s": old_latency_s,
                        "raw_model_latency_s": predicted_latency_s,
                        "latency_s": updated_latency_s,
                        "end_time_s": q.rel_start_time_s + updated_latency_s,
                    }
                )
                q.latency_s = updated_latency_s

            # If we are the start of a new billing window on the cluster, set
            # the billing window start time.
            if (
                self._most_recent_billing_window_start_time_per_cluster_s[
                    selected_cluster_name
                ]
                is None
            ):
                self._most_recent_billing_window_start_time_per_cluster_s[
                    selected_cluster_name
                ] = query.rel_start_time_s

        workload_end_time_s = max(
            q.rel_start_time_s + q.latency_s
            for cluster_name in self._blueprint.cluster_names
            for q in self._completed_queries_per_cluster[cluster_name]
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
        ended_with_times: list[tuple[Query, float]] = []
        for cluster, active_queries in self._active_queries_per_cluster.items():
            still_active_queries = []
            for query in active_queries:
                end_time_s = query.rel_start_time_s + query.latency_s
                if (current_time_s is None) or (end_time_s <= current_time_s):
                    self._completed_queries_per_cluster[cluster].append(query)
                    del self._neighbors_per_active_query[query.query_id]
                    self._log_if_verbose(
                        {
                            "timestamp": end_time_s,
                            "event_type": "completion",
                            "query_id": query.query_id,
                            "cluster_name": query.cluster_name,
                            "old_latency_s": None,
                            "raw_model_latency_s": None,
                            "latency_s": query.latency_s,
                            "end_time_s": end_time_s,
                        }
                    )
                else:
                    still_active_queries.append(query)
            self._active_queries_per_cluster[cluster] = still_active_queries
            # Invalidate the before-cache if active queries changed.
            if len(still_active_queries) != len(active_queries):
                self._before_cache_valid[cluster] = False

            billing_window_start = (
                self._most_recent_billing_window_start_time_per_cluster_s[
                    cluster
                ]
            )
            if (
                (current_time_s is not None)
                and (len(still_active_queries) == 0)
                and (billing_window_start is not None)
                and (
                    billing_window_start + Billing.REDSHIFT_BILLING_THRESHOLD_S
                    < current_time_s
                )
            ):
                # If there are no more active queries on the cluster and the
                # billing window has passed, reset the billing window start time.
                self._most_recent_billing_window_start_time_per_cluster_s[
                    cluster
                ] = None

    def _slo_cmp_with_tolerance(self, a: float, b: float) -> int:
        """
        Compare two SLO violation amounts with a tolerance for optimization.

        Returns:
            -1 if a < b (considering the tolerance),
            0 if a and b are close enough to be considered equal,
            1 if a > b (considering the tolerance).
        """
        if a + self.TOLERANCE_FOR_SLO_VIOLATION_AMOUNT_OPTIMIZATION_S < b:
            return -1
        elif b + self.TOLERANCE_FOR_SLO_VIOLATION_AMOUNT_OPTIMIZATION_S < a:
            return 1
        else:
            return 0

    def _find_best_cluster_for_query(
        self,
        query: Query,
    ) -> tuple[str, Query, dict[str, float]]:
        """
        Finds the best cluster to route the query to, based on the projected
        latency and SLO violation amount.

        Parameters:
            query: The query to route.

        Returns:
            best_cluster: The cluster that was chosen as the best for routing
                    the query.
            query: The input query with updated fields reflecting the best
                cluster choice and latency prediction.
            latencies_on_best_cluster_s: A dictionary mapping query IDs to their
                projected latencies on the best cluster.
        """

        # Best bookkeeping.
        best_cluster_name = None
        marginal_slo_violation_on_best_cluster = float("inf")
        marginal_cost_on_best_cluster = float("inf")
        stage_latency_prediction_on_best_cluster_s = float("inf")
        latencies_on_best_cluster_s: dict[str, float] = {}

        query.featurization = self._iconq_model.iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
            query.tpcds_temp_and_q_idx
        )

        before_costs: dict[str, float] = {}
        before_slo_violations: dict[str, float] = {}
        run_to_base_to_neighbors: dict[str, dict[Query, list[Query]]] = {}

        # Single pass over clusters: compute before-state (with caching) and
        # build neighbor sets. For each active query, _neighbors_per_active_query
        # already holds its full co-runner history; we just append the incoming
        # query as a hypothetical. The incoming query's own list is the active set.
        for (
            cluster_name,
            active_queries,
        ) in self._active_queries_per_cluster.items():

            # --- neighbor sets ---
            run_to_base_to_neighbors[cluster_name] = {
                q: self._neighbors_per_active_query[q.query_id] + [query]
                for q in active_queries
            }
            run_to_base_to_neighbors[cluster_name][query] = active_queries + [
                query
            ]

            # --- before-state (cached) ---
            if self._before_cache_valid[cluster_name]:
                before_costs[cluster_name] = self._cached_before_cost[
                    cluster_name
                ]
                before_slo_violations[cluster_name] = (
                    self._cached_before_slo_violation[cluster_name]
                )
                continue

            query.cluster_name = cluster_name
            query.stage_latency_prediction_s = (
                self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                    {query.query_id: query.tpcds_temp_and_q_idx}, cluster_name
                )[query.query_id].overall_mean_s()
            )

            individual_before_slo_violations: list[float] | list[bool] = [
                (
                    q.slo_violation_amount_s(self._slo_s)
                    if self._optimize_based_on_slo_violation_amount
                    else q.violates_slo(self._slo_s)
                )
                for q in active_queries
            ]
            before_slo_violation = sum(individual_before_slo_violations)
            before_query_intervals = [q.as_interval() for q in active_queries]
            billing_window_start = (
                self._most_recent_billing_window_start_time_per_cluster_s[
                    cluster_name
                ]
            )
            if billing_window_start is not None:
                before_query_intervals.append(
                    Interval(billing_window_start, query.rel_start_time_s)
                )
            before_billed_s = sum(
                iv.end - iv.begin
                for iv in Billing.billed_intervals(before_query_intervals)
            )
            before_cost = (
                self._cost_per_second_per_cluster[cluster_name]
                * before_billed_s
            )
            before_costs[cluster_name] = before_cost
            before_slo_violations[cluster_name] = before_slo_violation
            self._cached_before_cost[cluster_name] = before_cost
            self._cached_before_slo_violation[cluster_name] = (
                before_slo_violation
            )
            self._before_cache_valid[cluster_name] = True

        dataset = ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=self._iconq_model.iconq_interaction_featurizer,
            run_to_base_to_neighbors=run_to_base_to_neighbors,
        )
        all_predictions = self._iconq_model.predict_from_dataset(dataset)

        for cluster_name, predictions in all_predictions.items():

            latencies_after = [predictions[query.query_id].overall_mean_s()]
            for q in run_to_base_to_neighbors[cluster_name].keys():
                query_id = q.query_id
                latencies_after.append(
                    max(q.latency_s, predictions[query_id].overall_mean_s())
                )

            # Process prediction results.
            individual_after_slo_violations: list[float] | list[bool] = [
                (
                    max(0, latency_s - self._slo_s)
                    if self._optimize_based_on_slo_violation_amount
                    else latency_s > self._slo_s
                )
                for latency_s in latencies_after
            ]
            after_slo_violation = sum(individual_after_slo_violations)
            marginal_slo_violation = (
                after_slo_violation - before_slo_violations[cluster_name]
            )
            after_query_intervals = [
                Interval(q.rel_start_time_s, q.rel_start_time_s + latency_s)
                for q, latency_s in zip(
                    run_to_base_to_neighbors[cluster_name].keys(),
                    latencies_after,
                )
            ]
            if (
                self._most_recent_billing_window_start_time_per_cluster_s[
                    cluster_name
                ]
                is not None
            ):
                after_query_intervals.append(
                    Interval(
                        self._most_recent_billing_window_start_time_per_cluster_s[
                            cluster_name
                        ],
                        query.rel_start_time_s,
                    )
                )
            after_billed_intervals = Billing.billed_intervals(
                after_query_intervals
            )
            # Sum billed intervals directly instead of calling Billing.billed_s().
            after_billed_s = sum(
                iv.end - iv.begin for iv in after_billed_intervals
            )
            after_cost = (
                self._cost_per_second_per_cluster[cluster_name] * after_billed_s
            )
            marginal_cost = after_cost - before_costs[cluster_name]

            # Compare and update best if needed. Cache the cmp result to avoid
            # calling _slo_cmp_with_tolerance twice with the same arguments.
            slo_cmp = self._slo_cmp_with_tolerance(
                marginal_slo_violation, marginal_slo_violation_on_best_cluster
            )
            if (
                (best_cluster_name is None)
                or (slo_cmp < 0)
                or (
                    slo_cmp == 0
                    and marginal_cost < marginal_cost_on_best_cluster
                )
            ):
                best_cluster_name = cluster_name
                marginal_slo_violation_on_best_cluster = marginal_slo_violation
                marginal_cost_on_best_cluster = marginal_cost
                stage_latency_prediction_on_best_cluster_s = (
                    query.stage_latency_prediction_s
                )
                latencies_on_best_cluster_s = {
                    q.query_id: latency_s
                    for q, latency_s in zip(
                        run_to_base_to_neighbors[cluster_name].keys(),
                        latencies_after,
                    )
                }

            self._log_if_verbose(
                {
                    "timestamp": query.rel_start_time_s,
                    "event_type": "cluster_consideration",
                    "query_id": query.query_id,
                    "tpcds_temp_and_q_idx": query.tpcds_temp_and_q_idx,
                    "cluster_name": cluster_name,
                    "old_latency_s": None,
                    "raw_model_latency_s": None,
                    "latency_s": None,
                    "end_time_s": None,
                    "marginal_slo_violation": marginal_slo_violation,
                    "marginal_cost": marginal_cost,
                }
            )

        assert best_cluster_name is not None

        # Update the query's cluster and stage latency prediction to reflect the
        # best cluster choice.
        query.cluster_name = best_cluster_name
        query.stage_latency_prediction_s = (
            stage_latency_prediction_on_best_cluster_s
        )
        query.latency_s = latencies_on_best_cluster_s[query.query_id]
        query.latency_is_lower_bound = False

        return (best_cluster_name, query, latencies_on_best_cluster_s)

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

        for cluster_name in self._blueprint.cluster_names:
            completed_queries = self._completed_queries_per_cluster[
                cluster_name
            ]
            if len(completed_queries) == 0:
                continue

            billed_intervals = Billing.billed_intervals(
                [q.as_interval() for q in completed_queries],
            )
            total_duration_s = sum(iv.end - iv.begin for iv in billed_intervals)
            cost_per_second = self._cost_per_second_per_cluster[cluster_name]
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
            completions = log[log["event_type"] == "completion"]
            num_queries = len(completions)
            if num_queries > 0 and self._slo_s:
                durations = completions["latency_s"].fillna(0.0)
                violations = durations > self._slo_s
                violation_rate = float(violations.mean())
                violation_amount_s = float(
                    (durations - self._slo_s).clip(lower=0.0).sum()
                )

        run_entry = {
            "run_id": self._run_id,
            "seed": self._seed,
            "slo_s": self._slo_s,
            "blueprint_name": self._blueprint_name,
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
