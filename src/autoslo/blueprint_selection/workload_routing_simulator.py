import logging
import os
from datetime import datetime
from typing import Optional

import yaml
from intervaltree import Interval  # type: ignore[import]

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.query_timeline_visualizer_2 import (
    GanttRecorder,
    export_gantt_video,
    render_gantt_scrubber,
)
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.utils.billing import Billing
from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_definition.query import Query


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
        self._recorder = GanttRecorder()

        workload = Chunk.load(workload_name)  # FIXME: generalize to workloads.
        self._workload = workload

        self._run_id = simulator_run_id or str(
            int(datetime.now().timestamp() * 1000)
        )

        # Setup the outputs directory.
        self._out_dir = os.path.join(
            pu.get_data_path(), "simulator_runs", self._run_id
        )
        os.makedirs(self._out_dir, exist_ok=True)
        config_out_path = os.path.join(self._out_dir, "config.yml")
        with open(config_out_path, "w") as f:
            d = {
                "run_id": self._run_id,
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
            }
            yaml.safe_dump(d, f, sort_keys=False)

        # Set up logging if verbose is enabled.
        if self._verbose:
            log_filename = os.path.join(self._out_dir, "solve.log")
            print(f"Configuring log at {log_filename}")
            logging.basicConfig(
                filename=log_filename,
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                force=True,
            )
            logging.info(
                f"Starting simulator run for workload "
                f"'{self._workload_name}' with model '{self._iconq_model_id}' "
                f"and blueprint '{self._blueprint_name}'"
            )
            logging.info(
                f"Optimize based on SLO violation amount: "
                f"{self._optimize_based_on_slo_violation_amount} "
                f"with thresholds: "
                f"slo_violation_rate_threshold = "
                f"{self._slo_violation_rate_threshold}, "
                f"slo_violation_amount_threshold_s = "
                f"{self._slo_violation_amount_threshold_s}, "
                f"slo_s = "
                f"{self._slo_s}"
            )

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

    def _log_if_verbose(self, message: str) -> None:
        if self._verbose:
            logging.info(message)

    def first_pass(self) -> None:
        """
        First pass: route queries as they come in, preferring active endpoints
        and minimizing SLO violations.
        """

        self._recorder.snapshot(
            self._cost_per_second_per_cluster,
            self._completed_queries_per_cluster,
            self._active_queries_per_cluster,
            label="Start (t = 0.0s)",
            slo_s=self._slo_s,
        )

        seq_num_to_cluster_name: dict[int, str] = {}

        for i, query in enumerate(self._workload.queries):

            self._cleanup_completed_queries_up_to(query.start_time_s)

            self._log_if_verbose(
                f"({query.start_time_s:.3f}) Routing query {query.query_id} "
                f"with template and idx {query.tpcds_temp_and_q_idx}."
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
                f"Routing query {query.query_id} to cluster "
                f"{selected_cluster_name}. Predicted latency is "
                f"{self_latency_s:.2f}s (ends at "
                f"{query.start_time_s + self_latency_s:.2f}s). "
            )
            self._active_queries_per_cluster[selected_cluster_name].append(
                updated_query
            )
            seq_num_to_cluster_name[i] = selected_cluster_name

            # Go through the active queries on the best cluster and update their
            # latencies based on the prediction results.
            if len(self._active_queries_per_cluster[selected_cluster_name]) > 1:
                self._log_if_verbose(
                    f"Updating predicted latencies for active queries on "
                    f"cluster {selected_cluster_name}."
                )
            for q in self._active_queries_per_cluster[selected_cluster_name]:
                if q.query_id == query.query_id:
                    continue
                old_latency_s = q.latency_s
                predicted_latency_s = latencies_on_best_cluster_s[q.query_id]
                updated_latency_s = max(old_latency_s, predicted_latency_s)
                self._log_if_verbose(
                    f"\tQuery {q.query_id}: Old: {old_latency_s:.2f}s, "
                    f"Pred: {predicted_latency_s:.2f}s, "
                    f"New: {updated_latency_s:.2f}s (ends at "
                    f"{q.start_time_s + updated_latency_s:.2f}s)"
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
                ] = query.start_time_s

            self._recorder.snapshot(
                self._cost_per_second_per_cluster,
                self._completed_queries_per_cluster,
                self._active_queries_per_cluster,
                label=f"First pass iter {i} (t = {query.start_time_s:.3f}s)",
                slo_s=self._slo_s,
            )

        workload_end_time_s = max(
            q.start_time_s + q.latency_s
            for cluster_name in self._blueprint.cluster_names
            for q in self._completed_queries_per_cluster[cluster_name]
        )

        self._cleanup_completed_queries_up_to()

        self._recorder.snapshot(
            self._cost_per_second_per_cluster,
            self._completed_queries_per_cluster,
            self._active_queries_per_cluster,
            label=f"Final (t = {workload_end_time_s:.3f}s)",
            slo_s=self._slo_s,
        )

        self.write_out_visualization()
        self.write_out_billing_interval_analysis()

        mapping_out_path = os.path.join(self._out_dir, "mapping.yml")
        with open(mapping_out_path, "w") as f:
            yaml.safe_dump(seq_num_to_cluster_name, f, sort_keys=False)

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
                end_time_s = query.start_time_s + query.latency_s
                if (current_time_s is None) or (end_time_s <= current_time_s):
                    self._completed_queries_per_cluster[cluster].append(query)
                    ended_with_times.append((query, end_time_s))
                else:
                    still_active_queries.append(query)
            self._active_queries_per_cluster[cluster] = still_active_queries

            if (
                (current_time_s is not None)
                and (len(still_active_queries) == 0)
                and (
                    self._most_recent_billing_window_start_time_per_cluster_s[
                        cluster
                    ]
                    is not None
                )
                and (
                    self._most_recent_billing_window_start_time_per_cluster_s[
                        cluster
                    ]
                    + Billing.REDSHIFT_BILLING_THRESHOLD_S
                    < current_time_s
                )
            ):
                # If there are no more active queries on the cluster and the
                # billing window has passed, reset the billing window start time.
                self._most_recent_billing_window_start_time_per_cluster_s[
                    cluster
                ] = None

        for query, end_time_s in ended_with_times:
            self._log_if_verbose(
                f"({end_time_s:.3f}) Query {query.query_id} completed on "
                f"cluster {query.cluster_name} with latency "
                f"{query.latency_s:.2f}s."
            )

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

        for (
            cluster_name,
            active_queries,
        ) in self._active_queries_per_cluster.items():

            query.cluster_name = cluster_name
            query.stage_latency_prediction_s = (
                self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                    {query.query_id: query.tpcds_temp_and_q_idx}, cluster_name
                )[query.query_id].overall_mean_s()
            )

            # Before routing:
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
            if (
                self._most_recent_billing_window_start_time_per_cluster_s[
                    cluster_name
                ]
                is not None
            ):
                ongoing_billing_interval = Interval(
                    self._most_recent_billing_window_start_time_per_cluster_s[
                        cluster_name
                    ],
                    query.start_time_s,
                )
                before_query_intervals.append(ongoing_billing_interval)
            before_billed_intervals = Billing.billed_intervals(
                before_query_intervals
            )
            before_cost = self._cost_per_second_per_cluster[
                cluster_name
            ] * Billing.billed_s(before_billed_intervals)

            # After routing:
            active_w_current = active_queries + [query]
            dataset = ConcurrentQueryDataset.build_from_query_groups(
                iconq_interaction_featurizer=self._iconq_model.iconq_interaction_featurizer,
                base_queries=active_w_current,
                query_neighbors={
                    q.query_id: active_w_current for q in active_w_current
                },
                use_log_runtime=self._iconq_model.trained_on_log_runtime,
            )
            predictions = self._iconq_model.predict_from_dataset(dataset)
            latencies_after = [predictions[query.query_id].overall_mean_s()]
            for q in active_queries:
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
            marginal_slo_violation = after_slo_violation - before_slo_violation
            after_query_intervals = [
                Interval(q.start_time_s, q.start_time_s + latency_s)
                for q, latency_s in zip(active_w_current, latencies_after)
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
                        query.start_time_s,
                    )
                )
            after_billed_intervals = Billing.billed_intervals(
                after_query_intervals
            )
            after_cost = self._cost_per_second_per_cluster[
                cluster_name
            ] * Billing.billed_s(after_billed_intervals)
            marginal_cost = after_cost - before_cost

            # Compare and update best if needed.
            if (
                (best_cluster_name is None)
                or (
                    self._slo_cmp_with_tolerance(
                        marginal_slo_violation,
                        marginal_slo_violation_on_best_cluster,
                    )
                    < 0
                )
                or (
                    (
                        self._slo_cmp_with_tolerance(
                            marginal_slo_violation,
                            marginal_slo_violation_on_best_cluster,
                        )
                        == 0
                    )
                    and (marginal_cost < marginal_cost_on_best_cluster)
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
                    for q, latency_s in zip(active_w_current, latencies_after)
                }

            self._log_if_verbose(
                f"\tCluster {cluster_name}: Marginal SLO violation: "
                f"{marginal_slo_violation:.4f}, Marginal cost: "
                f"{marginal_cost:.4f} (Billed intervals before: {before_billed_intervals}, "
                f"after: {after_billed_intervals})"
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
        Write out an HTML visualization of the query assignment.
        Optionally also exports a video if export_video flag is set.
        """
        fig = render_gantt_scrubber(
            self._recorder.snapshots,
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
                snapshots=self._recorder.snapshots,
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
