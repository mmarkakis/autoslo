import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_definition.query import Query
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset

from intervaltree import Interval  # type: ignore[import]

from autoslo.utils.billing import Billing

from math import isclose

from autoslo.blueprints.blueprint import Blueprint


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

        workload = Chunk.load(workload_name)  # FIXME: generalize to workloads.
        self._workload = workload

        self._run_id = simulator_run_id or str(int(datetime.now().timestamp()))

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
                f"{self._optimize_based_on_slo_violation_amount}"
                f"with thresholds: "
                f"slo_violation_rate_threshold = "
                f"{self._slo_violation_rate_threshold}, "
                f"slo_violation_amount_threshold_s = "
                f"{self._slo_violation_amount_threshold_s}, "
                f"slo_s = "
                f"{self._slo_s}"
            )

        # Set up bookkeeping etc.

    def first_pass(self) -> None:
        """
        First pass: route queries as they come in, preferring active endpoints
        and minimizing SLO violations.
        """

        active_queries_per_cluster: dict[Cluster, list[Query]] = {
            Cluster.from_config(cluster_name): []
            for cluster_name in self._blueprint.cluster_names
        }

        for i, query in enumerate(self._workload.queries):
            # Add query to the right cluster.
            (
                selected_cluster,
                updated_query,
                latencies_on_best_cluster_s,
            ) = self._find_best_cluster_for_query(
                query,
                active_queries_per_cluster,
            )
            active_queries_per_cluster[selected_cluster].append(updated_query)

            # Go through the active queries on the best cluster and update their
            # latencies based on the prediction results.
            for q in active_queries_per_cluster[selected_cluster]:
                q.latency_s = latencies_on_best_cluster_s[q.query_id]

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
        active_queries_per_cluster: dict[Cluster, list[Query]],
    ) -> tuple[Cluster, Query, dict[str, float]]:
        """
        Finds the best cluster to route the query to, based on the projected
        latency and SLO violation amount.

        Parameters:
            query: The query to route.
            active_queries_per_cluster: A dictionary mapping clusters to their
                currently active queries.

        Returns:
            best_cluster: The cluster that was chosen as the best for routing
                    the query.
            query: The input query with updated fields reflecting the best
                cluster choice and latency prediction.
            latencies_on_best_cluster_s: A dictionary mapping query IDs to their
                projected latencies on the best cluster.
        """

        # Best bookkeeping.
        best_cluster = None
        marginal_slo_violation_on_best_cluster = float("inf")
        marginal_cost_on_best_cluster = float("inf")
        stage_latency_prediction_on_best_cluster_s = float("inf")
        latencies_on_best_cluster_s: dict[str, float] = {}

        query.featurization = self._iconq_model.iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
            query.tpcds_temp_and_q_idx
        )

        for cluster, active_queries in active_queries_per_cluster.items():

            query.cluster_name = cluster.name
            query.stage_latency_prediction_s = (
                self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                    {query.query_id: query.tpcds_temp_and_q_idx}, cluster.name
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
            before_cost = cluster.cost_per_second * Billing.billed_s(
                [q.as_interval() for q in active_queries]
            )
            # TODO: account for the minimum duration properly in case there was
            # active time right before the current active set?

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
            latencies = [pred.overall_mean_s() for pred in predictions.values()]

            # Process prediction results.
            individual_after_slo_violations: list[float] | list[bool] = [
                (
                    max(0, latency_s - self._slo_s)
                    if self._optimize_based_on_slo_violation_amount
                    else latency_s > self._slo_s
                )
                for latency_s in latencies
            ]
            after_slo_violation = sum(individual_after_slo_violations)
            marginal_slo_violation = after_slo_violation - before_slo_violation

            after_cost = cluster.cost_per_second * Billing.billed_s(
                [
                    Interval(q.start_time_s, q.start_time_s + latency_s)
                    for q, latency_s in zip(active_w_current, latencies)
                ]
            )
            # TODO: account for the minimum duration properly in case there was
            # active time right before the current active set?
            marginal_cost = after_cost - before_cost

            # Compare and update best if needed.
            if (
                (best_cluster is None)
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
                best_cluster = cluster
                marginal_slo_violation_on_best_cluster = marginal_slo_violation
                marginal_cost_on_best_cluster = marginal_cost
                stage_latency_prediction_on_best_cluster_s = (
                    query.stage_latency_prediction_s
                )
                latencies_on_best_cluster_s = {
                    q.query_id: latency_s
                    for q, latency_s in zip(active_w_current, latencies)
                }

        assert best_cluster is not None

        # Update the query's cluster and stage latency prediction to reflect the
        # best cluster choice.
        query.cluster_name = best_cluster.name
        query.stage_latency_prediction_s = (
            stage_latency_prediction_on_best_cluster_s
        )
        query.latency_s = latencies_on_best_cluster_s[query.query_id]
        query.latency_is_lower_bound = False

        return (best_cluster, query, latencies_on_best_cluster_s)
