import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from intervaltree import Interval  # type: ignore[import]
from pyparsing import Any

from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.routing.query_router import QueryRouter
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query


@dataclass
class RoutingScenario:
    """
    A data class representing a routing scenario for Iconq.
    """

    cluster_name: str
    start_times: dict[str, float]
    predictions: dict[str, ModelPrediction]


class RIconq(QueryRouter):
    """
    A QueryRouter that uses Iconq for routing decisions.
    """

    def __init__(
        self,
        iconq_model_id: str,
        eligible_cluster_names: list[str],
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize an RIconq instance.

        Parameters:
            iconq_model_id: The identifier for the IconqModel to use for routing
                decisions.
            eligible_cluster_names: A list of cluster names that are eligible
                for routing.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """
        # Load the model.
        self._iconq_model_id = iconq_model_id
        self._name = f"RIconq(iconq_model_id={repr(self._iconq_model_id)})"
        self._iconq_model = IconqModel.load(model_id=iconq_model_id)

        # Set up the eligible clusters.
        self._eligible_cluster_names = eligible_cluster_names
        self._blueprint = Blueprint(
            clusters=[
                Cluster.from_config(cluster_name=cluster_name)
                for cluster_name in eligible_cluster_names
            ]
        )
        # From cluster name to a mapping from query ID to Query for
        # running queries.
        self._running_queries: dict[str, dict[str, Query]] = {
            cluster_name: {} for cluster_name in eligible_cluster_names
        }
        self._running_queries_lock = threading.Lock()

    @property
    def name(self) -> str:
        """
        Get the name of the RIconq instance.
        """
        return self._name

    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this RIconq.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint

    def route_query(
        self,
        query_id: Any,
        tpcds_temp_and_q_idx: Any,
        start_time_s: float,
        latency_slo_s: float,
        use_stage_for_isolated_queries: bool = True,
        weigh_by_violation_amount: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Route a query using the Iconq model.

        Parameters:
            query_id: The ID of the query to route.
            tpcds_temp_and_q_idx: The TPC-DS template and query index.
            start_time_s: The start time of the query (in seconds).
            latency_slo_s: The latency SLO for the query (in seconds).
            use_stage_for_isolated_queries: Whether to use stage model
                predictions for isolated queries.
            weigh_by_violation_amount: If True, weigh violations by the amount
                by which they exceed the SLO; otherwise, count violations
                equally.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The cluster name to which the query should be routed.
        """
        featurization = self._iconq_model.iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx
        )

        routing_scenarios: dict[str, RoutingScenario] = {}

        for cluster_name in self._eligible_cluster_names:
            stage_prediction: ModelPrediction = (
                self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                    {query_id: tpcds_temp_and_q_idx}, cluster_name
                )[query_id]
            )
            base_query = Query(
                query_id=query_id,
                tpcds_temp_and_q_idx=tpcds_temp_and_q_idx,
                start_time_s=start_time_s,
                cluster_name=cluster_name,
                featurization=featurization,
                stage_latency_prediction_s=stage_prediction.overall_mean_s(),
            )
            with self._running_queries_lock:
                running_queries = list(
                    self._running_queries[cluster_name].values()
                )

            # Generate the predictions after the contemplated move.
            new_running_queries = running_queries + [base_query]
            dataset = ConcurrentQueryDataset.build_from_query_groups(
                iconq_interaction_featurizer=self._iconq_model.iconq_interaction_featurizer,
                base_queries=new_running_queries,
                query_neighbors={
                    q.query_id: new_running_queries for q in new_running_queries
                },
                use_log_runtime=self._iconq_model.trained_on_log_runtime,
            )
            predictions = self._iconq_model.predict_from_dataset(
                dataset,
            )

            # Add the routing scenario.
            routing_scenarios[cluster_name] = RoutingScenario(
                cluster_name=cluster_name,
                start_times={
                    q.query_id: q.start_time_s for q in new_running_queries
                },
                predictions=predictions,
            )

        # Pick the best scenario.
        best_cluster_name = self._pick_best_among(
            routing_scenarios,
            latency_slo_s=latency_slo_s,
            weigh_by_violation_amount=weigh_by_violation_amount,
        )
        return best_cluster_name

    @staticmethod
    def _pick_best_among(
        scenarios: dict[str, RoutingScenario],
        latency_slo_s: float,
        weigh_by_violation_amount: bool = False,
    ) -> str:
        """
        Pick the best among the given routing scenarios. A scenario is
        considered better if it results in fewer queries violating the latency
        SLO, with ties broken by cost.

        Parameters:
            scenarios: A dictionary mapping cluster names to their respective
                RoutingScenario instances.
            latency_slo_s: The latency SLO in seconds.
            weigh_by_violation_amount: If True, weigh violations by the amount
                by which they exceed the SLO; otherwise, count violations
                equally.

        Returns:
            The cluster name of the best routing scenario.
        """
        best_cluster_name = list(scenarios.keys())[0]
        best_violation = np.inf
        best_cost = np.inf

        for cluster_name, scenario in scenarios.items():
            violation = sum(
                (
                    max(0.0, prediction.overall_mean_s() - latency_slo_s)
                    if weigh_by_violation_amount
                    else int(prediction.overall_mean_s() > latency_slo_s)
                )
                for prediction in scenario.predictions.values()
            )
            billed_s = Billing.billed_s(
                query_intervals=[
                    Interval(
                        scenario.start_times[query_id],
                        scenario.start_times[query_id]
                        + scenario.predictions[query_id].overall_mean_s(),
                    )
                    for query_id in scenario.predictions.keys()
                ]
            )
            cost = billed_s * Cluster.from_config(cluster_name).cost_per_second

            if (violation < best_violation) or (
                (violation == best_violation) and (cost < best_cost)
            ):
                best_cluster_name = cluster_name
                best_violation = violation
                best_cost = cost
        return best_cluster_name

    def on_query_start(
        self,
        query_id: Any,
        cluster_name: str,
        tpcds_temp_and_q_idx: Any,
        start_time_s: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Called when a query starts executing on a cluster.

        Parameters:
            query_id: The ID of the query.
            cluster_name: The name of the cluster where the query is executed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """

        featurization = self._iconq_model.iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx
        )
        stage_prediction: ModelPrediction = (
            self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                {query_id: tpcds_temp_and_q_idx}, cluster_name
            )[query_id]
        )
        query_info = Query(
            query_id=query_id,
            tpcds_temp_and_q_idx=tpcds_temp_and_q_idx,
            start_time_s=start_time_s,
            cluster_name=cluster_name,
            featurization=featurization,
            stage_latency_prediction_s=stage_prediction.overall_mean_s(),
        )
        with self._running_queries_lock:
            self._running_queries[cluster_name][query_id] = query_info

    def on_query_finish(
        self, query_id: Any, cluster_name: str, *args, **kwargs
    ) -> None:
        """
        Called when a query finishes executing.

        Parameters:
            query_id: The ID of the query.
            cluster_name: The name of the cluster where the query is executed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Raises:
            KeyError: If the query ID is not found in the running queries for
            the specified cluster.
        """

        with self._running_queries_lock:
            if query_id not in self._running_queries[cluster_name]:
                raise KeyError(
                    f"Query ID {repr(query_id)} not found in running queries "
                    f"for cluster {cluster_name}."
                )
            del self._running_queries[cluster_name][query_id]
