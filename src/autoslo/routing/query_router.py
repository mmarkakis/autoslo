import logging
import random
from enum import Enum

from intervaltree import Interval  # type: ignore[import]

from autoslo.clusters.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_resolver import SloResolver
from autoslo.slo.slo_objective import SloObjective
from autoslo.utils.billing import Billing
from autoslo.utils.logging import emit_structured
from autoslo.utils.structured_events import (
    RoutingDecisionEvent,
    RoutingScoreEvent,
)
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)


class QueryRouterPolicy(Enum):
    USE_ICONQ_MODEL = "use_iconq_model"
    ROUND_ROBIN = "round_robin"
    UNIFORM_RANDOM = "uniform_random"


class QueryRouter:

    def __init__(
        self,
        slo_resolver: SloResolver,
        slo_metric: SloMetric,
        routing_policy: QueryRouterPolicy = QueryRouterPolicy.USE_ICONQ_MODEL,
    ):
        self._slo_resolver = slo_resolver
        self._slo_metric = slo_metric
        self._routing_policy = routing_policy
        self._round_robin_idx = 0

    @property
    def routing_policy(self) -> QueryRouterPolicy:
        return self._routing_policy

    def route_query(
        self,
        query: Query,
        clusters: dict[str, Cluster],
        iconq_model: IconqModel,
        rel_time_s: float,
    ) -> tuple[str, dict[str, float]]:

        # Collect before-state per cluster
        before_viols_and_costs = {}
        cluster_name_to_queries_to_neighbors = {}
        for cluster_name, cluster in clusters.items():
            before_violation, before_cost = self.compute_slo_metric_and_cost(
                cluster,
                rel_time_s,
            )
            before_viols_and_costs[cluster_name] = (
                before_violation,
                before_cost,
            )

            cluster.add_query(query)
            cluster_name_to_queries_to_neighbors[cluster_name] = (
                cluster.queries_to_neighbors()
            )

        # Perform the prediction and constraint to non-decreasing latency.
        dataset = ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=iconq_model.iconq_interaction_featurizer,
            cluster_to_base_to_neighbors=cluster_name_to_queries_to_neighbors,
        )
        all_predictions = iconq_model.predict_from_dataset(dataset)
        new_predicted_latencies: dict[str, dict[str, float]] = {}
        for cluster_name in all_predictions.keys():
            new_predicted_latencies[cluster_name] = {}
            for query_id, pred in all_predictions[cluster_name].items():
                new_predicted_latencies[cluster_name][query_id] = max(
                    pred.overall_mean_s(),
                    clusters[cluster_name].predicted_latencies.get(
                        query_id, 0.0
                    ),
                )

        # Compute after-states per cluster
        marginal_viols_and_costs = {}
        for cluster_name, cluster in clusters.items():
            before_violation, before_cost = before_viols_and_costs[cluster_name]
            cluster.predicted_latencies = new_predicted_latencies[cluster_name]
            after_violation, after_cost = self.compute_slo_metric_and_cost(
                cluster,
                rel_time_s,
            )
            marginal_violation = after_violation - before_violation
            marginal_cost = after_cost - before_cost
            marginal_viols_and_costs[cluster_name] = (
                marginal_violation,
                marginal_cost,
            )
            latency_s = new_predicted_latencies[cluster_name][query.query_id]
            emit_structured(
                RoutingScoreEvent(
                    rel_time_s=rel_time_s,
                    source="QueryRouter",
                    query_id=query.query_id,
                    query_text_id=query.query_text_id,
                    cluster_name=cluster_name,
                    latency_s=latency_s,
                    marginal_slo_violation=marginal_violation,
                    marginal_cost=marginal_cost,
                )
            )

        # Choose and return best.
        selected_cluster_name = self.select_best(marginal_viols_and_costs)

        selected_marginal = marginal_viols_and_costs[selected_cluster_name]
        selected_latency = new_predicted_latencies[selected_cluster_name][
            query.query_id
        ]
        emit_structured(
            RoutingDecisionEvent(
                rel_time_s=rel_time_s,
                source="QueryRouter",
                query_id=query.query_id,
                query_text_id=query.query_text_id,
                cluster_name=selected_cluster_name,
                latency_s=selected_latency,
                marginal_slo_violation=selected_marginal[0],
                marginal_cost=selected_marginal[1],
            )
        )

        return (
            selected_cluster_name,
            new_predicted_latencies[selected_cluster_name],
        )

    def compute_slo_metric_and_cost(
        self,
        cluster: Cluster,
        rel_time_s: float,
    ) -> tuple[float, float]:
        """
        Compute the cost and SLO-violation metric for a cluster.

        Parameters
        ----------
        cluster:
            The cluster (or clone) whose before-state to compute.
            Must have ``predicted_latencies`` populated for all active queries.
        rel_time_s:
            Relative time in seconds since run start.

        Returns
        -------
        (slo_violation, cost)
        """
        lat_and_slos = []
        intervals = []

        for q in cluster.active_queries:
            lat = cluster.predicted_latencies[q.query_id]
            slo = self._slo_resolver.resolve(q.query_text_id)
            interval = Query.query_interval(q.rel_start_time_s, lat, q.query_id)
            lat_and_slos.append((lat, slo))
            intervals.append(interval)

        slo_violation = self._slo_metric.aggregate_batch(lat_and_slos)

        if cluster.billing_window_start_s is not None:
            intervals.append(
                Interval(cluster.billing_window_start_s, rel_time_s)
            )
        billed_s = Billing.billed_s(intervals)
        cost = cluster.cost_per_second * billed_s

        return slo_violation, cost

    def select_best(
        self,
        marginal_viols_and_costs: dict[str, tuple[float, float]],
    ):
        cluster_names = sorted(marginal_viols_and_costs.keys())

        if self._routing_policy == QueryRouterPolicy.ROUND_ROBIN:
            cluster_name = cluster_names[
                self._round_robin_idx % len(cluster_names)
            ]
            self._round_robin_idx += 1
            return cluster_name

        if self._routing_policy == QueryRouterPolicy.UNIFORM_RANDOM:
            return random.choice(cluster_names)

        # USE_ICONQ_MODEL
        best = cluster_names[0]
        best_marginal_viol_and_cost = marginal_viols_and_costs[best]

        if len(cluster_names) == 1:
            return best

        slo_objective = SloObjective(
            slo_metric=self._slo_metric, slo_threshold=0.0
        )
        # Here the threshold is zero because these are marginal.

        for cluster_name in cluster_names[1:]:
            this_marginal_viol_and_cost = marginal_viols_and_costs[cluster_name]
            if slo_objective.is_b_better(
                best_marginal_viol_and_cost,
                this_marginal_viol_and_cost,
            ):
                best = cluster_name
                best_marginal_viol_and_cost = this_marginal_viol_and_cost

        return best
