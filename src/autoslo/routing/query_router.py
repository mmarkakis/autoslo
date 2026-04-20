import logging
import random
from enum import Enum

from autoslo.clusters.cluster import ClusterView
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.logging import emit_structured
from autoslo.utils.structured_events import EventType, QueryRelatedEvent
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
        slo_objective: SloObjective,
        routing_policy: QueryRouterPolicy = QueryRouterPolicy.USE_ICONQ_MODEL,
    ):
        self._slo_resolver = slo_resolver
        self._slo_objective = slo_objective
        self._routing_policy = routing_policy
        self._round_robin_idx = 0

    @property
    def routing_policy(self) -> QueryRouterPolicy:
        return self._routing_policy

    def route_query(
        self,
        query: Query,
        snapshot: dict[str, ClusterView],
        iconq_model: IconqModel,
        rel_time_s: float,
    ) -> tuple[str, dict[str, float]]:

        # Collect before-state raw pairs and cost per cluster, and build the
        # hypothetical neighbor map for each cluster as if *query* were added.
        before_pairs: dict[str, list[LatencySlo]] = {}
        before_costs: dict[str, float] = {}
        cluster_name_to_queries_to_neighbors = {}
        for cluster_name, cluster in snapshot.items():
            pairs = self._collect_cluster_pairs(
                queries=cluster.active_queries,
                predicted_latencies=cluster.predicted_latencies,
            )
            cost = cluster.cost_until(rel_time_s)
            before_pairs[cluster_name] = pairs
            before_costs[cluster_name] = cost

            cluster_name_to_queries_to_neighbors[cluster_name] = (
                cluster.hypothetical_neighbors_with(query)
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
                    snapshot[cluster_name].predicted_latencies.get(
                        query_id, 0.0
                    ),
                )

        # For each candidate cluster, compute the global after-state
        # (aggregating raw pairs across ALL clusters, with the candidate
        # cluster using updated predictions).
        all_after_viols_and_costs: dict[str, tuple[float, float]] = {}
        for candidate_name, cluster in snapshot.items():
            after_pairs = self._collect_cluster_pairs(
                queries=cluster.active_queries + [query],
                predicted_latencies=new_predicted_latencies[candidate_name],
            )
            after_cost = cluster.cost_with_query_start_until(
                query_start_s=query.rel_start_time_s,
                rel_time_s=rel_time_s,
            )

            # Build pair list: updated pairs for candidate,
            # unchanged before-pairs for all others.
            all_after_pairs: list[LatencySlo] = list(after_pairs)
            total_after_cost = after_cost
            for other_name in snapshot:
                if other_name != candidate_name:
                    all_after_pairs.extend(before_pairs[other_name])
                    total_after_cost += before_costs[other_name]

            after_violation = self._slo_objective.slo_metric.aggregate_batch(
                all_after_pairs
            )
            all_after_viols_and_costs[candidate_name] = (
                after_violation,
                total_after_cost,
            )

            latency_s = new_predicted_latencies[candidate_name][query.query_id]
            emit_structured(
                QueryRelatedEvent(
                    rel_time_s=rel_time_s,
                    event_type=EventType.ROUTING_SCORE,
                    source="QueryRouter",
                    cluster_name=candidate_name,
                    details={
                        "latency_s": latency_s,
                        "slo_violation": after_violation,
                        "cost": total_after_cost,
                    },
                    query_id=query.query_id,
                    query_text_id=query.query_text_id,
                )
            )

        # Choose and return best.
        selected_cluster_name = self.select_best(all_after_viols_and_costs)
        selected_viol, selected_cost = all_after_viols_and_costs[
            selected_cluster_name
        ]
        selected_latency = new_predicted_latencies[selected_cluster_name][
            query.query_id
        ]
        emit_structured(
            QueryRelatedEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.ROUTING,
                source="QueryRouter",
                cluster_name=selected_cluster_name,
                details={
                    "latency_s": selected_latency,
                    "slo_violation": selected_viol,
                    "cost": selected_cost,
                },
                query_id=query.query_id,
                query_text_id=query.query_text_id,
            )
        )

        return (
            selected_cluster_name,
            new_predicted_latencies[selected_cluster_name],
        )

    def _collect_cluster_pairs(
        self,
        queries: list[Query],
        predicted_latencies: dict[str, float],
    ) -> list[LatencySlo]:
        """
        Collect raw (latency, slo) pairs from the provided query set and
        latency map, without mutating any cluster.

        Parameters
        ----------
        queries:
            The active queries to evaluate.
        predicted_latencies:
            Mapping of query_id → predicted latency in seconds.
            Must contain an entry for every query in *queries*.
        Returns:
            The list of (latency, slo) pairs.
        """
        lat_and_slos: list[LatencySlo] = []

        for q in queries:
            lat = predicted_latencies[q.query_id]
            slo = self._slo_resolver.resolve(q.query_text_id)
            lat_and_slos.append(LatencySlo(lat, slo))

        return lat_and_slos

    def select_best(
        self,
        viols_and_costs: dict[str, tuple[float, float]],
    ):
        cluster_names = sorted(viols_and_costs.keys())

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
        best_viol_and_cost = viols_and_costs[best]

        if len(cluster_names) == 1:
            return best

        for cluster_name in cluster_names[1:]:
            this_viol_and_cost = viols_and_costs[cluster_name]
            if self._slo_objective.is_b_better(
                best_viol_and_cost,
                this_viol_and_cost,
            ):
                best = cluster_name
                best_viol_and_cost = this_viol_and_cost

        return best
