import logging
import random

import numpy as np

from autoslo.clusters.cluster import ClusterView
from autoslo.config.component_configs import QueryRouterConfig
from autoslo.filesystem.logging import emit_structured
from autoslo.filesystem.structured_events import EventType, QueryRelatedEvent
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.routing.query_router_policy import QueryRouterPolicy
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)


class QueryRouter:
    def __init__(
        self,
        slo_resolver: SloResolver,
        slo_objective: SloObjective,
        query_router_config: QueryRouterConfig,
        iconq_model: IconqModel,
        rel_time_s_to_forecasted_table_vecs: dict[float, np.ndarray],
    ):
        self._slo_resolver = slo_resolver
        self._slo_objective = slo_objective
        self._routing_policy = QueryRouterPolicy(
            query_router_config.routing_policy_name
        )
        self._iconq_model = iconq_model
        self._round_robin_idx = 0
        self._query_router_config = query_router_config
        self._rel_time_s_to_forecasted_table_vecs = (
            rel_time_s_to_forecasted_table_vecs
        )
        self._sorted_forecast_times = sorted(
            self._rel_time_s_to_forecasted_table_vecs.keys()
        )
        self._idx_into_forecast_sequence = 0

    @property
    def routing_policy(self) -> QueryRouterPolicy:
        return self._routing_policy

    def route_query(
        self,
        query: Query,
        snapshot: dict[str, ClusterView],
        rel_time_s: float,
    ) -> tuple[str, dict[str, float], np.ndarray]:

        # Collect before-state raw pairs and cost per cluster, and build the
        # hypothetical neighbor map for each cluster as if *query* were added.
        before_pairs: dict[str, list[LatencySlo]] = {}
        before_costs: dict[str, float] = {}
        before_cache_states: dict[str, np.ndarray] = {}
        cluster_name_to_queries_to_neighbors = {}
        for cluster_name, cluster in snapshot.items():
            pairs = self._collect_cluster_pairs(
                queries=cluster.active_queries,
                predicted_latencies=cluster.predicted_latencies,
            )
            cost = cluster.cost_until(rel_time_s)
            before_pairs[cluster_name] = pairs
            before_costs[cluster_name] = cost
            before_cache_states[cluster_name] = cluster.cache_state

            cluster_name_to_queries_to_neighbors[cluster_name] = (
                cluster.hypothetical_neighbors_with(query)
            )

        # Perform the prediction and constraint to non-decreasing latency.
        dataset = ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=self._iconq_model.iconq_interaction_featurizer,
            cluster_to_base_to_neighbors=cluster_name_to_queries_to_neighbors,
        )
        all_predictions = self._iconq_model.predict_from_dataset(dataset)
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

        # Retrieve the appropriate forecasted query vecs for this time.
        while (
            self._idx_into_forecast_sequence
            < (len(self._sorted_forecast_times) - 1)
        ) and (
            self._sorted_forecast_times[self._idx_into_forecast_sequence + 1]
            <= rel_time_s
        ):
            self._idx_into_forecast_sequence += 1
        forecasted_table_vecs = self._rel_time_s_to_forecasted_table_vecs[
            self._sorted_forecast_times[self._idx_into_forecast_sequence]
        ]

        # For each candidate cluster, compute the global after-state
        # (aggregating raw pairs across ALL clusters, with the candidate
        # cluster using updated predictions).
        all_after_viols_and_costs: dict[str, ViolationCost] = {}
        all_new_cache_states: dict[str, np.ndarray] = {}
        all_cache_risks: dict[str, float] = {}
        for candidate_name, cluster in snapshot.items():
            after_pairs = self._collect_cluster_pairs(
                queries=cluster.active_queries + [query],
                predicted_latencies=new_predicted_latencies[candidate_name],
            )
            after_cost = cluster.cost_with_query_start_until(
                query_start_s=query.rel_start_time_s,
                rel_time_s=rel_time_s,
            )

            # Build pair and cluster state list: updated for candidate,
            # unchanged for all others.
            all_after_pairs: list[LatencySlo] = list(after_pairs)
            total_after_cost = after_cost
            new_cache_state = self._updated_cluster_state(
                current_state=before_cache_states[candidate_name],
                table_vector=self._iconq_model.iconq_query_featurizer.table_vector_for(
                    query.query_text_id
                ),
            )
            after_cache_states: list[np.ndarray] = [new_cache_state]
            for other_name in snapshot:
                if other_name != candidate_name:
                    all_after_pairs.extend(before_pairs[other_name])
                    total_after_cost += before_costs[other_name]
                    after_cache_states.append(before_cache_states[other_name])

            after_violation = self._slo_objective.slo_metric.aggregate_batch(
                all_after_pairs
            )
            cache_risk = self._score_cache_risk(
                caches_per_cluster=np.stack(after_cache_states, axis=0),
                forecasted_table_vecs=forecasted_table_vecs,
            )
            all_after_viols_and_costs[candidate_name] = ViolationCost(
                after_violation, total_after_cost
            )
            all_new_cache_states[candidate_name] = new_cache_state
            all_cache_risks[candidate_name] = cache_risk

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
                        "cache_risk": cache_risk,
                    },
                    query_id=query.query_id,
                    query_text_id=query.query_text_id,
                )
            )

        # Choose and return best.
        selected_cluster_name = self.select_best(
            all_after_viols_and_costs, all_cache_risks
        )
        selected = all_after_viols_and_costs[selected_cluster_name]
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
                    "slo_violation": selected.violation,
                    "cost": selected.cost,
                    "cache_risk": all_cache_risks[selected_cluster_name],
                },
                query_id=query.query_id,
                query_text_id=query.query_text_id,
            )
        )

        return (
            selected_cluster_name,
            new_predicted_latencies[selected_cluster_name],
            all_new_cache_states[selected_cluster_name],
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
        viols_and_costs: dict[str, ViolationCost],
        cache_risks: dict[str, float],
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

        if self._routing_policy == QueryRouterPolicy.CACHE_AWARE:
            adjusted_tups: list[ViolationCost] = []
            for cn in cluster_names:
                violation = viols_and_costs[cn].violation
                base_cost = viols_and_costs[cn].cost
                added_cost = (
                    base_cost
                    * self._query_router_config.cache_risk_cost_multiplier
                    * cache_risks[cn]
                )
                adjusted_tups.append(
                    ViolationCost(
                        violation=violation,
                        cost=base_cost + added_cost,
                    )
                )
            best_local_idx = self._slo_objective.idx_of_best(adjusted_tups)
            return cluster_names[best_local_idx]

        # Default: USE_ICONQ_MODEL.
        best_local_idx = self._slo_objective.idx_of_best(
            [viols_and_costs[cn] for cn in cluster_names]
        )
        return cluster_names[best_local_idx]

    def _updated_cluster_state(
        self,
        current_state: np.ndarray,
        table_vector: np.ndarray,
    ) -> np.ndarray:
        alpha = self._query_router_config.cluster_cache_state_update_alpha
        return alpha * current_state + (1 - alpha) * table_vector

    def _score_cache_risk(
        self,
        caches_per_cluster: np.ndarray,
        forecasted_table_vecs: np.ndarray,
    ) -> float:
        """
        Compute the scalar cache risk score.

        Parameters
        ----------
        caches_per_cluster :
            Array of shape (C, N) where C is the number of clusters and N
            is the dimensionality of the cache state vector.
        forecasted_table_vecs :
            Array of shape (K, N) where K is the number of forecasted queries
            and N is the same dimensionality as above. Each row is a single
            query, so that differences in relative frequency of arrival are
            captured by the number of rows per query type.

        Returns
        -------
        float
            Scalar risk score in the range [0, 1], where higher means more risk
            of cache-unfavorableness.
        """

        # Normalize
        caches_per_cluster_norms = np.linalg.norm(
            caches_per_cluster, axis=1, keepdims=True
        )
        forecasted_table_vecs_norms = np.linalg.norm(
            forecasted_table_vecs, axis=1, keepdims=True
        )
        epsilon = self._query_router_config.cache_risk_epsilon
        caches_per_cluster_safe = caches_per_cluster / np.maximum(
            caches_per_cluster_norms, epsilon
        )
        forecasted_table_vecs_safe = forecasted_table_vecs / np.maximum(
            forecasted_table_vecs_norms, epsilon
        )

        # Compute cosine similarities (C, K)
        cosine_similarities = (
            caches_per_cluster_safe @ forecasted_table_vecs_safe.T
        )

        # For each query, find the maximum similarity across clusters (K,)
        max_similarities = np.max(cosine_similarities, axis=0)

        # The risk score is the minimum max_similarity across the relevant
        # queries. The relevant queries are determined by `cache_risk_coverage`.
        # This is equivalent to finding the (100 - coverage)th percentile of the
        # max similarities.
        coverage_pct = (1 - self._query_router_config.cache_risk_coverage) * 100
        risk_score = np.percentile(max_similarities, max(0, coverage_pct))
        return risk_score
