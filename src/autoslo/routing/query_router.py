import bisect
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from autoslo.clusters.cluster import ClusterView, cluster_cost_until_drained
from autoslo.config.component_configs import QueryRouterConfig
from autoslo.filesystem.structured_events import EventType, QueryRelatedEvent
from autoslo.filesystem.structured_log import emit_structured
from autoslo.forecasting.forecaster import Forecaster
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.nn.lstm_state import AfterLSTMState
from autoslo.routing.query_router_policy import QueryRouterPolicy
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import ClusterAwareQueryId, Query
from autoslo.workload_definition.workload import Workload

logger = logging.getLogger(__name__)


@dataclass
class QueryRouterState:
    """Snapshot of the mutable state inside a QueryRouter."""

    round_robin_idx: int = 0


class QueryRouter:
    def __init__(
        self,
        slo_resolver: SloResolver,
        slo_objective: SloObjective,
        query_router_config: QueryRouterConfig,
        out_dir: Path,
        iconq_model: Optional[IconqModel] = None,
        source_for_log_records: str = "QueryRouter",
    ):
        self._slo_resolver = slo_resolver
        self._slo_objective = slo_objective
        self._routing_policy = QueryRouterPolicy(
            query_router_config.routing_policy_name
        )
        self._iconq_model = (
            iconq_model
            if iconq_model is not None
            else IconqModel.load(
                query_router_config.iconq_model_id, inference_mode=True
            )
        )
        self._round_robin_idx = 0
        self._query_router_config = query_router_config
        self._source_for_log_records = source_for_log_records
        self._rel_time_s_to_forecasted_table_vecs = (
            self._read_or_derive_rel_time_s_to_forecasted_table_vecs(out_dir)
        )
        self._sorted_forecast_times = sorted(
            self._rel_time_s_to_forecasted_table_vecs.keys()
        )

    def _read_or_derive_rel_time_s_to_forecasted_table_vecs(
        self, out_dir: Path
    ) -> dict[float, np.ndarray]:

        out_path = Path(out_dir) / "rel_time_s_to_forecasted_table_vecs.npz"
        if self._routing_policy != QueryRouterPolicy.CACHE_AWARE:
            return {}
        elif out_path.exists():
            loaded = np.load(out_path, allow_pickle=True)
            return {float(k): v for k, v in loaded.items()}
        else:
            forecaster_config = self._query_router_config.forecaster_config
            if (forecaster_config is None) or (
                forecaster_config.reservoir_config is None
            ):
                raise ValueError(
                    "forecaster_config and reservoir_config must be provided "
                    "for CACHE_AWARE routing policy"
                )
            reservoir_config = forecaster_config.reservoir_config

            target_date = (
                pd.Timestamp(reservoir_config.last_day_date_inclusive)
                + pd.Timedelta(days=1)
            ).date()
            forecaster = Forecaster(forecaster_config=forecaster_config)
            forecasted_workload_config = forecaster.forecast(
                target_date=target_date,
                use_fixed_queries_per_hour=True,
                out_dir=out_dir,
                workload_name="forecasted_workload",
            )
            forecasted_workload = Workload(
                workload_config=forecasted_workload_config
            )
            forecasted_workload = forecasted_workload.rescale_rel_times(
                forecaster_config.rescale_factor
            )
            rel_time_s_to_forecasted_table_vecs = forecasted_workload.get_rel_time_s_to_table_vecs(
                iconq_query_featurizer=self._iconq_model.iconq_query_featurizer
            )

            np.savez(
                out_path,
                **{
                    str(rel_time_s): vec
                    for rel_time_s, vec in rel_time_s_to_forecasted_table_vecs.items()
                },
            )  # type: ignore[arg-type]
            return rel_time_s_to_forecasted_table_vecs

    @property
    def routing_policy(self) -> QueryRouterPolicy:
        return self._routing_policy

    @property
    def iconq_model(self) -> IconqModel:
        return self._iconq_model

    def get_state(self) -> QueryRouterState:
        """Return a snapshot of the router's mutable sequencing state."""
        return QueryRouterState(round_robin_idx=self._round_robin_idx)

    def set_state(self, state: QueryRouterState) -> None:
        """Restore the router to a previously captured state snapshot."""
        self._round_robin_idx = state.round_robin_idx

    def route_query(
        self,
        query: Query,
        snapshot: dict[str, ClusterView],
        rel_time_s: float,
    ) -> tuple[str, dict[str, float], np.ndarray, dict[str, AfterLSTMState]]:

        use_stage = self._routing_policy == QueryRouterPolicy.USE_STAGE_MODEL

        # Under USE_STAGE_MODEL, look up each query's pre-computed stage-model
        # prediction for the cluster's RPU. These are used purely for
        # routing-time scoring (SLO pairs and cost drain times). The iconq
        # forward pass below still runs and its predictions are still what we
        # return downstream, so cluster state stays accurate.
        stage_latencies: dict[str, dict[str, float]] = {}
        if use_stage:
            for cluster_name, cluster in snapshot.items():
                stage_latencies[cluster_name] = {
                    q.query_id: q.stage_predictions_per_rpu[cluster.rpu]
                    for q in cluster.active_queries
                }

        # Collect before-state raw pairs and cost per cluster, and build the
        # hypothetical neighbor map for each cluster as if *query* were added.
        before_pairs: dict[str, list[LatencySlo]] = {}
        before_costs: dict[str, float] = {}
        before_cache_states: dict[str, np.ndarray] = {}
        cluster_name_to_queries_to_neighbors = {}
        for cluster_name, cluster in snapshot.items():
            before_latencies = (
                stage_latencies[cluster_name]
                if use_stage
                else cluster.predicted_latencies
            )
            pairs = self._collect_cluster_pairs(
                queries=cluster.active_queries,
                predicted_latencies=before_latencies,
            )
            cost = cluster_cost_until_drained(
                queries=cluster.active_queries,
                predicted_latencies=before_latencies,
                billing_accumulator=cluster.billing_accumulator,
                billing_window_start_s=cluster.billing_window_start_s,
                cost_per_second=cluster.cost_per_second,
                current_rel_time_s=rel_time_s,
            )
            before_pairs[cluster_name] = pairs
            before_costs[cluster_name] = cost
            before_cache_states[cluster_name] = cluster.cache_state

            cluster_name_to_queries_to_neighbors[cluster_name] = (
                cluster.hypothetical_neighbors_with(query)
            )

        # Perform the prediction and constraint to non-decreasing latency.
        incremental_inference_possible = (
            self._iconq_model.supports_stateful_inference
            and all(
                q.query_id in cv.lstm_states
                for cv in snapshot.values()
                for q in cv.active_queries
            )
        )
        new_states: dict[ClusterAwareQueryId, AfterLSTMState] = {}
        if incremental_inference_possible:
            iconq_predictions, new_states = self._predict_stateful(
                snapshot, query
            )
        else:
            iconq_predictions = self._iconq_model.predict_from_query_groups(
                cluster_name_to_queries_to_neighbors
            )
        iconq_predicted_latencies: dict[str, dict[str, float]] = {}
        iconq_new_states: dict[str, dict[str, AfterLSTMState]] = {}
        for cluster_aware_query_id, pred in iconq_predictions.items():
            cluster_name = cluster_aware_query_id.cluster_name
            if cluster_name not in iconq_predicted_latencies:
                iconq_predicted_latencies[cluster_name] = {}
            query_id = cluster_aware_query_id.query_id
            cluster_view = snapshot[cluster_name]
            prev_latency_prediction_s = cluster_view.predicted_latencies.get(
                query_id, 0.0
            )
            query_entry = cluster_view.queries.get(query_id, None)
            latency_so_far_s = 0.0
            if query_entry is not None:
                latency_so_far_s = rel_time_s - query_entry.rel_start_time_s
            iconq_predicted_latencies[cluster_name][query_id] = max(
                pred.overall_mean_s(),
                prev_latency_prediction_s,
                latency_so_far_s,
            )
        for cluster_aware_query_id, state in new_states.items():
            cluster_name = cluster_aware_query_id.cluster_name
            if cluster_name not in iconq_new_states:
                iconq_new_states[cluster_name] = {}
            query_id = cluster_aware_query_id.query_id
            iconq_new_states[cluster_name][query_id] = state

        # Retrieve the appropriate forecasted query vecs for this time.
        forecasted_table_vecs: Optional[np.ndarray] = None
        if self._routing_policy == QueryRouterPolicy.CACHE_AWARE:
            idx = max(
                0,
                bisect.bisect_right(self._sorted_forecast_times, rel_time_s)
                - 1,
            )
            forecasted_table_vecs = self._rel_time_s_to_forecasted_table_vecs[
                self._sorted_forecast_times[idx]
            ]

        # For each candidate cluster, compute the global after-state
        # (aggregating raw pairs across ALL clusters, with the candidate
        # cluster using updated predictions).
        all_after_viols_and_costs: dict[str, ViolationCost] = {}
        all_new_cache_states: dict[str, np.ndarray] = {}
        all_cache_risks: dict[str, float] = {}
        for candidate_name, cluster in snapshot.items():
            after_latencies = (
                stage_latencies[candidate_name]
                | {query.query_id: query.stage_predictions_per_rpu[cluster.rpu]}
                if use_stage
                else iconq_predicted_latencies[candidate_name]
            )
            after_queries = cluster.active_queries + [query]
            after_pairs = self._collect_cluster_pairs(
                queries=after_queries,
                predicted_latencies=after_latencies,
            )
            after_cost = cluster_cost_until_drained(
                queries=after_queries,
                predicted_latencies=after_latencies,
                billing_accumulator=cluster.billing_accumulator,
                billing_window_start_s=cluster.billing_window_start_s,
                cost_per_second=cluster.cost_per_second,
                current_rel_time_s=rel_time_s,
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

            after_violation = round(
                self._slo_objective.slo_metric.aggregate_batch(all_after_pairs),
                3,
            )
            total_after_cost = round(total_after_cost, 3)
            cache_risk = self._score_cache_risk(
                caches_per_cluster=np.stack(after_cache_states, axis=0),
                forecasted_table_vecs=forecasted_table_vecs,
            )
            all_after_viols_and_costs[candidate_name] = ViolationCost(
                after_violation, total_after_cost
            )
            all_new_cache_states[candidate_name] = new_cache_state
            all_cache_risks[candidate_name] = cache_risk

            latency_s = round(after_latencies[query.query_id], 3)
            emit_structured(
                QueryRelatedEvent(
                    rel_time_s=rel_time_s,
                    event_type=EventType.ROUTING_SCORE,
                    source=self._source_for_log_records,
                    cluster_name=candidate_name,
                    details={
                        "latency_s_for_routing": latency_s,
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
        selected_latency = round(
            (
                query.stage_predictions_per_rpu[snapshot[selected_cluster_name].rpu]
                if use_stage
                else iconq_predicted_latencies[selected_cluster_name][
                    query.query_id
                ]
            ),
            3,
        )
        emit_structured(
            QueryRelatedEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.ROUTING,
                source=self._source_for_log_records,
                cluster_name=selected_cluster_name,
                details={
                    "latency_s_for_routing": selected_latency,
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
            iconq_predicted_latencies[selected_cluster_name],
            all_new_cache_states[selected_cluster_name],
            iconq_new_states.get(selected_cluster_name, {}),
        )

    def _predict_stateful(
        self,
        snapshot: dict[str, ClusterView],
        query: Query,
    ) -> tuple[
        dict[ClusterAwareQueryId, ModelPrediction],
        dict[ClusterAwareQueryId, AfterLSTMState],
    ]:
        """Score phase of the stateful routing path.

        For each candidate cluster:
          * Each already-active query is scored via one incremental
            ``predict_incremental`` call (O(1) LSTM step).
          * The newly arriving *query* is scored via a full
            ``compute_initial_state`` call (requires full forward+after pass).

        Returns a dict in the same shape as ``predict_from_dataset``.
        State is **not mutated** here; the commit phase runs later in
        ``route_query`` after ``select_best`` has been called.
        """
        predictions: dict[ClusterAwareQueryId, ModelPrediction] = {}
        new_states: dict[ClusterAwareQueryId, AfterLSTMState] = {}

        # Batch all incremental steps across every cluster into one GPU call.
        incremental_items: list[tuple[AfterLSTMState, ClusterAwareQueryId]] = []
        for cluster_name, cluster in snapshot.items():
            for q_active in cluster.active_queries:
                caqi = ClusterAwareQueryId.make(cluster_name, q_active.query_id)
                incremental_items.append(
                    (cluster.lstm_states[q_active.query_id], caqi)
                )
        for caqi, (pred, state) in self._iconq_model.predict_incremental_batch(
            incremental_items, query
        ).items():
            predictions[caqi] = pred
            new_states[caqi] = state

        # Initial pass: batch all clusters into one GPU call.  Sequences are
        # padded to a common length; pack_padded_sequence in get_after_final_state
        # ensures each cluster's state is taken at its true last position.
        initial_items = [
            (
                ClusterAwareQueryId.make(cluster_name, query.query_id),
                query,
                cluster.active_queries,
            )
            for cluster_name, cluster in snapshot.items()
        ]
        for caqi, (pred, state) in self._iconq_model.predict_initial_batch(
            initial_items
        ).items():
            predictions[caqi] = pred
            new_states[caqi] = state

        return predictions, new_states

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

        # Default: USE_ICONQ_MODEL or USE_STAGE_MODEL. Both score using the
        # raw (violation, cost) tuples; they differ only in which model
        # supplies the latencies populating those tuples upstream.
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
        forecasted_table_vecs: Optional[np.ndarray] = None,
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
        # If no forecast is loaded, eeffectively ignore this term.
        if forecasted_table_vecs is None:
            return 0.0

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
