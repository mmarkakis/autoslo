import logging
from typing import Optional

from intervaltree import Interval  # type: ignore[import]

from autoslo.clusters.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_resolver import SloResolver
from autoslo.slo.slo_objective import SloObjective
from autoslo.utils.billing import Billing
from autoslo.utils.structured_log import LOGGER_NAME, emit_structured
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)
_has_structured = lambda: bool(logging.getLogger(LOGGER_NAME).handlers)


class QueryRouter:

    def __init__(
        self,
        slo_resolver: SloResolver,
        slo_metric: SloMetric,
        round_robin_cluster_names: Optional[list[str]] = None,
    ):
        self._slo_resolver = slo_resolver
        self._slo_metric = slo_metric
        self._round_robin_names = round_robin_cluster_names
        self._round_robin_idx = 0

    @property
    def round_robin_names(self) -> Optional[list[str]]:
        return self._round_robin_names

    @property
    def round_robin_idx(self) -> int:
        return self._round_robin_idx

    def route_query(
        self,
        query: Query,
        clusters: dict[str, Cluster],
        initial_predicted_latencies: dict[str, dict[str, float]],
        iconq_model: IconqModel,
        current_time_s: float,
    ) -> tuple[str, dict[str, float]]:

        # Collect before-state per cluster
        before_viols_and_costs = {}
        cluster_name_to_queries_to_neighbors = {}
        for cluster_name, cluster in clusters.items():
            before_violation, before_cost = self.compute_slo_metric_and_cost(
                cluster,
                current_time_s,
                initial_predicted_latencies.get(cluster_name, {}),
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
                    initial_predicted_latencies.get(cluster_name, {}).get(
                        query_id, 0.0
                    ),
                )

        # Compute after-states per cluster
        marginal_viols_and_costs = {}
        for cluster_name, cluster in clusters.items():
            before_violation, before_cost = before_viols_and_costs[cluster_name]
            after_violation, after_cost = self.compute_slo_metric_and_cost(
                cluster, current_time_s, new_predicted_latencies[cluster_name]
            )
            marginal_violation = after_violation - before_violation
            marginal_cost = after_cost - before_cost
            marginal_viols_and_costs[cluster_name] = (
                marginal_violation,
                marginal_cost,
            )
            if _has_structured():
                record = {
                    "timestamp": current_time_s,
                    "source": "routing",
                    "event_type": "routing_score",
                    "query_id": query.query_id,
                    "cluster_name": cluster_name,
                    "end_time_s": (
                        query.rel_start_time_s
                        + all_predictions[cluster_name][
                            query.query_id
                        ].overall_mean_s()
                    ),
                    "marginal_slo_violation": marginal_violation,
                    "marginal_cost": marginal_cost,
                }
                emit_structured(record)

        # Choose and return best.
        selected_cluster_name = self.select_best(marginal_viols_and_costs)

        if _has_structured():
            emit_structured(
                {
                    "timestamp": current_time_s,
                    "event_type": "routing",
                    "source": "router",
                    "query_id": str(query.query_id),
                    "query_text_id": str(query.query_text_id),
                    "cluster_name": selected_cluster_name,
                }
            )

        return (
            selected_cluster_name,
            new_predicted_latencies[selected_cluster_name],
        )

    def compute_slo_metric_and_cost(
        self,
        cluster: Cluster,
        current_time_s: float,
        predicted_latencies: dict[str, float],
    ) -> tuple[float, float]:
        """
        Compute the cost and SLO-violation metric for a cluster.

        Parameters
        ----------
        cluster:
            The cluster (or clone) whose before-state to compute.
        current_time_s:
            Wall-clock (or simulated) time of the incoming query's arrival.
        latencies:
            ``{query_id: predicted_latency_s}`` for the currently-active queries.

        Returns
        -------
        (slo_violation, cost)
        """
        lat_and_slos = []
        intervals = []

        for q in cluster.active_queries:
            lat = predicted_latencies[q.query_id]
            slo = self._slo_resolver.resolve(q.query_text_id)
            interval = Query.query_interval(q.rel_start_time_s, lat, q.query_id)
            lat_and_slos.append((lat, slo))
            intervals.append(interval)

        slo_violation = self._slo_metric.aggregate_batch(lat_and_slos)

        if cluster.billing_window_start_s is not None:
            intervals.append(
                Interval(cluster.billing_window_start_s, current_time_s)
            )
        billed_s = Billing.billed_s(intervals)
        cost = cluster.cost_per_second * billed_s

        return slo_violation, cost

    def select_best(
        self,
        marginal_viols_and_costs: dict[str, tuple[float, float]],
    ):
        if self._round_robin_names:
            cluster_name = self._round_robin_names[self._round_robin_idx]
            self._round_robin_idx = (self._round_robin_idx + 1) % len(
                self._round_robin_names
            )
            return cluster_name

        cluster_names = list(marginal_viols_and_costs.keys())
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
