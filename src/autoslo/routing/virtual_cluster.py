from dataclasses import dataclass, field

from autoslo.routing.managed_cluster_pool import ClusterSnapshot
from autoslo.workload_definition.query import Query


@dataclass
class VirtualCluster:
    """Mutable per-cluster state used during counterfactual replay."""

    rpu: int
    cost_per_second: float
    active_queries: dict[str, Query] = field(default_factory=dict)
    latencies: dict[str, float] = field(default_factory=dict)
    completion_times: dict[str, float] = field(default_factory=dict)
    billing_window_start_s: float | None = None

    def to_snapshot(self, name: str) -> ClusterSnapshot:
        """Build an immutable :class:`ClusterSnapshot` from current state."""
        return ClusterSnapshot(
            cluster_name=name,
            cost_per_second=self.cost_per_second,
            active_queries=list(self.active_queries.values()),
            billing_window_start_s=self.billing_window_start_s,
        )

    def expire_before(self, time_s: float) -> None:
        """Remove queries whose estimated completion is ≤ *time_s*."""
        expired = [
            qid for qid, ct in self.completion_times.items() if ct <= time_s
        ]
        for qid in expired:
            self.active_queries.pop(qid, None)
            self.latencies.pop(qid, None)
            self.completion_times.pop(qid, None)
        # Close billing window when empty.
        if not self.active_queries:
            self.billing_window_start_s = None

    def add_query(
        self,
        query: Query,
        returned_latencies: dict[str, float],
    ) -> None:
        """Register *query* and refresh latencies for the cluster.

        *returned_latencies* comes from
        :meth:`RoutingPolicy.score_counterfactual` and already respects
        the ``max(current, predicted)`` monotonicity invariant.
        """
        self.active_queries[query.query_id] = query
        if self.billing_window_start_s is None:
            self.billing_window_start_s = query.rel_start_time_s
        # Update latencies and completion times for all affected queries.
        for qid, lat in returned_latencies.items():
            self.latencies[qid] = lat
            q = self.active_queries.get(qid)
            if q is not None:
                self.completion_times[qid] = q.rel_start_time_s + lat
