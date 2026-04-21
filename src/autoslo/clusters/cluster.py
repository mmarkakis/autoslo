from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Optional

from autoslo.clusters.cluster_conn_info import ClusterConnInfo
from autoslo.utils.billing import Billing, BillingInterval
from autoslo.workload_definition.query import Query

_VALID_CLUSTER_STATE_TRANSITIONS = {
    "pending": {"ready"},
    "ready": {"draining", "collecting_stats", "removed"},
    "draining": {"collecting_stats", "removed"},
    "collecting_stats": {"removed"},
    "removed": set(),
}


class ClusterState(Enum):
    PENDING = "pending"
    READY = "ready"
    DRAINING = "draining"
    COLLECTING_STATS = "collecting_stats"
    REMOVED = "removed"

    @staticmethod
    def is_valid_transition(
        old_state: ClusterState, new_state: ClusterState
    ) -> bool:
        """Check if transitioning from self to new_state is valid."""
        return (old_state == new_state) or (
            new_state.value in _VALID_CLUSTER_STATE_TRANSITIONS[old_state.value]
        )


@dataclass(eq=False)
class Cluster:
    """Mutable cluster with identity, workload state, and neighbor tracking.

    The neighbor map is the primary data structure: it records, for each
    active query, which other queries were concurrently active when that
    query started.  ``active_queries`` is a derived view of its keys.

    Use :meth:`clone` to obtain a deep copy for counterfactual replay or
    safe exposure across thread boundaries.
    """

    # --- Class-level constants -------------------------------------------
    US_EAST_1_COST_PER_RPU_HOUR: ClassVar[float] = 0.375
    ONE_HOUR_S: ClassVar[int] = 3600
    UP_TO_32_RPU_SIZES: ClassVar[list[int]] = [4, 8, 16, 32]
    ALL_ALLOWED_RPU_SIZES: ClassVar[list[int]] = UP_TO_32_RPU_SIZES
    DEFAULT_SPIN_UP_DELAY_S: ClassVar[int] = 300

    _new_counter: ClassVar[itertools.count] = itertools.count()

    # --- Identity (set once) ---------------------------------------------
    creation_time_s: float
    rpu: int
    name: str
    conn_info: Optional[ClusterConnInfo] = field(default=None, repr=False)
    cost_per_rpu_hour: float = field(default=US_EAST_1_COST_PER_RPU_HOUR)

    # --- Mutable state ----------------------------------------
    #
    # Invariant: queries.keys() == _neighbor_ids.keys()
    #
    state: ClusterState = ClusterState.PENDING
    billing_window_start_s: Optional[float] = field(default=None, repr=False)
    past_billing_intervals: list[tuple[float, float]] = field(
        default_factory=list, repr=False
    )

    queries: dict[str, Query] = field(default_factory=dict, repr=False)
    id_to_neighbors: dict[str, list[Query]] = field(
        default_factory=dict, repr=False
    )

    # --- Construction ----------------------------------------------

    def __init__(
        self,
        creation_time_s: float,
        rpu: int,
        name: str | None = None,
        conn_info: Optional[ClusterConnInfo] = None,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
        state: ClusterState = ClusterState.PENDING,
        billing_window_start_s: Optional[float] = None,
        past_billing_intervals: Optional[list[tuple[float, float]]] = None,
        most_recent_query_completion_rel_time_s: Optional[float] = None,
    ) -> None:
        """Create a fresh cluster with no active queries.

        The name must start with "autoslo-{rpu}-".
        """
        if name is None:
            seq = next(Cluster._new_counter)
            name = f"autoslo-{rpu}-{int(datetime.now().timestamp())}-{seq}"
        elif not name.startswith(f"autoslo-{rpu}-"):
            raise ValueError(
                f"Cluster name {name!r} must start with 'autoslo-{rpu}-'."
            )
        self.creation_time_s = creation_time_s
        self.rpu = rpu
        self.name = name
        self.conn_info = conn_info
        self.cost_per_rpu_hour = cost_per_rpu_hour
        self.state = state
        self.billing_window_start_s = billing_window_start_s
        self.past_billing_intervals = list(past_billing_intervals or [])
        self.most_recent_query_completion_rel_time_s: float = (
            most_recent_query_completion_rel_time_s
            if most_recent_query_completion_rel_time_s is not None
            else self.creation_time_s
        )

        self.queries = {}
        self.id_to_neighbors = {}
        self.predicted_latencies: dict[str, float] = {}

    def clone(self) -> Cluster:
        """
        Deep-copy. Relies on `Query` and `ClusterConnInfo` being
        immutable/frozen dataclasses.
        """
        c = Cluster(
            creation_time_s=self.creation_time_s,
            rpu=self.rpu,
            name=self.name,
            conn_info=self.conn_info,
            cost_per_rpu_hour=self.cost_per_rpu_hour,
            state=self.state,
            billing_window_start_s=self.billing_window_start_s,
            past_billing_intervals=self.past_billing_intervals,
            most_recent_query_completion_rel_time_s=(
                self.most_recent_query_completion_rel_time_s
            ),
        )
        c.queries = dict(self.queries)
        c.id_to_neighbors = {
            qid: list(nbs) for qid, nbs in self.id_to_neighbors.items()
        }
        c.predicted_latencies = dict(self.predicted_latencies)
        return c

    @staticmethod
    def _billed_seconds_from_raw_intervals(
        intervals: list[tuple[float, float]],
    ) -> float:
        if len(intervals) == 0:
            return 0.0
        query_intervals = [
            BillingInterval(start_s, end_s)
            for start_s, end_s in intervals
            if end_s > start_s
        ]
        return Billing.billed_s(query_intervals)

    def billing_intervals_until(
        self, rel_time_s: float
    ) -> list[tuple[float, float]]:
        intervals = list(self.past_billing_intervals)
        if (self.billing_window_start_s is not None) and (
            rel_time_s > self.billing_window_start_s
        ):
            intervals.append((self.billing_window_start_s, rel_time_s))
        return intervals

    def billed_seconds_until(self, rel_time_s: float) -> float:
        return Cluster._billed_seconds_from_raw_intervals(
            self.billing_intervals_until(rel_time_s)
        )

    def cost_until(self, rel_time_s: float) -> float:
        return self.cost_per_second * self.billed_seconds_until(rel_time_s)

    # --- Derived properties ----------------------------------------------

    @property
    def cost_per_second(self) -> float:
        """Cost per second for the cluster."""
        return self.cost_per_second_for_rpu(self.rpu, self.cost_per_rpu_hour)

    @property
    def active_queries(self) -> list["Query"]:
        """Currently-active queries (order not guaranteed)."""
        return list(self.queries.values())

    @property
    def active_query_ids(self) -> list[str]:
        return list(self.queries.keys())

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cluster):
            return NotImplemented
        return self.name == other.name

    # --- Workload mutations ----------------------------------------------

    def update_state(self, new_state: ClusterState) -> None:
        """Update the cluster's lifecycle state."""
        if not ClusterState.is_valid_transition(self.state, new_state):
            raise ValueError(
                f"Invalid state transition from {self.state.value} to "
                f"{new_state.value} for cluster {self.name!r}."
            )
        self.state = new_state

    def add_query(
        self,
        query: "Query",
        new_predicted_latencies: dict[str, float],
    ) -> None:
        """
        Register a query as actively running.
        """
        new_query_id = query.query_id

        # Update neighbor maps
        self.id_to_neighbors[new_query_id] = self.active_queries
        for active_query_id in self.active_query_ids:
            self.id_to_neighbors[active_query_id].append(query)

        # Mark the query as active and set billing window start.
        self.queries[new_query_id] = query
        if self.billing_window_start_s is None:
            self.billing_window_start_s = query.rel_start_time_s

        self.predicted_latencies = dict(new_predicted_latencies)

    def finish_query(
        self,
        query_id: str,
        rel_time_s: float,
    ) -> tuple[Query, float]:
        """
        Remove a query from active tracking.

        Parameters:
        -----------
        query_id: The ID of the query to finish.
        rel_time_s: Relative time in seconds since run start.

        Returns:
        --------
        query: The finished Query object.
        latency_s: The latency of the query in seconds.
        """

        if query_id not in self.active_query_ids:
            raise ValueError(
                f"Cannot finish query {query_id}: not found on cluster "
                f"{self.name}."
            )

        q = self.queries.pop(query_id)
        self.id_to_neighbors.pop(query_id, None)
        self.predicted_latencies.pop(query_id, None)

        # Close the current active-service window when the cluster becomes
        # idle. Billing threshold/granularity is applied by Billing.billed_s.
        if (len(self.queries) == 0) and (
            self.billing_window_start_s is not None
        ):
            self.past_billing_intervals.append(
                (self.billing_window_start_s, rel_time_s)
            )
            self.billing_window_start_s = None

        self.most_recent_query_completion_rel_time_s = rel_time_s
        return q, rel_time_s - q.rel_start_time_s

    def finish_queries_until(
        self,
        rel_time_s: float,
    ) -> list[tuple[Query, float]]:
        """
        Finish all queries that have completed by the given time.

        Parameters:
        -----------
        rel_time_s: Relative time in seconds since run start.
        """
        if not set(self.active_query_ids).issubset(
            self.predicted_latencies.keys()
        ):
            breakpoint()
        times_and_ids_of_finished_queries = []
        for qid, q in self.queries.items():
            predicted_completion_rel_time_s = (
                q.rel_start_time_s + self.predicted_latencies[qid]
            )
            if predicted_completion_rel_time_s <= rel_time_s:
                times_and_ids_of_finished_queries.append(
                    (predicted_completion_rel_time_s, qid)
                )
        times_and_ids_of_finished_queries.sort()

        qs_and_latencies = []
        for (
            predicted_completion_rel_time_s,
            qid,
        ) in times_and_ids_of_finished_queries:
            qs_and_latencies.append(
                self.finish_query(qid, predicted_completion_rel_time_s)
            )
        return qs_and_latencies

    # --- Static helpers --------------------------------------------------

    @staticmethod
    def all_allowed_rpu_sizes() -> list[int]:
        return Cluster.ALL_ALLOWED_RPU_SIZES

    @staticmethod
    def rpu_for_cluster_name(cluster_name: str) -> int:
        """Parse RPU from a cluster name.

        Supports the naming convention
        ``"autoslo-{rpu}-{timestamp}-{counter}"``.
        """
        parts = cluster_name.split("-")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        raise ValueError(
            f"Cannot parse RPU from cluster name: {cluster_name!r}"
        )

    @staticmethod
    def cost_per_second_for_rpu(
        rpu: int,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
    ) -> float:
        """Return the cost-per-second for the given RPU size."""
        return cost_per_rpu_hour * rpu / Cluster.ONE_HOUR_S


class ClusterView:
    """Immutable, deep-copied view of a Cluster for safe read-only use."""

    __slots__ = (
        "creation_time_s",
        "rpu",
        "name",
        "conn_info",
        "cost_per_rpu_hour",
        "state",
        "billing_window_start_s",
        "past_billing_intervals",
        "most_recent_query_completion_rel_time_s",
        "queries",
        "id_to_neighbors",
        "predicted_latencies",
    )

    def __init__(self, cluster: "Cluster"):
        self.creation_time_s = cluster.creation_time_s
        self.rpu = cluster.rpu
        self.name = cluster.name
        self.conn_info = cluster.conn_info
        self.cost_per_rpu_hour = cluster.cost_per_rpu_hour
        self.state = cluster.state
        self.billing_window_start_s = cluster.billing_window_start_s
        self.past_billing_intervals = list(cluster.past_billing_intervals)
        self.most_recent_query_completion_rel_time_s = (
            cluster.most_recent_query_completion_rel_time_s
        )
        # Deep copy all mutable state
        self.queries = dict(cluster.queries)
        self.id_to_neighbors = {
            qid: list(nbs) for qid, nbs in cluster.id_to_neighbors.items()
        }
        self.predicted_latencies = dict(cluster.predicted_latencies)

    # --- Read-only properties ---
    @property
    def active_queries(self) -> list["Query"]:
        return list(self.queries.values())

    @property
    def active_query_ids(self) -> list[str]:
        return list(self.queries.keys())

    @property
    def cost_per_second(self) -> float:
        return Cluster.cost_per_second_for_rpu(self.rpu, self.cost_per_rpu_hour)

    def billing_intervals_until(
        self, rel_time_s: float
    ) -> list[tuple[float, float]]:
        intervals = list(self.past_billing_intervals)
        if (self.billing_window_start_s is not None) and (
            rel_time_s > self.billing_window_start_s
        ):
            intervals.append((self.billing_window_start_s, rel_time_s))
        return intervals

    def billed_seconds_until(self, rel_time_s: float) -> float:
        return Cluster._billed_seconds_from_raw_intervals(
            self.billing_intervals_until(rel_time_s)
        )

    def cost_until(self, rel_time_s: float) -> float:
        return self.cost_per_second * self.billed_seconds_until(rel_time_s)

    def cost_with_query_start_until(
        self, query_start_s: float, rel_time_s: float
    ) -> float:
        """Cost until *rel_time_s* if a query starts at *query_start_s*."""
        effective_window_start = (
            self.billing_window_start_s
            if self.billing_window_start_s is not None
            else query_start_s
        )
        intervals = list(self.past_billing_intervals)
        if rel_time_s > effective_window_start:
            intervals.append((effective_window_start, rel_time_s))
        return (
            self.cost_per_second
            * Cluster._billed_seconds_from_raw_intervals(intervals)
        )

    def hypothetical_neighbors_with(
        self, query: "Query"
    ) -> dict["Query", list["Query"]]:
        """
        Return the neighbor map that would result if *query* were added as active.
        Mirrors the semantics of Cluster.add_query() without mutating self.
        """
        new_query_id = query.query_id
        # Snapshot of existing state before the new query is added.
        # id_to_neighbors[new_query_id] = queries currently active (not including itself).
        existing_active = list(self.queries.values())
        queries = dict(self.queries)
        queries[new_query_id] = query
        id_to_neighbors: dict[str, list[Query]] = {
            qid: list(nbs) for qid, nbs in self.id_to_neighbors.items()
        }
        id_to_neighbors[new_query_id] = existing_active
        # Mirror add_query: append the new query to every existing active query's neighbor list.
        for active_query_id in self.queries:
            id_to_neighbors[active_query_id] = list(
                id_to_neighbors[active_query_id]
            ) + [query]
        return {queries[qid]: list(nbs) for qid, nbs in id_to_neighbors.items()}

    def to_cluster(self) -> "Cluster":
        """Reconstruct a mutable Cluster from this view's deep-copied data."""
        c = Cluster(
            creation_time_s=self.creation_time_s,
            rpu=self.rpu,
            name=self.name,
            conn_info=self.conn_info,
            cost_per_rpu_hour=self.cost_per_rpu_hour,
            state=self.state,
            billing_window_start_s=self.billing_window_start_s,
            past_billing_intervals=self.past_billing_intervals,
            most_recent_query_completion_rel_time_s=(
                self.most_recent_query_completion_rel_time_s
            ),
        )
        c.queries = dict(self.queries)
        c.id_to_neighbors = {
            qid: list(nbs) for qid, nbs in self.id_to_neighbors.items()
        }
        c.predicted_latencies = dict(self.predicted_latencies)
        return c

    # --- Block all mutation ---
    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError(f"ClusterView is immutable: cannot set {name}")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"ClusterView is immutable: cannot delete {name}")
