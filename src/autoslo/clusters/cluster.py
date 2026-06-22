from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

import numpy as np

from autoslo.clusters.billing import BillingAccumulator, BillingInterval
from autoslo.clusters.cluster_conn_info import ClusterConnInfo
from autoslo.nn.lstm_state import AfterLSTMState
from autoslo.workload_definition.query import Query

# Cluster names must match "autoslo-{rpu}-{suffix}" where rpu is a positive integer.
_CLUSTER_NAME_RE = re.compile(r"^autoslo-(\d+)-.+$")

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


def cluster_cost_until_drained(
    queries: list[Query],
    predicted_latencies: dict[str, float],
    billing_accumulator: BillingAccumulator,
    billing_window_start_s: Optional[float],
    cost_per_second: float,
    current_rel_time_s: float,
) -> float:
    """
    Cost billed from the start of this query's lifetime until all
    currently-active queries, as well as any additional queries, have
    drained.
    """
    end_s = current_rel_time_s
    if queries:
        end_s = max(
            q.rel_start_time_s + predicted_latencies[q.query_id]
            for q in queries
        )
        end_s = max(end_s, current_rel_time_s)

    if (billing_window_start_s is not None) and (
        current_rel_time_s > billing_window_start_s
    ):
        billed_seconds = billing_accumulator.billed_s_with_window(
            billing_window_start_s, end_s
        )
    else:
        billed_seconds = billing_accumulator.billed_s()
    return cost_per_second * billed_seconds


@dataclass(eq=False)
class Cluster:
    """Mutable cluster with identity, workload state, and neighbor tracking.

    The neighbor map is the primary data structure: it records, for each
    active query, which other queries were concurrently active when that
    query started.  ``active_queries`` is a derived view of its keys.

    Use :meth:`clone` to obtain a deep copy for counterfactual replay or
    safe exposure across thread boundaries.

    We assume that cluster names are GLOBALLY UNIQUE, across all runs of any
    workload.
    """

    # --- Class-level constants -------------------------------------------
    US_EAST_1_COST_PER_RPU_HOUR: ClassVar[float] = 0.375
    ONE_HOUR_S: ClassVar[int] = 3600
    UP_TO_32_RPU_SIZES: ClassVar[list[int]] = [4, 8, 16, 32]
    ALL_ALLOWED_RPU_SIZES: ClassVar[list[int]] = UP_TO_32_RPU_SIZES
    DEFAULT_SPIN_UP_DELAY_S: ClassVar[int] = 300

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
    billing_accumulator: BillingAccumulator = field(
        default_factory=BillingAccumulator, repr=False
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
        cache_state: np.ndarray,
        name: str,
        conn_info: Optional[ClusterConnInfo] = None,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
        state: ClusterState = ClusterState.PENDING,
        billing_window_start_s: Optional[float] = None,
        past_billing_intervals: Optional[list[BillingInterval]] = None,
        most_recent_query_completion_rel_time_s: Optional[float] = None,
    ) -> None:
        """Create a fresh cluster with no active queries.

        The name must match the pattern ``autoslo-{rpu}-<suffix>`` where
        ``{rpu}`` matches the *rpu* argument and ``<suffix>`` is any
        non-empty string.
        """
        m = _CLUSTER_NAME_RE.fullmatch(name)
        if m is None or int(m.group(1)) != rpu:
            raise ValueError(
                f"Cluster name {name!r} must match 'autoslo-{rpu}-<suffix>'."
            )
        self.creation_time_s = creation_time_s
        self.rpu = rpu
        self.name = name
        self.conn_info = conn_info
        self.cost_per_rpu_hour = cost_per_rpu_hour
        self.state = state
        self.billing_window_start_s = billing_window_start_s
        self.billing_accumulator = BillingAccumulator()
        for iv in past_billing_intervals or []:
            self.billing_accumulator.add_interval(iv.start, iv.end)
        self.most_recent_query_completion_rel_time_s: float = (
            most_recent_query_completion_rel_time_s
            if most_recent_query_completion_rel_time_s is not None
            else self.creation_time_s
        )
        self.cache_state = cache_state

        self.queries = {}
        self.id_to_neighbors = {}
        self.predicted_latencies: dict[str, float] = {}
        self.lstm_states: dict[str, AfterLSTMState] = {}

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
        new_cache_state: np.ndarray,
        new_lstm_states_on_selected: dict[str, AfterLSTMState],
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
        self.cache_state = new_cache_state.copy()
        self.lstm_states = dict(new_lstm_states_on_selected)

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
        self.lstm_states.pop(query_id, None)

        # Close the current active-service window when the cluster becomes
        # idle. Billing threshold/granularity is applied by Billing.billed_s.
        if (len(self.queries) == 0) and (
            self.billing_window_start_s is not None
        ):
            self.billing_accumulator.add_interval(
                self.billing_window_start_s, rel_time_s
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
            raise ValueError(
                f"Predicted latencies missing for some active queries on "
                f"cluster {self.name}."
            )
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
        ``"autoslo-{rpu}-{run_id}-{counter}"``.
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
    def run_id_for_cluster_name(cluster_name: str) -> str:
        """Parse run ID from a cluster name.

        Supports the naming convention
        ``"autoslo-{rpu}-{run_id}-{counter}"``.
        """
        parts = cluster_name.split("-")
        if len(parts) >= 3:
            return parts[2]
        raise ValueError(
            f"Cannot parse run ID from cluster name: {cluster_name!r}"
        )

    @staticmethod
    def counter_for_cluster_name(cluster_name: str) -> Optional[int]:
        """Parse counter from a cluster name.

        Supports the naming convention
        ``"autoslo-{rpu}-{run_id}-{counter}"``.
        """
        parts = cluster_name.split("-")
        if len(parts) >= 4:
            try:
                return int(parts[3])
            except ValueError:
                pass
        return None

    @staticmethod
    def cost_per_second_for_rpu(
        rpu: int,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
    ) -> float:
        """Return the cost-per-second for the given RPU size."""
        return cost_per_rpu_hour * rpu / Cluster.ONE_HOUR_S


@dataclass(frozen=True, slots=True)
class ClusterView:
    """Immutable, deep-copied view of a Cluster for safe read-only use."""

    creation_time_s: float
    rpu: int
    name: str
    conn_info: Optional[ClusterConnInfo] = field(default=None, repr=False)
    cost_per_rpu_hour: float = field(
        default=Cluster.US_EAST_1_COST_PER_RPU_HOUR
    )
    state: ClusterState = field(default=ClusterState.PENDING)
    billing_window_start_s: Optional[float] = field(default=None, repr=False)
    billing_accumulator: BillingAccumulator = field(
        default_factory=BillingAccumulator, repr=False
    )
    most_recent_query_completion_rel_time_s: Optional[float] = field(
        default=None
    )
    queries: dict[str, "Query"] = field(default_factory=dict, repr=False)
    id_to_neighbors: dict[str, list["Query"]] = field(
        default_factory=dict, repr=False
    )
    predicted_latencies: dict[str, float] = field(
        default_factory=dict, repr=False
    )
    cache_state: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False
    )
    lstm_states: dict[str, AfterLSTMState] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def from_cluster(cls, cluster: Cluster) -> "ClusterView":
        return cls(
            creation_time_s=cluster.creation_time_s,
            rpu=cluster.rpu,
            name=cluster.name,
            conn_info=cluster.conn_info,
            cost_per_rpu_hour=cluster.cost_per_rpu_hour,
            state=cluster.state,
            billing_window_start_s=cluster.billing_window_start_s,
            billing_accumulator=cluster.billing_accumulator.copy(),
            most_recent_query_completion_rel_time_s=(
                cluster.most_recent_query_completion_rel_time_s
            ),
            queries=dict(cluster.queries),
            id_to_neighbors={
                qid: list(nbs) for qid, nbs in cluster.id_to_neighbors.items()
            },
            predicted_latencies=dict(cluster.predicted_latencies),
            cache_state=(
                cluster.cache_state.copy()
                if cluster.cache_state is not None
                else None
            ),
            lstm_states=dict(cluster.lstm_states),
        )

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
            most_recent_query_completion_rel_time_s=(
                self.most_recent_query_completion_rel_time_s
            ),
            cache_state=(
                self.cache_state.copy()
                if self.cache_state is not None
                else None
            ),
        )
        c.billing_accumulator = self.billing_accumulator.copy()
        c.queries = dict(self.queries)
        c.id_to_neighbors = {
            qid: list(nbs) for qid, nbs in self.id_to_neighbors.items()
        }
        c.predicted_latencies = dict(self.predicted_latencies)
        c.lstm_states = dict(self.lstm_states)
        return c
