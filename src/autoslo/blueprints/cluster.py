from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import ClassVar, Optional

from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from autoslo.utils.billing import Billing
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
    rpu: int
    name: str
    conn_info: Optional[ClusterConnInfo] = field(default=None, repr=False)
    cost_per_rpu_hour: float = field(default=US_EAST_1_COST_PER_RPU_HOUR)

    # --- Mutable state ----------------------------------------
    #
    # Invariant: _queries.keys() == _neighbor_map.keys()
    #
    state: ClusterState = ClusterState.PENDING
    predicted_ready_time_s: Optional[float] = None
    billing_window_start_s: Optional[float] = field(default=None, repr=False)

    _queries: dict[str, "Query"] = field(default_factory=dict, repr=False)
    neighbor_map: dict[str, list["Query"]] = field(
        default_factory=dict, repr=False
    )

    # --- Construction ----------------------------------------------

    def __init__(
        self,
        rpu: int,
        name: str | None = None,
        conn_info: Optional[ClusterConnInfo] = None,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
        state: ClusterState = ClusterState.PENDING,
        predicted_ready_time_s: Optional[float] = None,
        billing_window_start_s: Optional[float] = None,
    ) -> None:
        """Create a fresh cluster with no active queries.

        The name must start with "cluster_{rpu}_".
        """
        if name is None:
            seq = next(Cluster._new_counter)
            name = f"cluster_{rpu}_{int(datetime.now().timestamp())}_{seq}"
        elif not name.startswith(f"cluster_{rpu}_"):
            raise ValueError(
                f"Cluster name {name!r} must start with 'cluster_{rpu}_'."
            )
        self.rpu = rpu
        self.name = name
        self.conn_info = conn_info
        self.cost_per_rpu_hour = cost_per_rpu_hour
        self.state = state
        self.predicted_ready_time_s = predicted_ready_time_s
        self.billing_window_start_s = billing_window_start_s

        self._queries = {}
        self.neighbor_map = {}

    def clone(self) -> Cluster:
        """
        Deep-copy. Relies on `Query` and `ClusterConnInfo` being
        immutable/frozen dataclasses.
        """
        c = Cluster(
            rpu=self.rpu,
            name=self.name,
            conn_info=self.conn_info,
            cost_per_rpu_hour=self.cost_per_rpu_hour,
            state=self.state,
            predicted_ready_time_s=self.predicted_ready_time_s,
            billing_window_start_s=self.billing_window_start_s,
        )
        c._queries = dict(self._queries)
        c.neighbor_map = {
            qid: list(nbs) for qid, nbs in self.neighbor_map.items()
        }
        return c

    # --- Derived properties ----------------------------------------------

    @property
    def cost_per_second(self) -> float:
        """Cost per second for the cluster."""
        return self.cost_per_second_for_rpu(self.rpu, self.cost_per_rpu_hour)

    @property
    def active_queries(self) -> list["Query"]:
        """Currently-active queries (order not guaranteed)."""
        return list(self._queries.values())

    @property
    def active_query_ids(self) -> set[str]:
        return set(self._queries.keys())

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
    ) -> None:
        """
        Register a query as actively running.
        """
        new_query_id = query.query_id

        # Update neighbor maps
        self.neighbor_map[new_query_id] = self.active_queries
        for active_query_id in self.active_query_ids:
            self.neighbor_map[active_query_id].append(query)

        # Mark the query as active and set billing window start.
        self._queries[new_query_id] = query
        if self.billing_window_start_s is None:
            self.billing_window_start_s = query.rel_start_time_s

    def finish_query(
        self,
        query_id: str,
        current_time_s: float,
        min_billing_window_size_s: float = Billing.REDSHIFT_BILLING_THRESHOLD_S,
    ) -> "Query":
        """
        Remove a query from active tracking.

        Parameters:
        -----------
        query_id: The ID of the query to finish.
        current_time_s: The current time in seconds
        min_billing_window_size_s: The minimum size of a billing window. If the
            time since the start of the current billing window exceeds this
            threshold, the billing window is closed.

        Returns:
        --------
        query: The finished Query object.
        billing_interval: If a billing interval was completed, its endpoints.
        """

        if query_id not in self.active_query_ids:
            raise ValueError(
                f"Cannot finish query {query_id}: not found on cluster "
                f"{self.name}."
            )

        q = self._queries.pop(query_id)
        self.neighbor_map.pop(query_id, None)

        if (self.billing_window_start_s is not None) and (
            (current_time_s - self.billing_window_start_s)
            >= min_billing_window_size_s
        ):
            self.billing_window_start_s = None
        return q

    # --- Static helpers --------------------------------------------------

    @staticmethod
    def all_allowed_rpu_sizes() -> list[int]:
        return Cluster.ALL_ALLOWED_RPU_SIZES

    @staticmethod
    def rpu_for_cluster_name(cluster_name: str) -> int:
        """Parse RPU from a cluster name.

        Supports the dynamic naming convention
        ``"cluster_{rpu}_{timestamp}_{counter}"`` as well as static
        config names of the form ``"cluster_{rpu}_..."``.
        """
        parts = cluster_name.split("_")
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
