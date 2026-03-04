"""
cluster_state_tracker.py
------------------------
Thread-safe bookkeeping for per-cluster query state.

A :class:`ClusterStateTracker` owns the mutable per-cluster data that
routing decisions and lifecycle hooks need: active queries, neighbour
history, billing windows, and per-cluster metadata (cost, RPU).

The tracker is **model-agnostic** — it stores whatever :class:`Query`
objects it receives and exposes atomic snapshot helpers so that a
:class:`~autoslo.routing.routing_policy.RoutingPolicy` can evaluate
placements without holding the lock.
"""

from __future__ import annotations

import threading
from typing import Optional

from autoslo.blueprints.cluster import Cluster
from autoslo.blueprints.cluster_pool import ClusterPool
from autoslo.routing.routing_core import ClusterSnapshot
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query


class ClusterStateTracker:
    """Mutable, thread-safe store of per-cluster query bookkeeping.

    Parameters
    ----------
    cluster_names :
        Initial set of cluster names.
    cost_per_second :
        ``{cluster_name: cost_per_second}`` for billing.
    rpu_per_cluster :
        ``{cluster_name: rpu}`` for featuriser lookups.
    """

    def __init__(
        self,
        cluster_names: list[str],
        cost_per_second: dict[str, float],
        rpu_per_cluster: dict[str, int],
    ) -> None:
        self._lock = threading.Lock()
        self._cluster_names: list[str] = list(cluster_names)
        self._cost_per_second: dict[str, float] = dict(cost_per_second)
        self._rpu_per_cluster: dict[str, int] = dict(rpu_per_cluster)

        # Per-cluster active queries: cluster_name → {query_id → Query}
        self._active_queries: dict[str, dict[str, Query]] = {
            cn: {} for cn in cluster_names
        }
        # Billing window tracking: cluster → start time (None until first query).
        self._billing_window_start_s: dict[str, Optional[float]] = {
            cn: None for cn in cluster_names
        }
        # Per-query neighbour history: query_id → [Query, …]
        self._neighbors_per_active_query: dict[str, list[Query]] = {}
        # Per-cluster recently-touched tables (stub for cache-affinity).
        self._recent_tables: dict[str, set[str]] = {
            cn: set() for cn in cluster_names
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_cluster_pool(cls, cluster_pool: ClusterPool) -> ClusterStateTracker:
        """Create a tracker from an existing :class:`ClusterPool`."""
        names = cluster_pool.cluster_names
        cost = {c.name: c.cost_per_second for c in cluster_pool.clusters}
        rpus = {c.name: c.rpu for c in cluster_pool.clusters}
        return cls(
            cluster_names=names,
            cost_per_second=cost,
            rpu_per_cluster=rpus,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cluster_names(self) -> list[str]:
        """Current cluster names (copy, safe to iterate outside lock)."""
        with self._lock:
            return list(self._cluster_names)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_query_start(self, query: Query) -> None:
        """Register *query* as actively running on ``query.cluster_name``.

        Must be called *before* the query begins executing so that
        concurrent routing decisions see it.
        """
        with self._lock:
            cn = query.cluster_name
            qid = query.query_id

            # Initialise this query's neighbour list with all currently
            # active co-runners, and append it to each of their lists.
            current_actives = list(self._active_queries[cn].values())
            self._neighbors_per_active_query[qid] = list(current_actives)
            for active_q in current_actives:
                self._neighbors_per_active_query[active_q.query_id].append(
                    query
                )

            self._active_queries[cn][qid] = query

            # If this is the first query on this cluster, start a billing
            # window.
            if self._billing_window_start_s[cn] is None:
                self._billing_window_start_s[cn] = query.rel_start_time_s

    def on_query_finish(
        self,
        query_id: str,
        cluster_name: str,
        current_time_s: float,
    ) -> None:
        """Remove *query_id* from the active set for *cluster_name*."""
        query_id = str(query_id)
        with self._lock:
            if query_id not in self._active_queries[cluster_name]:
                raise KeyError(
                    f"Query {query_id!r} not found in active queries "
                    f"for cluster {cluster_name}."
                )
            del self._active_queries[cluster_name][query_id]
            # Remove the neighbour-history entry (the Query object itself
            # persists inside other queries' neighbour lists, which is
            # the desired behaviour).
            self._neighbors_per_active_query.pop(query_id, None)

            # If the cluster has no more running queries, close the billing
            # window only if it has also lasted at least as long as the
            # billing threshold.
            billing_start = self._billing_window_start_s[cluster_name]
            if (
                (not self._active_queries[cluster_name])
                and (billing_start is not None)
                and (
                    billing_start + Billing.REDSHIFT_BILLING_THRESHOLD_S
                    <= current_time_s
                )
            ):
                self._billing_window_start_s[cluster_name] = None

    # ------------------------------------------------------------------
    # Snapshot builders (for routing decisions)
    # ------------------------------------------------------------------

    def build_routing_context(
        self, incoming: Query
    ) -> tuple[
        dict[str, ClusterSnapshot],
        dict[str, dict[Query, list[Query]]],
    ]:
        """Build snapshots **and** neighbour maps atomically.

        Parameters
        ----------
        incoming :
            The yet-to-be-placed query.  It is added to every cluster's
            neighbour map so the policy can evaluate hypothetical
            placements.

        Returns
        -------
        snapshots :
            ``{cluster_name: ClusterSnapshot}``
        neighbor_map :
            ``{cluster_name: {base_query: [neighbour, …]}}``
            including *incoming* in each cluster's context.
        """
        with self._lock:
            snapshots: dict[str, ClusterSnapshot] = {}
            neighbor_map: dict[str, dict[Query, list[Query]]] = {}

            for cn in self._cluster_names:
                active_list = list(self._active_queries[cn].values())

                snapshots[cn] = ClusterSnapshot(
                    cluster_name=cn,
                    cost_per_second=self._cost_per_second[cn],
                    active_queries=active_list,
                    billing_window_start_s=self._billing_window_start_s[cn],
                )

                # For each active query, use its accumulated co-runner
                # history (may include finished queries) plus the incoming
                # query.  The incoming query itself sees only the currently
                # active queries + itself.
                neighbor_map[cn] = {
                    q: self._neighbors_per_active_query[q.query_id]
                    + [incoming]
                    for q in active_list
                }
                neighbor_map[cn][incoming] = active_list + [incoming]

            return snapshots, neighbor_map

    def build_snapshots(self) -> dict[str, ClusterSnapshot]:
        """Return snapshots without a neighbour map (for introspection)."""
        with self._lock:
            return {
                cn: ClusterSnapshot(
                    cluster_name=cn,
                    cost_per_second=self._cost_per_second[cn],
                    active_queries=list(self._active_queries[cn].values()),
                    billing_window_start_s=self._billing_window_start_s[cn],
                )
                for cn in self._cluster_names
            }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_active_queries(self, cluster_name: str) -> list[Query]:
        """Return a copy of the active-query list for *cluster_name*."""
        with self._lock:
            return list(self._active_queries[cluster_name].values())

    def get_all_active_queries(self) -> dict[str, list[Query]]:
        """Return a snapshot of active queries across all clusters."""
        with self._lock:
            return {
                cn: list(qs.values())
                for cn, qs in self._active_queries.items()
            }

    def get_rpu(self, cluster_name: str) -> int:
        """Return the RPU for *cluster_name*."""
        with self._lock:
            return self._rpu_per_cluster[cluster_name]

    # ------------------------------------------------------------------
    # Dynamic cluster management
    # ------------------------------------------------------------------

    def add_cluster(self, cluster: Cluster) -> None:
        """Register a dynamically provisioned cluster.

        Initialises all per-cluster bookkeeping so that the new cluster
        becomes eligible for routing immediately.

        Raises
        ------
        ValueError
            If a cluster with the same name is already registered.
        """
        cn = cluster.name
        with self._lock:
            if cn in self._active_queries:
                raise ValueError(f"Cluster {cn!r} is already registered.")
            self._cluster_names.append(cn)
            self._active_queries[cn] = {}
            self._billing_window_start_s[cn] = None
            self._cost_per_second[cn] = cluster.cost_per_second
            self._recent_tables[cn] = set()
            self._rpu_per_cluster[cn] = cluster.rpu

    def remove_cluster(self, cluster_name: str) -> None:
        """Unregister a cluster, making it ineligible for routing.

        Raises
        ------
        KeyError
            If the cluster name is not registered.
        ValueError
            If there are still active queries on the cluster.
        """
        with self._lock:
            if cluster_name not in self._active_queries:
                raise KeyError(f"Cluster {cluster_name!r} is not registered.")
            if self._active_queries[cluster_name]:
                raise ValueError(
                    f"Cluster {cluster_name!r} has active queries."
                )
            self._cluster_names.remove(cluster_name)
            for qid in list(self._active_queries[cluster_name].keys()):
                self._neighbors_per_active_query.pop(qid, None)
            del self._active_queries[cluster_name]
            del self._billing_window_start_s[cluster_name]
            del self._cost_per_second[cluster_name]
            del self._recent_tables[cluster_name]
            del self._rpu_per_cluster[cluster_name]
