"""
router.py
---------
Concrete query router that delegates placement decisions to a
:class:`~autoslo.routing.routing_policy.RoutingPolicy` and lifecycle
bookkeeping to a
:class:`~autoslo.routing.cluster_state_tracker.ClusterStateTracker`.

This is the single entry-point used by
:class:`~autoslo.workload_execution.query_runner.WorkloadRunner` for all
routing variants — the *behaviour* is determined entirely by the
plugged-in policy.
"""

from __future__ import annotations

import time
from typing import Any

from autoslo.routing.cluster_state_tracker import ClusterStateTracker
from autoslo.routing.routing_core import RoutingResult
from autoslo.routing.routing_policy import RoutingPolicy
from autoslo.workload_definition.query import Query


class Router:
    """Thin coordinator between a routing policy and a state tracker.

    Parameters
    ----------
    policy :
        Determines *which* cluster each query is sent to.
    state_tracker :
        Owns the mutable per-cluster bookkeeping that the policy reads
        (via snapshots) and that the lifecycle hooks update.
    """

    def __init__(
        self,
        policy: RoutingPolicy,
        state_tracker: ClusterStateTracker,
    ) -> None:
        self._policy = policy
        self._tracker = state_tracker
        # Allow the policy to perform one-time setup with the tracker
        # (e.g. inject RPU lookup into a featuriser).
        self._policy.on_attach(self._tracker)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def policy(self) -> RoutingPolicy:
        return self._policy

    @property
    def state_tracker(self) -> ClusterStateTracker:
        return self._tracker

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route_query(
        self,
        query_id: Any,
        query_text_id: Any,
        start_time_s: float | None = None,
        exclude_clusters: set[str] | None = None,
    ) -> str:
        """Choose the best cluster for the incoming query.

        Parameters
        ----------
        query_id :
            Unique query identifier.
        query_text_id :
            Query-text identifier used for featurisation lookup.
        start_time_s :
            Arrival wall-clock (or simulated) time in seconds.
            Defaults to ``time.time()``.
        exclude_clusters :
            Optional set of cluster names to exclude from routing
            (e.g. draining clusters in the simulator).

        Returns
        -------
        str
            Name of the cluster the query should be sent to.
        """
        if start_time_s is None:
            start_time_s = time.time()
        return self._policy.select_cluster(
            query_id=str(query_id),
            query_text_id=str(query_text_id),
            start_time_s=start_time_s,
            state_tracker=self._tracker,
            exclude_clusters=exclude_clusters,
        )

    def route_query_with_predictions(
        self,
        query_id: Any,
        query_text_id: Any,
        start_time_s: float | None = None,
        exclude_clusters: set[str] | None = None,
    ) -> RoutingResult:
        """Route a query and return the full routing result.

        Unlike :meth:`route_query` which returns only a cluster name,
        this method returns a :class:`RoutingResult` containing the
        :class:`PlacementScore` (with per-query latency predictions)
        and a fully-built tracking :class:`Query`.

        Parameters
        ----------
        query_id :
            Unique query identifier.
        query_text_id :
            Query-text identifier used for featurisation lookup.
        start_time_s :
            Arrival wall-clock (or simulated) time in seconds.
            Defaults to ``time.time()``.
        exclude_clusters :
            Optional set of cluster names to exclude from routing.

        Returns
        -------
        RoutingResult
        """
        if start_time_s is None:
            start_time_s = time.time()
        return self._policy.route_with_details(
            query_id=str(query_id),
            query_text_id=str(query_text_id),
            start_time_s=start_time_s,
            state_tracker=self._tracker,
            exclude_clusters=exclude_clusters,
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by WorkloadRunner)
    # ------------------------------------------------------------------

    def on_query_start(
        self,
        query_id: Any,
        cluster_name: str,
        query_text_id: Any,
        start_time_s: float,
    ) -> None:
        """Register a query as actively running on *cluster_name*.

        Builds a tracking :class:`Query` via the policy (so that
        model-based policies can attach featurisations) and hands it
        to the state tracker.
        """
        query = self._policy.build_tracking_query(
            query_id=str(query_id),
            cluster_name=cluster_name,
            query_text_id=str(query_text_id),
            start_time_s=start_time_s,
        )
        self._tracker.on_query_start(query)

    def on_query_finish(
        self,
        query_id: Any,
        cluster_name: str,
        current_time_s: float | None = None,
    ) -> None:
        """Remove a query from the active set for *cluster_name*."""
        if current_time_s is None:
            current_time_s = time.time()
        self._tracker.on_query_finish(
            query_id=str(query_id),
            cluster_name=cluster_name,
            current_time_s=current_time_s,
        )
