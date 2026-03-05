"""
routing_policy.py
-----------------
Abstract base for routing policies and simple built-in implementations.

A :class:`RoutingPolicy` encapsulates *how* a query is assigned to a
cluster, without owning any per-query bookkeeping.  Bookkeeping lives in
:class:`~autoslo.routing.managed_cluster_pool.ManagedClusterPool`.
"""

from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from autoslo.routing.routing_core import RoutingResult
from autoslo.utils.class_with_factory import ClassWithFactory
from autoslo.workload_definition.query import Query, QueryTextId

if TYPE_CHECKING:
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool


class RoutingPolicy(ClassWithFactory):
    """Base class for all routing policies.

    Subclasses must implement :meth:`select_cluster`.  Optionally override
    :meth:`build_tracking_query` to create richer :class:`Query` objects
    for the pool (e.g. with featurisations for model-based
    policies).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def select_cluster(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        pool: ManagedClusterPool,
        exclude_clusters: set[str] | None = None,
    ) -> str:
        """Choose the best cluster for the incoming query.

        Parameters
        ----------
        query_id :
            Unique query identifier.
        query_text_id :
            Query-text identifier (used for featurisation lookup).
        start_time_s :
            Arrival time (wall-clock or simulated) in seconds.
        pool :
            Read-only view of per-cluster bookkeeping.

        Returns
        -------
        str
            Name of the cluster the query should be sent to.
        """
        ...

    def build_tracking_query(
        self,
        query_id: str,
        cluster_name: str,
        query_text_id: str,
        start_time_s: float,
        cluster_rpu: int = 0,
    ) -> Query:
        """Create a :class:`Query` suitable for the pool.

        The default implementation returns a minimal object.  Model-based
        policies should override this to attach featurisations and
        stage-model predictions.
        """
        return Query(
            query_id=str(query_id),
            query_text_id=QueryTextId(value=str(query_text_id)),
            rel_start_time_s=start_time_s,
            cluster_name=cluster_name,
            latency_s=-1,
        )

    def route_with_details(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        pool: ManagedClusterPool,
        exclude_clusters: set[str] | None = None,
    ) -> RoutingResult:
        """Route with full results including placement score and tracking query.

        The default implementation delegates to :meth:`select_cluster` and
        returns a :class:`RoutingResult` with *score* set to ``None``.
        Model-based policies should override this to return the full
        :class:`PlacementScore`.
        """
        cluster = self.select_cluster(
            query_id=query_id,
            query_text_id=query_text_id,
            start_time_s=start_time_s,
            pool=pool,
            exclude_clusters=exclude_clusters,
        )
        tracking_query = self.build_tracking_query(
            query_id=query_id,
            cluster_name=cluster,
            query_text_id=query_text_id,
            start_time_s=start_time_s,
            cluster_rpu=pool.get_rpu(cluster),
        )
        return RoutingResult(
            cluster_name=cluster,
            score=None,
            tracking_query=tracking_query,
        )

    def on_attach(self, pool: ManagedClusterPool) -> None:
        """Called once when the policy is attached to a Router.

        Override to perform one-time setup that requires the pool
        (e.g. injecting an RPU-lookup callback into a featuriser).
        """


# -----------------------------------------------------------------------
# Built-in simple policies
# -----------------------------------------------------------------------


class RoundRobinPolicy(RoutingPolicy):
    """Cycle through eligible clusters in order."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cycle: itertools.cycle[str] | None = None

    def select_cluster(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        pool: ManagedClusterPool,
        exclude_clusters: set[str] | None = None,
    ) -> str:
        names = pool.cluster_names
        if exclude_clusters:
            names = [n for n in names if n not in exclude_clusters]
        if not names:
            raise RuntimeError("No clusters available for routing.")
        if self._cycle is None:
            self._cycle = itertools.cycle(names)
        return next(self._cycle)


class FixedPolicy(RoutingPolicy):
    """Always route to a single pre-configured cluster."""

    def __init__(self, cluster_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cluster_name = cluster_name

    def select_cluster(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        pool: ManagedClusterPool,
        exclude_clusters: set[str] | None = None,
    ) -> str:
        return self._cluster_name
