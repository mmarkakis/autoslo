"""
autoscaling_policy.py
---------------------
Abstract base for autoscaling policies and shared data types.

An :class:`AutoscalingPolicy` encapsulates *how* capacity decisions are
made (spin-up / tear-down), without owning execution.  The
:class:`~autoslo.capacity.autoscaler.Autoscaler` coordinator dispatches
events to the policy and executes returned actions.

This mirrors the :class:`~autoslo.routing.routing_policy.RoutingPolicy`
pattern: the policy is passive, stateless with respect to execution, and
pluggable.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from autoslo.routing.routing_core import RoutingResult
from autoslo.utils.class_with_factory import ClassWithFactory

if TYPE_CHECKING:
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SpinUpRequest:
    """Request to spin up a new cluster with a given RPU size."""

    rpu: int
    reason: str


@dataclass
class TearDownRequest:
    """Request to tear down an existing cluster."""

    cluster_name: str
    reason: str


@dataclass
class AutoscalingAction:
    """Returned by policy event handlers to request capacity changes.

    An empty action (no spin-ups, no tear-downs) means "do nothing".
    Multiple spin-ups and/or tear-downs in one action are allowed.
    """

    spin_ups: list[SpinUpRequest] = field(default_factory=list)
    tear_downs: list[TearDownRequest] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class AutoscalingPolicy(ClassWithFactory):
    """Abstract base class for autoscaling policies.

    Policies are passive — they receive events, maintain internal state,
    and return :class:`AutoscalingAction` objects.  They never execute
    actions directly (no callbacks, no threading).

    Lifecycle
    ---------
    1. Policy is constructed with its own parameters.
    2. :meth:`on_attach` is called by the :class:`Autoscaler` at
       construction, providing the live
       :class:`~autoslo.routing.managed_cluster_pool.ManagedClusterPool`.
    3. Event methods are called by the :class:`Autoscaler` for each
       routing result, query completion, or time advance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Event handlers (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def on_routing_result(
        self,
        result: RoutingResult,
        current_time_s: float,
        current_latencies: dict[str, float] | None = None,
    ) -> AutoscalingAction:
        """Called after each routing decision.  May return any action."""
        ...

    @abstractmethod
    def on_query_complete(
        self,
        query_id: str,
        cluster_name: str,
        current_time_s: float,
    ) -> AutoscalingAction:
        """Called when a query finishes.  May return any action."""
        ...

    @abstractmethod
    def on_time_advance(
        self,
        current_time_s: float,
    ) -> AutoscalingAction:
        """Called periodically.  May return any action."""
        ...

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional overrides)
    # ------------------------------------------------------------------

    def on_attach(self, pool: ManagedClusterPool) -> None:
        """Called once when the policy is attached to an Autoscaler.

        Gives the policy a reference to the live pool, from which it can
        call ``get_all_active_queries()``, ``cluster_names``,
        ``get_rpu()``, etc.  Matches the
        :meth:`~autoslo.routing.routing_policy.RoutingPolicy.on_attach`
        pattern.

        Default implementation stores the pool as ``self._pool``.
        """
        self._pool = pool

    def on_cluster_ready(
        self,
        cluster_name: str,
        rpu: int,
        ready_time_s: float,
    ) -> None:
        """Called when a pending cluster becomes READY.

        Default implementation is a no-op.  Policies that track pending
        capacity or minimum cluster lifetimes should override this.
        """


# ---------------------------------------------------------------------------
# NoOpPolicy
# ---------------------------------------------------------------------------


class NoOpPolicy(AutoscalingPolicy):
    """Autoscaling policy that does nothing.

    Useful for:
    - Testing routing policies in isolation.
    - Running with a fixed cluster set (current ``WorkloadRunner`` default).
    """

    @property
    def name(self) -> str:
        return "NoOpPolicy"

    def on_routing_result(
        self,
        result: RoutingResult,
        current_time_s: float,
        current_latencies: dict[str, float] | None = None,
    ) -> AutoscalingAction:
        return AutoscalingAction()

    def on_query_complete(
        self,
        query_id: str,
        cluster_name: str,
        current_time_s: float,
    ) -> AutoscalingAction:
        return AutoscalingAction()

    def on_time_advance(
        self,
        current_time_s: float,
    ) -> AutoscalingAction:
        return AutoscalingAction()
