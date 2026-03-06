"""
autoscaler.py
-------------
Thin coordinator between an :class:`AutoscalingPolicy` and a
:class:`~autoslo.routing.managed_cluster_pool.ManagedClusterPool`.

Analogous to :class:`~autoslo.routing.router.Router` for routing:
the ``Autoscaler`` receives events from the orchestrator, delegates
to the policy, and executes the returned
:class:`~autoslo.capacity.autoscaling_policy.AutoscalingAction` via
callbacks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from autoslo.capacity.autoscaling_policy import (
    AutoscalingAction,
    AutoscalingPolicy,
)
from autoslo.routing.routing_core import RoutingResult

if TYPE_CHECKING:
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool

logger = logging.getLogger(__name__)


class Autoscaler:
    """Coordinator that dispatches events to an autoscaling policy and
    executes the returned actions via callbacks.

    Parameters
    ----------
    policy :
        Determines *when* and *how* to spin up / tear down clusters.
    pool :
        Live cluster pool (passed to the policy via ``on_attach``).
    on_spin_up :
        Callback ``(reason: str, rpu: int) -> None`` invoked for each
        spin-up request in a returned action.
    on_tear_down :
        Callback ``(cluster_name: str) -> None`` invoked for each
        tear-down request in a returned action.
    """

    def __init__(
        self,
        policy: AutoscalingPolicy,
        pool: "ManagedClusterPool",
        on_spin_up: Callable[[str, int], None],
        on_tear_down: Callable[[str], None],
    ) -> None:
        self._policy = policy
        self._pool = pool
        self._on_spin_up = on_spin_up
        self._on_tear_down = on_tear_down
        # Attach policy to pool (like Router calls policy.on_attach).
        self._policy.on_attach(pool)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def policy(self) -> AutoscalingPolicy:
        return self._policy

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def on_routing_result(
        self, result: RoutingResult, current_time_s: float
    ) -> None:
        """Forward a routing result to the policy and execute actions."""
        action = self._policy.on_routing_result(result, current_time_s)
        self._execute(action)

    def on_query_complete(
        self, query_id: str, cluster_name: str, current_time_s: float
    ) -> None:
        """Forward a query-completion event and execute actions."""
        action = self._policy.on_query_complete(
            query_id, cluster_name, current_time_s
        )
        self._execute(action)

    def on_time_advance(self, current_time_s: float) -> None:
        """Forward a time-advance tick and execute actions."""
        action = self._policy.on_time_advance(current_time_s)
        self._execute(action)

    def notify_cluster_ready(
        self, cluster_name: str, rpu: int, ready_time_s: float
    ) -> None:
        """Inform the policy that a pending cluster has become READY."""
        self._policy.on_cluster_ready(cluster_name, rpu, ready_time_s)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute(self, action: AutoscalingAction) -> None:
        """Execute all requests in an :class:`AutoscalingAction`."""
        for req in action.spin_ups:
            try:
                self._on_spin_up(req.reason, req.rpu)
            except Exception:
                logger.exception(
                    "on_spin_up callback failed (rpu=%d)", req.rpu
                )
        for req in action.tear_downs:
            try:
                self._on_tear_down(req.cluster_name)
            except Exception:
                logger.exception(
                    "on_tear_down callback failed (cluster=%s)",
                    req.cluster_name,
                )
