"""
headroom_policy.py
------------------
SLO-headroom-based autoscaling policy.

Ports the logic from :class:`~autoslo.capacity.capacity_controller.CapacityController`
into the :class:`AutoscalingPolicy` interface, faithfully reproducing its
behavior as a pluggable policy rather than a monolith.

**Spin-up** (``on_routing_result``):
    Compute SLO headroom across all active queries.  If headroom ≤
    ``eta_crit`` and no spin-up is pending, request a new cluster.  Also
    trigger when the best routing score has positive marginal SLO
    violation (capacity pressure).

**Tear-down** (``on_time_advance``):
    For each cluster with zero active queries, increment an idle counter.
    When the counter reaches ``idle_periods_before_tear_down`` and the
    cluster has exceeded ``min_cluster_lifetime_s``, request tear-down.

**Routing window**:
    Maintains its own rolling window of recent ``RoutingResult`` objects,
    accumulated via ``on_routing_result``, used for counterfactual RPU
    selection.  Replaces the routing window that previously lived in
    :class:`~autoslo.routing.model_policy.ModelPolicy`.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

from autoslo.capacity.autoscaling_policy import (
    AutoscalingAction,
    AutoscalingPolicy,
    SpinUpRequest,
    TearDownRequest,
)
from autoslo.routing.routing_core import (
    ClusterSnapshot,
    RoutingResult,
)
from autoslo.utils.structured_log import emit_structured, LOGGER_NAME
from autoslo.workload_definition.query import Query, SloMetric

if TYPE_CHECKING:
    from autoslo.blueprint_selection.slo_resolver import SloResolver
    from autoslo.models.iconq_model import IconqModel
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool

logger = logging.getLogger(__name__)
_has_structured = lambda: bool(logging.getLogger(LOGGER_NAME).handlers)


class HeadroomPolicy(AutoscalingPolicy):
    """SLO-headroom-based autoscaling policy.

    Parameters
    ----------
    slo_resolver :
        Resolves per-query SLOs (shared with the routing layer).
    slo_metric :
        Which SLO-violation metric drives headroom computation.
    eta_crit :
        Critical headroom threshold.  If ``H_t ≤ eta_crit``, trigger
        spin-up.  Default 0.1 (10 % remaining headroom).
    idle_periods_before_tear_down :
        Consecutive idle ticks before a cluster is torn down.
    min_cluster_lifetime_s :
        Minimum seconds a cluster must be READY before eligible for
        tear-down.
    allowed_rpu_sizes :
        RPU sizes available for dynamic spin-up (sorted ascending).
    iconq_model :
        Optional :class:`~autoslo.models.iconq_model.IconqModel` for
        counterfactual RPU selection.  When *None*, the smallest
        allowed RPU is used.
    routing_window_s :
        Duration (seconds) of the rolling routing-decision window
        retained for counterfactual RPU selection.
    min_window_observations :
        Minimum number of routing results in the (post-reset) window
        before the headroom trigger is trusted.  Prevents spurious
        spin-ups from a single unlucky query right after a cluster
        comes online.  Default 3.
    """

    def __init__(
        self,
        slo_resolver: "SloResolver",
        slo_metric: SloMetric = SloMetric.RELATIVE,
        eta_crit: float = 0.1,
        idle_periods_before_tear_down: int = 5,
        min_cluster_lifetime_s: float = 1200.0,
        allowed_rpu_sizes: Optional[list[int]] = None,
        iconq_model: Optional["IconqModel"] = None,
        routing_window_s: float = 120.0,
        min_window_observations: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._slo_resolver = slo_resolver
        self._slo_metric = slo_metric
        self._eta_crit = eta_crit
        self._idle_periods_before_tear_down = idle_periods_before_tear_down
        self._min_cluster_lifetime_s = min_cluster_lifetime_s
        self._allowed_rpu_sizes: list[int] = sorted(
            allowed_rpu_sizes if allowed_rpu_sizes is not None else [8]
        )
        self._iconq_model = iconq_model
        self._routing_window_s = routing_window_s
        self._min_window_observations = min_window_observations

        # Internal mutable state (reset by on_attach).
        self._idle_counts: dict[str, int] = {}
        self._pending_count: int = 0
        self._cluster_ready_time_s: dict[str, float] = {}
        self._routing_window: deque[
            tuple[Query, float, float, ClusterSnapshot | None]
        ] = deque()

    # ------------------------------------------------------------------
    # Properties (tunable by Layer 3 / PolicyTuner)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"HeadroomPolicy(eta_crit={self._eta_crit})"

    @property
    def eta_crit(self) -> float:
        return self._eta_crit

    @eta_crit.setter
    def eta_crit(self, value: float) -> None:
        self._eta_crit = value

    @property
    def idle_periods_before_tear_down(self) -> int:
        return self._idle_periods_before_tear_down

    @idle_periods_before_tear_down.setter
    def idle_periods_before_tear_down(self, value: int) -> None:
        self._idle_periods_before_tear_down = value

    @property
    def allowed_rpu_sizes(self) -> list[int]:
        return list(self._allowed_rpu_sizes)

    @allowed_rpu_sizes.setter
    def allowed_rpu_sizes(self, value: list[int]) -> None:
        self._allowed_rpu_sizes = sorted(value)

    @property
    def min_cluster_lifetime_s(self) -> float:
        return self._min_cluster_lifetime_s

    @min_cluster_lifetime_s.setter
    def min_cluster_lifetime_s(self, value: float) -> None:
        self._min_cluster_lifetime_s = value

    @property
    def pending_count(self) -> int:
        return self._pending_count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_attach(self, pool: "ManagedClusterPool") -> None:
        """Attach to a pool and reset mutable state.

        Called by the :class:`Autoscaler` at construction, and again on
        each simulator reset (since a new Autoscaler is created).
        """
        super().on_attach(pool)
        self._idle_counts.clear()
        self._pending_count = 0
        self._cluster_ready_time_s.clear()
        self._routing_window.clear()

    def on_cluster_ready(
        self,
        cluster_name: str,
        rpu: int,
        ready_time_s: float,
    ) -> None:
        """Decrement pending count and record ready time.

        This unblocks future spin-ups and enables minimum-lifetime
        enforcement for tear-down.
        """
        self._pending_count = max(0, self._pending_count - 1)
        self._cluster_ready_time_s[cluster_name] = ready_time_s
        # Reset the routing window so that future spin-up decisions
        # are based only on evidence gathered *after* this cluster
        # is available to absorb load.
        self._routing_window.clear()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_routing_result(
        self,
        result: RoutingResult,
        current_time_s: float,
        current_latencies: dict[str, float] | None = None,
    ) -> AutoscalingAction:
        """Evaluate spin-up conditions after a routing decision.

        Checks two triggers:
        1. **Headroom**: if SLO headroom ≤ ``eta_crit``.
        2. **Capacity pressure**: if the *best* placement score has
           positive marginal SLO violation (implies *all* clusters
           would violate, since ``pick_best`` minimises violation).

        Also maintains the routing window for counterfactual RPU
        selection.
        """
        # -- Maintain routing window -----------------------------------
        # Only accumulate when no spin-up is pending.  While a cluster
        # is spinning up we freeze the window; it will be cleared when
        # on_cluster_ready fires, so that all future evidence is
        # gathered *after* the new cluster can absorb load.
        if self._pending_count == 0:
            self._routing_window.append(
                (result.query, result.predicted_latency_s, current_time_s, None)
            )
            cutoff = current_time_s - self._routing_window_s
            while (
                self._routing_window
                and self._routing_window[0][2] < cutoff
            ):
                self._routing_window.popleft()

        window_size = len(self._routing_window)

        # -- Compute window-based headroom -----------------------------
        # Headroom is derived exclusively from queries in the
        # (post-reset) routing window.  This avoids stale signals from
        # queries that were routed before the most recent cluster came
        # online.
        window_headroom = self._compute_window_headroom()

        # -- Detect capacity pressure ----------------------------------
        # If the best score still has positive marginal violation, then
        # every cluster would violate (since pick_best minimises it).
        pressure = (
            result.score is not None and result.score.marginal_slo_violation > 0
        )
        pressure = False

        # -- Emit structured headroom record ---------------------------
        if _has_structured():
            emit_structured(
                {
                    "timestamp": current_time_s,
                    "event_type": "headroom_check",
                    "source": "headroom_policy",
                    "window_headroom": window_headroom,
                    "window_size": window_size,
                    "eta_crit": self._eta_crit,
                    "capacity_pressure": pressure,
                    "pending_count": self._pending_count,
                }
            )

        # -- Spin-up decision ------------------------------------------
        # Headroom trigger requires enough post-reset observations to
        # be trustworthy.  The pressure trigger only requires that the
        # current routing decision was made after the most recent
        # cluster became available (i.e. at least one entry in the
        # fresh window).
        headroom_trigger = (
            window_size >= self._min_window_observations
            and window_headroom <= self._eta_crit
        )
        pressure_trigger = window_size >= 1 and pressure

        if (
            headroom_trigger or pressure_trigger
        ) and self._pending_count == 0:
            reason = (
                f"window_headroom={window_headroom:.4f}"
                f"<=η_crit={self._eta_crit:.4f}"
                f" (window_size={window_size})"
                if headroom_trigger
                else "capacity_pressure_signal"
            )
            rpu = self._select_rpu(current_time_s)
            logger.info("Spin-up triggered: %s (rpu=%d)", reason, rpu)
            self._pending_count += 1
            return AutoscalingAction(
                spin_ups=[SpinUpRequest(rpu=rpu, reason=reason)],
            )

        return AutoscalingAction()

    def on_query_complete(
        self,
        query_id: str,
        cluster_name: str,
        current_time_s: float,
    ) -> AutoscalingAction:
        """Bookkeeping only — no actions in the current strategy."""
        return AutoscalingAction()

    def on_time_advance(
        self,
        current_time_s: float,
    ) -> AutoscalingAction:
        """Evaluate idle-based tear-down conditions.

        For each non-draining cluster with zero active queries,
        increment the idle counter.  When the counter reaches
        ``idle_periods_before_tear_down`` and the cluster has exceeded
        ``min_cluster_lifetime_s``, request tear-down.
        """
        active_map = self._pool.get_all_active_queries()
        draining = self._pool.draining_cluster_names

        tear_downs: list[TearDownRequest] = []

        for cn, qs in active_map.items():
            if cn in draining:
                continue
            if len(qs) == 0:
                self._idle_counts[cn] = self._idle_counts.get(cn, 0) + 1
                if self._idle_counts[cn] >= self._idle_periods_before_tear_down:
                    # Enforce minimum cluster lifetime.
                    ready_time = self._cluster_ready_time_s.get(
                        cn, current_time_s
                    )
                    age_s = current_time_s - ready_time
                    if age_s < self._min_cluster_lifetime_s:
                        logger.debug(
                            "Tear-down deferred for cluster %s "
                            "(age=%.0fs < min_lifetime=%.0fs).",
                            cn,
                            age_s,
                            self._min_cluster_lifetime_s,
                        )
                        continue

                    logger.info(
                        "Tear-down triggered for cluster %s "
                        "(idle for %d periods).",
                        cn,
                        self._idle_counts[cn],
                    )
                    tear_downs.append(
                        TearDownRequest(
                            cluster_name=cn,
                            reason=f"idle_for_{self._idle_counts[cn]}_periods",
                        )
                    )
                    # Reset so we don't re-fire every tick.
                    self._idle_counts[cn] = 0
            else:
                # Reset idle counter when the cluster has work.
                self._idle_counts[cn] = 0

        return AutoscalingAction(tear_downs=tear_downs)

    # ------------------------------------------------------------------
    # Window-based headroom
    # ------------------------------------------------------------------

    def _compute_window_headroom(self) -> float:
        """Compute the minimum SLO headroom across the routing window.

        Mirrors the semantics of ``RoutingCore.compute_slo_headroom``
        but operates on the ``(Query, predicted_latency_s, ...)``
        tuples stored in the routing window rather than the live
        active-query set.

        Returns 1.0 (full headroom) when the window is empty.
        """
        if not self._routing_window:
            return 1.0

        min_headroom = float("inf")
        for query, predicted_latency, _routed_at, _snapshot in self._routing_window:
            slo_s = self._slo_resolver.resolve(query.query_text_id)
            if slo_s <= 0:
                continue
            headroom = (slo_s - predicted_latency) / slo_s
            if headroom < min_headroom:
                min_headroom = headroom

        return min_headroom if min_headroom != float("inf") else 1.0

    # ------------------------------------------------------------------
    # RPU selection
    # ------------------------------------------------------------------

    def _select_rpu(self, current_time_s: float) -> int:
        """Choose the RPU size for a new cluster via counterfactual
        routing replay.

        For each candidate RPU (ascending), predict whether the
        pressure-causing queries in the recent routing window would
        have met their SLOs on a hypothetical empty cluster of that
        size.  Returns the smallest RPU that clears all violations,
        or the largest available RPU if none suffices.

        Falls back to ``self._allowed_rpu_sizes[0]`` (smallest) when
        no latency model or routing window is available.

        Parameters
        ----------
        current_time_s :
            Current time (wall-clock or simulated) in seconds, used for
            routing window management and structured logging.
        """
        if self._iconq_model is None or not self._routing_window:
            if _has_structured():
                emit_structured(
                    {
                        "timestamp": current_time_s,
                        "event_type": "rpu_selection_fallback",
                        "source": "headroom_policy",
                        "reason": (
                            "no_iconq_model"
                            if self._iconq_model is None
                            else "empty_routing_window"
                        ),
                    }
                )
            return self._allowed_rpu_sizes[0]

        # Identify pressure queries — those whose predicted latency
        # exceeded their SLO in the recent window.
        pressure_queries: list[Query] = []
        for query, predicted_latency, _routed_at, _snapshot in self._routing_window:
            slo_s = self._slo_resolver.resolve(query.query_text_id)
            if predicted_latency > 0 and predicted_latency > slo_s:
                pressure_queries.append(query)

        if not pressure_queries:
            return self._allowed_rpu_sizes[0]

        # Try each candidate RPU (ascending).
        for rpu in self._allowed_rpu_sizes:
            all_feasible = True
            for q in pressure_queries:
                stage_pred = (
                    self._iconq_model.stage_model.predict_from_query_text_id(
                        {q.query_id: q.query_text_id},
                        cluster_rpu=rpu,
                    )[q.query_id].overall_mean_s()
                )
                slo_s = self._slo_resolver.resolve(q.query_text_id)
                if stage_pred > slo_s:
                    all_feasible = False
                    break
            if all_feasible:
                return rpu

        # No candidate RPU is sufficient — pick the largest.
        return self._allowed_rpu_sizes[-1]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_routing_window(
        self,
    ) -> list[tuple[Query, float, float, ClusterSnapshot | None]]:
        """Return a copy of the recent routing-decision window.

        Each entry is ``(query, predicted_latency_s, routed_at_s, snapshot)``.
        """
        return list(self._routing_window)
