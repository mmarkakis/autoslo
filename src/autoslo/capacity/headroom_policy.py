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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from autoslo.blueprints.cluster import Cluster
from autoslo.capacity.autoscaling_policy import (
    AutoscalingAction,
    AutoscalingPolicy,
    SpinUpRequest,
    TearDownRequest,
)
from autoslo.routing.routing_core import ClusterSnapshot, RoutingResult
from autoslo.routing.routing_policy import RoutingPolicy
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.utils.structured_log import LOGGER_NAME, emit_structured
from autoslo.workload_definition.query import Query

if TYPE_CHECKING:
    from autoslo.models.iconq_model import IconqModel
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool
    from autoslo.slo.slo_resolver import SloResolver


# ---------------------------------------------------------------------------
# Lightweight virtual cluster for counterfactual replay
# ---------------------------------------------------------------------------


@dataclass
class _VirtualCluster:
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
    routing_policy :
        Optional :class:`~autoslo.routing.routing_policy.RoutingPolicy`
        reference used for counterfactual routing replay during RPU
        selection.  When *None*, falls back to stage-model-only sizing.
    slo_threshold :
        Maximum acceptable value for the aggregate SLO-violation metric
        during counterfactual replay.  Interpretation depends on
        *slo_metric*: for ``BINARY`` it is the tolerated violation
        **rate** (fraction); for ``ABSOLUTE_S`` / ``RELATIVE`` it is
        the maximum aggregate violation sum.  Default 0.0.
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
        slo_objective: SloObjective,
        eta_crit: float = 0.1,
        idle_periods_before_tear_down: int = 5,
        min_cluster_lifetime_s: float = 1200.0,
        allowed_rpu_sizes: Optional[list[int]] = None,
        iconq_model: Optional["IconqModel"] = None,
        routing_policy: Optional[RoutingPolicy] = None,
        routing_window_s: float = 120.0,
        min_window_observations: int = 3,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._slo_resolver = slo_resolver
        self._slo_objective = slo_objective
        self._eta_crit = eta_crit
        self._idle_periods_before_tear_down = idle_periods_before_tear_down
        self._min_cluster_lifetime_s = min_cluster_lifetime_s
        self._allowed_rpu_sizes: list[int] = sorted(
            allowed_rpu_sizes if allowed_rpu_sizes is not None else [8]
        )
        self._iconq_model = iconq_model
        self._routing_policy = routing_policy
        self._routing_window_s = routing_window_s
        self._min_window_observations = min_window_observations

        # Internal mutable state (reset by on_attach).
        self._idle_counts: dict[str, int] = {}
        self._pending_count: int = 0
        self._cluster_ready_time_s: dict[str, float] = {}
        self._routing_window: deque[
            tuple[Query, float, float, ClusterSnapshot | None]
        ] = deque()
        self._window_initial_snapshots: dict[str, ClusterSnapshot] | None = None
        self._window_initial_latencies: dict[str, float] | None = None

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
        self._window_initial_snapshots = None
        self._window_initial_latencies = None

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
        self._window_initial_snapshots = None
        self._window_initial_latencies = None

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
            # Capture initial pool state on the first entry of a fresh
            # window (after on_cluster_ready or on_attach cleared it).
            if not self._routing_window:
                self._window_initial_snapshots = self._pool.build_snapshots()
                self._window_initial_latencies = dict(current_latencies or {})

            self._routing_window.append(
                (result.query, result.predicted_latency_s, current_time_s, None)
            )
            cutoff = current_time_s - self._routing_window_s
            while self._routing_window and self._routing_window[0][2] < cutoff:
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

        if (headroom_trigger or pressure_trigger) and self._pending_count == 0:
            reason = (
                f"window_headroom={window_headroom:.4f}"
                f"<=η_crit={self._eta_crit:.4f}"
                f" (window_size={window_size})"
                if headroom_trigger
                else "capacity_pressure_signal"
            )
            rpu = self._select_rpu(current_time_s)
            logger.debug("Spin-up triggered: %s (rpu=%d)", reason, rpu)
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

                    logger.debug(
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
        for (
            query,
            predicted_latency,
            _routed_at,
            _snapshot,
        ) in self._routing_window:
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

        For each candidate RPU (ascending), replay all queries from the
        routing window through the routing policy as if a new cluster of
        that RPU had been available from the window's start.  Returns
        the smallest RPU whose counterfactual SLO-compliance metric
        is within ``slo_threshold``, or the RPU that gets closest to
        the threshold if none qualifies.

        Falls back to the legacy stage-model-only sizing when no
        ``routing_policy`` is available, and to the smallest RPU when
        no model or routing window exists.

        Parameters
        ----------
        current_time_s :
            Current time (wall-clock or simulated) in seconds, used for
            routing window management and structured logging.
        """
        # -- Guard: prerequisites for counterfactual replay ---------------
        can_replay = (
            self._routing_policy is not None
            and self._iconq_model is not None
            and self._routing_window
            and self._window_initial_snapshots is not None
        )

        if not can_replay:
            return self._select_rpu_legacy(current_time_s)

        # -- Counterfactual replay per candidate RPU ----------------------
        best_rpu: int | None = None
        best_metric: float = float("inf")

        for rpu in self._allowed_rpu_sizes:
            metric = self._counterfactual_replay(rpu, current_time_s)

            if _has_structured():
                emit_structured(
                    {
                        "timestamp": current_time_s,
                        "event_type": "rpu_counterfactual",
                        "source": "headroom_policy",
                        "candidate_rpu": rpu,
                        "metric": metric,
                        "slo_threshold": self._slo_objective.slo_threshold,
                    }
                )

            if metric <= self._slo_objective.slo_threshold:
                return rpu  # smallest acceptable
            if metric < best_metric:
                best_metric = metric
                best_rpu = rpu

        # No RPU fully satisfies the threshold — pick closest.
        return best_rpu if best_rpu is not None else self._allowed_rpu_sizes[-1]

    # ------------------------------------------------------------------
    # Counterfactual replay
    # ------------------------------------------------------------------

    def _counterfactual_replay(
        self,
        candidate_rpu: int,
        current_time_s: float,
    ) -> float:
        """Replay the routing window with a hypothetical new cluster of
        *candidate_rpu* and return the aggregate SLO-violation metric.

        Lower is better.  The metric's semantics depend on
        ``self._slo_metric``:

        * ``BINARY``     – violation **rate** (fraction of queries).
        * ``ABSOLUTE_S`` – total violation seconds.
        * ``RELATIVE``   – total relative violation.
        """
        assert self._routing_policy is not None
        assert self._iconq_model is not None
        assert self._window_initial_snapshots is not None

        # 1. Initialise virtual clusters from stored snapshots.
        virtuals: dict[str, _VirtualCluster] = {}
        initial_lats = self._window_initial_latencies or {}

        for cn, snap in self._window_initial_snapshots.items():
            rpu = self._pool.get_rpu(cn)
            vc = _VirtualCluster(
                rpu=rpu,
                cost_per_second=snap.cost_per_second,
                billing_window_start_s=snap.billing_window_start_s,
            )
            for q in snap.active_queries:
                lat = initial_lats.get(q.query_id, -1.0)
                vc.active_queries[q.query_id] = q
                vc.latencies[q.query_id] = lat
                if lat > 0:
                    vc.completion_times[q.query_id] = q.rel_start_time_s + lat
            virtuals[cn] = vc

        # 2. Add the hypothetical new cluster (empty).
        hyp_name = "__hypothetical__"
        virtuals[hyp_name] = _VirtualCluster(
            rpu=candidate_rpu,
            cost_per_second=Cluster.cost_per_second_for_rpu(candidate_rpu),
        )

        # 3. Ensure window queries have stage predictions for the
        #    candidate RPU (needed by the IconQ featuriser).
        window_queries = self._augment_window_queries(candidate_rpu)

        # 4. Sequential replay.
        lat_and_slos = []

        for query, _orig_lat, arrival_time_s in window_queries:
            # a. Expire completed queries on all virtual clusters.
            for vc in virtuals.values():
                vc.expire_before(arrival_time_s)

            # b. Build snapshots and RPU map for scoring.
            snapshots = {cn: vc.to_snapshot(cn) for cn, vc in virtuals.items()}
            cluster_rpus = {cn: vc.rpu for cn, vc in virtuals.items()}
            current_lats = {}
            for vc in virtuals.values():
                current_lats.update(vc.latencies)

            # c. Route via the routing policy's counterfactual scorer.
            result = self._routing_policy.score_counterfactual(
                query=query,
                arrival_time_s=arrival_time_s,
                snapshots=snapshots,
                cluster_rpus=cluster_rpus,
                current_latencies=current_lats,
            )

            if result is None:
                # Policy cannot produce a score — skip this query.
                continue

            chosen_cn, returned_lats = result

            # d. Update virtual state for the chosen cluster.
            virtuals[chosen_cn].add_query(query, returned_lats)

            # e. Record latency for the incoming query.
            pred_lat = returned_lats.get(query.query_id, -1.0)
            slo_s = self._slo_resolver.resolve(query.query_text_id)
            lat_and_slos.append((pred_lat, slo_s))

        aggregate = self._slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        return aggregate

    def _augment_window_queries(
        self,
        candidate_rpu: int,
    ) -> list[tuple[Query, float, float]]:
        """Return window queries augmented with stage predictions for
        *candidate_rpu*.

        Returns a list of ``(query, original_predicted_latency, arrival_time)``
        sorted by arrival time.  If a query already has a stage prediction for
        *candidate_rpu*, it is returned as-is; otherwise a new ``Query`` is
        created with the additional entry.
        """
        assert self._iconq_model is not None
        result: list[tuple[Query, float, float]] = []

        for query, pred_lat, arrival_time, _snap in self._routing_window:
            if candidate_rpu in query.stage_predictions_per_rpu:
                result.append((query, pred_lat, arrival_time))
            else:
                # Compute missing stage prediction.
                sp = self._iconq_model.stage_model.predict_from_query_text_id(
                    {query.query_id: query.query_text_id},
                    cluster_rpu=candidate_rpu,
                )[query.query_id].overall_mean_s()
                new_preds = dict(query.stage_predictions_per_rpu)
                new_preds[candidate_rpu] = sp
                augmented = Query(
                    query_id=query.query_id,
                    query_text_id=query.query_text_id,
                    featurization=query.featurization,
                    abs_start_time=query.abs_start_time,
                    rel_start_time_s=query.rel_start_time_s,
                    repetition_id=query.repetition_id,
                    stage_predictions_per_rpu=new_preds,
                )
                result.append((augmented, pred_lat, arrival_time))

        # Sort by arrival time (window deque is already in order, but be safe).
        result.sort(key=lambda x: x[2])
        return result

    # ------------------------------------------------------------------
    # Legacy RPU selection (stage-model-only fallback)
    # ------------------------------------------------------------------

    def _select_rpu_legacy(self, current_time_s: float) -> int:
        """Original stage-model-only RPU selection.

        Used as a fallback when ``routing_policy`` is not available for
        counterfactual replay.
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
        pressure_queries_and_slo_s: list[tuple[Query, float]] = []
        for (
            query,
            predicted_latency,
            _routed_at,
            _snapshot,
        ) in self._routing_window:
            slo_s = self._slo_resolver.resolve(query.query_text_id)
            if predicted_latency > 0 and predicted_latency > slo_s:
                pressure_queries_and_slo_s.append((query, slo_s))

        if not pressure_queries_and_slo_s:
            return self._allowed_rpu_sizes[0]

        # Try each candidate RPU (ascending).
        for rpu in self._allowed_rpu_sizes:
            pred_lats_and_slo_s: list[tuple[float, float]] = []
            for q, slo_s in pressure_queries_and_slo_s:
                stage_pred = (
                    self._iconq_model.stage_model.predict_from_query_text_id(
                        {q.query_id: q.query_text_id},
                        cluster_rpu=rpu,
                    )[q.query_id].overall_mean_s()
                )
                pred_lats_and_slo_s.append((stage_pred, slo_s))
            violation_rate = SloMetric.BINARY.aggregate_batch(
                pred_lats_and_slo_s
            )
            if violation_rate == 0.0:
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
