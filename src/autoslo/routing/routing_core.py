"""
routing_core.py
---------------
Stateless routing logic extracted from WorkloadSimulator and RIconq.

This module provides the pure computational core for query routing decisions:
scoring a placement, comparing candidates, computing SLO headroom, and
computing marginal billing cost. It has no state, no I/O, and no threading —
those concerns belong to the adapter layer (simulator or online runner).

Both the offline simulator and the online router should call these functions
so that they evaluate exactly the same routing policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from intervaltree import Interval  # type: ignore[import]

if TYPE_CHECKING:
    from autoslo.models.iconq_model import IconqModel
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool

from autoslo.blueprint_selection.slo_resolver import (
    SloResolver,
    slo_violation as _slo_violation,
    query_interval as _query_interval,
)
from autoslo.models.model_prediction import ModelPrediction
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query, QueryTextId, SloMetric


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ClusterSnapshot:
    """Immutable snapshot of a cluster's state at routing time.

    This captures everything the scoring functions need to evaluate a
    hypothetical placement, without requiring access to the simulator's
    mutable bookkeeping.
    """

    cluster_name: str
    cost_per_second: float
    active_queries: list[Query]
    billing_window_start_s: Optional[float]


@dataclass
class PlacementScore:
    """Result of scoring a query placement on a single cluster."""

    cluster_name: str
    marginal_slo_violation: float
    marginal_cost: float
    latencies: dict[str, float]
    """Maps query_id → predicted latency for every query involved
    (the incoming query plus all active co-runners)."""

    cache_risk: float = 0.0
    """Cache-risk penalty set by :class:`CacheAwarePolicy` (default 0)."""

    adjusted_slo_violation: float = 0.0
    """``marginal_slo_violation + λ·cache_risk``.  When zero (default),
    :meth:`RoutingCore.pick_best` falls back to ``marginal_slo_violation``."""


@dataclass
class RoutingResult:
    """Full result of a routing decision.

    Returned by :meth:`Router.route_query_with_predictions` to give
    callers access to the placement score and a fully-built tracking
    query (with featurisation already populated).

    The chosen cluster's per-query predictions are stored as independent
    fields on this object rather than being baked into the Query.
    """

    cluster_name: str
    score: Optional[PlacementScore]
    query: Query
    stage_prediction_s: float = -1
    predicted_latency_s: float = -1


class RoutingCore:
    """Namespace for routing core functions."""

    # --------------------------------------------------------------------------
    # Before-state computation
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_before_state(
        snapshot: ClusterSnapshot,
        current_time_s: float,
        slo_resolver: SloResolver,
        slo_metric: SloMetric,
        latencies: dict[str, float],
    ) -> tuple[float, float]:
        """Compute the cost and SLO-violation metric for a cluster *before*
        adding a new query.

        Parameters
        ----------
        snapshot:
            Current cluster state.
        current_time_s:
            Wall-clock (or simulated) time of the incoming query's arrival.
        slo_resolver:
            Resolves per-query SLOs.
        slo_metric:
            Which SLO-violation metric to optimise on.
        latencies:
            ``{query_id: predicted_latency_s}`` for the currently-active queries.

        Returns
        -------
        (before_cost, before_slo_violation)
        """
        # -- SLO violations -------------------------------------------------
        individual_violations: list[float] = [
            _slo_violation(
                latencies.get(q.query_id, -1.0),
                slo_resolver.resolve(q.query_text_id),
                slo_metric,
            )
            for q in snapshot.active_queries
        ]
        before_slo_violation = float(sum(individual_violations))

        # -- Billing cost ---------------------------------------------------
        before_query_intervals = [
            _query_interval(q.rel_start_time_s, latencies.get(q.query_id, 0.0), q.query_id)
            for q in snapshot.active_queries
        ]
        if snapshot.billing_window_start_s is not None:
            before_query_intervals.append(
                Interval(snapshot.billing_window_start_s, current_time_s)
            )
        before_billed_s = sum(
            iv.end - iv.begin
            for iv in Billing.billed_intervals(before_query_intervals)
        )
        before_cost = snapshot.cost_per_second * before_billed_s

        return before_cost, before_slo_violation

    # --------------------------------------------------------------------------
    # Placement scoring
    # --------------------------------------------------------------------------

    @staticmethod
    def score_placement(
        query: Query,
        snapshot: ClusterSnapshot,
        predictions: dict[str, ModelPrediction],
        current_time_s: float,
        slo_resolver: SloResolver,
        slo_metric: SloMetric,
        before_cost: float,
        before_slo_violation: float,
        current_latencies: dict[str, float],
    ) -> PlacementScore:
        """Score a hypothetical placement of *query* onto the cluster described
        by *snapshot*, given the latency *predictions* returned by the model.

        Parameters
        ----------
        query:
            The incoming query to be placed.
        snapshot:
            Cluster state snapshot.
        predictions:
            ``{query_id: ModelPrediction}`` for every query in the candidate
            group (all active queries on this cluster + the incoming query).
        current_time_s:
            Arrival time of *query* (simulated or real).
        slo_resolver:
            Resolves per-query SLOs.
        slo_metric:
            Which SLO-violation metric to optimise on.
        before_cost:
            Pre-computed cost of the cluster before adding *query*.
        before_slo_violation:
            Pre-computed SLO violation of the cluster before adding *query*.
        current_latencies:
            ``{query_id: predicted_latency_s}`` for currently-active queries.
            The incoming query may or may not have an entry; if absent, the
            model prediction alone is used.

        Returns
        -------
        PlacementScore
        """
        # All queries that will be on this cluster after placement.
        base_queries = list(snapshot.active_queries) + [query]

        # For each base query, predicted latency = max(current, model prediction).
        # For the incoming query, current_latencies won't have an entry so it
        # just gets the prediction.
        latencies: dict[str, float] = {}
        for q in base_queries:
            pred_s = predictions[q.query_id].overall_mean_s()
            current = current_latencies.get(q.query_id, -1.0)
            latencies[q.query_id] = max(current, pred_s)

        # -- After SLO violations ------------------------------------------
        individual_after_violations: list[float] = []
        for q in base_queries:
            slo_s = slo_resolver.resolve(q.query_text_id)
            predicted_latency = latencies[q.query_id]
            individual_after_violations.append(
                _slo_violation(predicted_latency, slo_s, slo_metric)
            )
        after_slo_violation = float(sum(individual_after_violations))
        marginal_slo_violation = after_slo_violation - before_slo_violation

        # -- After billing cost ---------------------------------------------
        after_query_intervals = [
            Interval(
                q.rel_start_time_s, q.rel_start_time_s + latencies[q.query_id]
            )
            for q in base_queries
        ]
        if snapshot.billing_window_start_s is not None:
            after_query_intervals.append(
                Interval(snapshot.billing_window_start_s, current_time_s)
            )
        after_billed_s = sum(
            iv.end - iv.begin
            for iv in Billing.billed_intervals(after_query_intervals)
        )
        after_cost = snapshot.cost_per_second * after_billed_s
        marginal_cost = after_cost - before_cost

        return PlacementScore(
            cluster_name=snapshot.cluster_name,
            marginal_slo_violation=marginal_slo_violation,
            marginal_cost=marginal_cost,
            latencies=latencies,
        )

    # --------------------------------------------------------------------------
    # Best-placement selection
    # --------------------------------------------------------------------------

    TOLERANCE_FOR_SLO_VIOLATION_AMOUNT_OPTIMIZATION_S = 1e-4

    @staticmethod
    def _slo_cmp_with_tolerance(
        a: float,
        b: float,
        tolerance: float = TOLERANCE_FOR_SLO_VIOLATION_AMOUNT_OPTIMIZATION_S,
    ) -> int:
        """Compare two SLO violation amounts with a tolerance.

        Returns -1 if a < b, 0 if approximately equal, 1 if a > b.
        """
        if a + tolerance < b:
            return -1
        elif b + tolerance < a:
            return 1
        return 0

    @staticmethod
    def pick_best(
        scores: list[PlacementScore],
        tolerance: float = TOLERANCE_FOR_SLO_VIOLATION_AMOUNT_OPTIMIZATION_S,
    ) -> PlacementScore:
        """Select the best placement from a list of scored candidates.

        Selection is lexicographic: minimise marginal SLO violation first, then
        minimise marginal cost to break ties (within tolerance).

        Parameters
        ----------
        scores:
            Non-empty list of placement scores to compare.
        tolerance:
            Tolerance for treating two SLO violations as equal.

        Returns
        -------
        The best PlacementScore.
        """
        if not scores:
            raise ValueError("Cannot pick from an empty list of scores.")

        def _effective_slo(s: PlacementScore) -> float:
            return (
                s.adjusted_slo_violation
                if s.adjusted_slo_violation != 0.0
                else s.marginal_slo_violation
            )

        best = scores[0]
        for candidate in scores[1:]:
            cmp = RoutingCore._slo_cmp_with_tolerance(
                _effective_slo(candidate),
                _effective_slo(best),
                tolerance,
            )
            if cmp < 0 or (
                cmp == 0 and candidate.marginal_cost < best.marginal_cost
            ):
                best = candidate
        return best

    # --------------------------------------------------------------------------
    # Full-pipeline placement scoring (featurise → snapshot → score)
    # --------------------------------------------------------------------------

    @staticmethod
    def score_query_on_clusters(
        iconq_model: "IconqModel",
        pool: "ManagedClusterPool",
        query_id: str,
        query_text_id: QueryTextId,
        start_time_s: float,
        slo_resolver: SloResolver,
        slo_metric: SloMetric,
        current_latencies: dict[str, float],
        cluster_names: list[str] | None = None,
    ) -> tuple[dict[str, "PlacementScore"], Query, dict[str, float]]:
        """Featurise a query and score it against a set of the pool's clusters.

        This is the shared core used by both
        :meth:`~autoslo.routing.model_policy.ModelPolicy.route_with_details`
        and
        :meth:`~autoslo.workload_execution.workload_simulator.WorkloadSimulator._score_with_model`.
        Centralising the featurise → snapshot → stage-predict → before-state
        → batched-predict → score pipeline here keeps both callers thin and
        guarantees they use exactly the same arithmetic.

        Parameters
        ----------
        iconq_model:
            Loaded :class:`~autoslo.models.iconq_model.IconqModel`.
        pool:
            Live cluster pool.
        query_id:
            Unique string identifier for the incoming query.
        query_text_id:
            Template identifier for the incoming query.
        start_time_s:
            Arrival time (simulated or real) in seconds.
        slo_resolver:
            Resolves per-template SLOs.
        slo_metric:
            Which SLO-violation metric drives scoring.
        current_latencies:
            ``{query_id: predicted_latency_s}`` for all currently-active
            queries across the pool.  Used in ``compute_before_state`` and
            ``score_placement``.
        cluster_names:
            Restrict scoring to these cluster names.  If *None*, score all
            READY clusters currently in the pool.

        Returns
        -------
        (scores, incoming, stage_preds)
            scores      : ``{cluster_name: PlacementScore}`` for every cluster
                          that produced valid model predictions.
            incoming    : Frozen :class:`Query` with ``stage_predictions_per_rpu``
                          populated for all unique RPUs across eligible clusters.
            stage_preds : ``{cluster_name: stage_pred_s}`` for every scored
                          cluster.
        """
        # ConcurrentQueryDataset lives in nn/, which does not import
        # routing_core — safe to import here at runtime.
        from autoslo.nn.concurrent_query_dataset import (  # noqa: PLC0415
            ConcurrentQueryDataset,
        )

        eligible: list[str] = (
            list(cluster_names)
            if cluster_names is not None
            else list(pool.cluster_names)
        )

        # -- Featurise the incoming query ----------------------------------
        featurization = (
            iconq_model.iconq_query_featurizer
            .featurize_from_query_text_id(query_text_id)
        )

        # -- Stage-model predictions (one per unique RPU) ------------------
        stage_predictions_per_rpu: dict[int, float] = {}
        stage_preds: dict[str, float] = {}
        for cn in eligible:
            rpu = pool.get_rpu(cn)
            if rpu not in stage_predictions_per_rpu:
                sp = (
                    iconq_model.stage_model
                    .predict_from_query_text_id(
                        {query_id: query_text_id}, rpu
                    )[query_id].overall_mean_s()
                )
                stage_predictions_per_rpu[rpu] = sp
            stage_preds[cn] = stage_predictions_per_rpu[rpu]

        # -- Build *frozen* incoming Query ---------------------------------
        incoming = Query(
            query_id=query_id,
            query_text_id=query_text_id,
            rel_start_time_s=start_time_s,
            featurization=featurization,
            stage_predictions_per_rpu=stage_predictions_per_rpu,
        )

        # -- Atomic pool snapshot ------------------------------------------
        snapshots, cluster_to_base_to_neighbors = pool.build_routing_context(
            incoming
        )
        snapshots = {cn: s for cn, s in snapshots.items() if cn in eligible}
        cluster_to_base_to_neighbors = {
            cn: n
            for cn, n in cluster_to_base_to_neighbors.items()
            if cn in eligible
        }

        # -- Before-state per cluster -------------------------------------
        before: dict[str, tuple[float, float]] = {}
        for cn, snap in snapshots.items():
            before[cn] = RoutingCore.compute_before_state(
                snapshot=snap,
                current_time_s=start_time_s,
                slo_resolver=slo_resolver,
                slo_metric=slo_metric,
                latencies=current_latencies,
            )

        # -- Batched model predictions ------------------------------------
        dataset = ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=iconq_model.iconq_interaction_featurizer,
            cluster_to_base_to_neighbors=cluster_to_base_to_neighbors,
        )
        all_predictions = iconq_model.predict_from_dataset(dataset)

        # -- Score each cluster ------------------------------------------
        scores: dict[str, PlacementScore] = {}
        for cn, predictions in all_predictions.items():
            if cn not in before or cn not in snapshots:
                continue
            bc, bv = before[cn]
            scores[cn] = RoutingCore.score_placement(
                query=incoming,
                snapshot=snapshots[cn],
                predictions=predictions,
                current_time_s=start_time_s,
                slo_resolver=slo_resolver,
                slo_metric=slo_metric,
                before_cost=bc,
                before_slo_violation=bv,
                current_latencies=current_latencies,
            )

        return scores, incoming, stage_preds

    # --------------------------------------------------------------------------
    # SLO headroom
    # --------------------------------------------------------------------------

    @staticmethod
    def compute_slo_headroom(
        active_queries: list[Query],
        slo_resolver: SloResolver,
        latencies: dict[str, float],
    ) -> float:
        """Compute the minimum SLO headroom across all active queries.

        Headroom is defined as ``1 − relative_violation(q)`` which equals
        ``(SLO − latency) / SLO`` when the query is within its SLO and
        becomes negative once it overshoots.

        Returns 1.0 when there are no active queries (full headroom).
        Returns ≤ 0 when at least one query is at or past its SLO.

        Parameters
        ----------
        active_queries:
            Currently running queries.
        slo_resolver:
            Resolves per-query SLOs.
        latencies:
            ``{query_id: predicted_latency_s}`` for the active queries.

        Returns
        -------
        Minimum headroom across all active queries. In [−∞, 1.0].
        """
        if not active_queries:
            return 1.0

        min_headroom = float("inf")
        for q in active_queries:
            slo_s = slo_resolver.resolve(q.query_text_id)
            if slo_s <= 0:
                continue  # degenerate SLO, skip
            lat = latencies.get(q.query_id, -1.0)
            headroom = (slo_s - lat) / slo_s
            if headroom < min_headroom:
                min_headroom = headroom

        return min_headroom if min_headroom != float("inf") else 1.0

