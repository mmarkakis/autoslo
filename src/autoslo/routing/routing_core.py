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
from typing import Optional, Union

from intervaltree import Interval  # type: ignore[import]

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.models.model_prediction import ModelPrediction
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query


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
        optimize_by_amount: bool,
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
        optimize_by_amount:
            If True, violations are measured in seconds of overshoot.
            If False, violations are binary (0 or 1) per query.

        Returns
        -------
        (before_cost, before_slo_violation)
        """
        # -- SLO violations -------------------------------------------------
        individual_violations: list[Union[float, bool]] = [
            (
                q.slo_violation_amount_s(
                    slo_resolver.resolve(q.tpcds_temp_and_q_idx)
                )
                if optimize_by_amount
                else q.violates_slo(
                    slo_resolver.resolve(q.tpcds_temp_and_q_idx)
                )
            )
            for q in snapshot.active_queries
        ]
        before_slo_violation = float(sum(individual_violations))

        # -- Billing cost ---------------------------------------------------
        before_query_intervals = [
            q.as_interval() for q in snapshot.active_queries
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
        optimize_by_amount: bool,
        before_cost: float,
        before_slo_violation: float,
    ) -> PlacementScore:
        """Score a hypothetical placement of *query* onto the cluster described
        by *snapshot*, given the latency *predictions* returned by the model.

        This is a corrected extraction of the scoring logic from
        ``WorkloadSimulator._find_best_cluster_for_query``.  The original
        code had a misalignment: ``latencies_after`` carried an extra element at
        index 0 (the incoming query's raw prediction) that shifted the
        ``zip(keys, latencies_after)`` used for billing intervals and the final
        latency map, causing each query to receive the *previous* query's
        predicted latency.  This version fixes the alignment by computing one
        latency per base query with no extra element.

        Parameters
        ----------
        query:
            The incoming query to be placed.
        snapshot:
            Cluster state snapshot.
        predictions:
            ``{query_id: ModelPrediction}`` for every query in the candidate
            group (all active queries on this cluster + the incoming query).
            Keyed by query ID.
        current_time_s:
            Arrival time of *query* (simulated or real).
        slo_resolver:
            Resolves per-query SLOs.
        optimize_by_amount:
            If True, measure violations in seconds; else binary.
        before_cost:
            Pre-computed cost of the cluster before adding *query*.
        before_slo_violation:
            Pre-computed SLO violation of the cluster before adding *query*.

        Returns
        -------
        PlacementScore
        """
        # All queries that will be on this cluster after placement.
        base_queries = list(snapshot.active_queries) + [query]

        # For each base query, predicted latency = max(current, model prediction).
        # For the incoming query, latency_s is -1, so it just gets the prediction.
        latencies: dict[str, float] = {}
        for q in base_queries:
            pred_s = predictions[q.query_id].overall_mean_s()
            latencies[q.query_id] = max(q.latency_s, pred_s)

        # -- After SLO violations ------------------------------------------
        individual_after_violations: list[Union[float, bool]] = [
            (
                max(
                    0.0,
                    latencies[q.query_id]
                    - slo_resolver.resolve(q.tpcds_temp_and_q_idx),
                )
                if optimize_by_amount
                else latencies[q.query_id]
                > slo_resolver.resolve(q.tpcds_temp_and_q_idx)
            )
            for q in base_queries
        ]
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

        best = scores[0]
        for candidate in scores[1:]:
            cmp = RoutingCore._slo_cmp_with_tolerance(
                candidate.marginal_slo_violation,
                best.marginal_slo_violation,
                tolerance,
            )
            if cmp < 0 or (
                cmp == 0 and candidate.marginal_cost < best.marginal_cost
            ):
                best = candidate
        return best

    # --------------------------------------------------------------------------
    # SLO headroom
    # --------------------------------------------------------------------------

    @staticmethod
    def compute_slo_headroom(
        active_queries: list[Query],
        slo_resolver: SloResolver,
    ) -> float:
        """Compute the minimum SLO headroom across all active queries.

        Headroom is defined as:

            h(q) = (SLO_q - L_q) / SLO_q

        where L_q is the current (predicted or observed) latency of query q.

        Returns 1.0 when there are no active queries (full headroom).
        Returns ≤ 0 when at least one query is at or past its SLO.

        Parameters
        ----------
        active_queries:
            Currently running queries with their latest latency estimates.
        slo_resolver:
            Resolves per-query SLOs.

        Returns
        -------
        Minimum headroom across all active queries. In [−∞, 1.0].
        """
        if not active_queries:
            return 1.0

        min_headroom = float("inf")
        for q in active_queries:
            slo_s = slo_resolver.resolve(q.tpcds_temp_and_q_idx)
            if slo_s <= 0:
                continue  # degenerate SLO, skip
            headroom = (slo_s - q.latency_s) / slo_s
            if headroom < min_headroom:
                min_headroom = headroom

        return min_headroom if min_headroom != float("inf") else 1.0

