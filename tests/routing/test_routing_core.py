"""
Tests for :mod:`autoslo.routing.routing_core`.

Covers:
- compute_before_state (empty/populated clusters, with/without billing window)
- score_placement (alignment correctness, SLO violations, billing cost)
- pick_best (lexicographic selection, tolerance handling)
- compute_slo_headroom (empty, healthy, violated)
- _slo_cmp_with_tolerance (comparison edge cases)
"""

from __future__ import annotations

import pytest

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.models.model_prediction import ModelPrediction
from autoslo.routing.routing_core import (
    ClusterSnapshot,
    PlacementScore,
    RoutingCore,
)
from autoslo.workload_definition.query import Query

from autoslo.utils.billing import Billing

from intervaltree import Interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(
    query_id: str,
    template: str = "1_0",
    start: float = 0.0,
    latency: float = -1.0,
) -> Query:
    """Create a minimal Query for testing."""
    return Query(
        query_id=query_id,
        tpcds_temp_and_q_idx=template,
        rel_start_time_s=start,
        latency_s=latency,
    )


def _pred(mean: float) -> ModelPrediction:
    """Create a constant-mean ModelPrediction."""
    return ModelPrediction(mean_s=[mean])


def _resolver(default: float = 10.0) -> SloResolver:
    """Create a SloResolver with a single global default (no file I/O)."""
    return SloResolver.from_dict(default_slo_s=default, slo_dict={})


# ---------------------------------------------------------------------------
# compute_before_state
# ---------------------------------------------------------------------------


class TestComputeBeforeState:

    def test_empty_cluster(self):
        """Empty cluster → zero cost and zero SLO violation."""
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=1.0,
            active_queries=[],
            billing_window_start_s=None,
        )
        cost, violation = RoutingCore.compute_before_state(
            snap,
            current_time_s=100.0,
            slo_resolver=_resolver(),
            optimize_by_amount=True,
        )
        assert cost == 0.0
        assert violation == 0.0

    def test_single_query_no_violation(self):
        """One active query well under SLO → zero violation, non-zero cost."""
        start_time = 0.0
        end_time = 5.0
        cost_per_second = 2.0

        q = _q("a", start=start_time, latency=end_time - start_time)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[q],
            billing_window_start_s=None,
        )
        cost, violation = RoutingCore.compute_before_state(
            snap,
            current_time_s=end_time + 5.0,
            slo_resolver=_resolver(10.0),
            optimize_by_amount=True,
        )
        assert violation == 0.0
        assert cost == pytest.approx(
            Billing.billed_s(query_intervals=[Interval(start_time, end_time)])
            * cost_per_second
        )

    def test_single_query_with_violation_by_amount(self):
        """Active query exceeds SLO → violation = overshoot seconds."""
        start_time = 0.0
        end_time = 12.0
        cost_per_second = 1.0
        slo_s = 10.0

        q = _q(
            "a", start=start_time, latency=end_time - start_time
        )  # SLO=10 → 2s violation
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[q],
            billing_window_start_s=None,
        )
        _, violation = RoutingCore.compute_before_state(
            snap,
            current_time_s=end_time + 5.0,
            slo_resolver=_resolver(slo_s),
            optimize_by_amount=True,
        )
        assert violation == pytest.approx(end_time - start_time - slo_s)

    def test_single_query_with_violation_binary(self):
        """Binary mode: any overshoot → violation count = 1."""
        start_time = 0.0
        end_time = 12.0
        cost_per_second = 1.0
        slo_s = 10.0

        q = _q("a", start=start_time, latency=end_time - start_time)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[q],
            billing_window_start_s=None,
        )
        _, violation = RoutingCore.compute_before_state(
            snap,
            current_time_s=end_time + 5.0,
            slo_resolver=_resolver(slo_s),
            optimize_by_amount=False,
        )
        assert violation == 1.0

    def test_billing_window_extends_cost(self):
        """An open billing window (from a recently finished query) should be
        included in the cost calculation."""
        window_start = 50.0
        current_time = 100.0
        cost_per_second = 1.0

        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[],
            billing_window_start_s=window_start,
        )
        cost, _ = RoutingCore.compute_before_state(
            snap,
            current_time_s=current_time,
            slo_resolver=_resolver(),
            optimize_by_amount=True,
        )
        assert cost == pytest.approx(
            Billing.billed_s(
                query_intervals=[Interval(window_start, current_time)]
            )
            * cost_per_second
        )


# ---------------------------------------------------------------------------
# score_placement
# ---------------------------------------------------------------------------


class TestScorePlacement:

    def _base_scenario(self):
        """Two active queries, one incoming query.  All under SLO."""
        a_pred = 5.0
        b_pred = 6.0
        slo_s = 10.0
        start_time_ab = 0.0
        now = 4.0
        cost_per_second = 1.0

        q_a = _q("a", start=start_time_ab, latency=a_pred)
        q_b = _q("b", start=start_time_ab, latency=b_pred)
        incoming = _q("new", start=now, latency=-1.0)

        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[q_a, q_b],
            billing_window_start_s=None,
        )

        # Predictions
        predictions = {
            "a": _pred(a_pred + (slo_s - a_pred) / 2),
            "b": _pred(b_pred + (slo_s - b_pred) / 2),
            "new": _pred(slo_s * 0.75),
        }

        resolver = _resolver(slo_s)
        before_cost, before_violation = RoutingCore.compute_before_state(
            snap,
            current_time_s=now,
            slo_resolver=resolver,
            optimize_by_amount=True,
        )
        return (
            incoming,
            snap,
            predictions,
            resolver,
            before_cost,
            before_violation,
        )

    def test_no_violation_placement(self):
        """All predictions under SLO → marginal SLO violation = 0."""
        incoming, snap, predictions, resolver, bc, bv = self._base_scenario()
        score = RoutingCore.score_placement(
            query=incoming,
            snapshot=snap,
            predictions=predictions,
            current_time_s=incoming.rel_start_time_s,
            slo_resolver=resolver,
            optimize_by_amount=True,
            before_cost=bc,
            before_slo_violation=bv,
        )
        assert score.marginal_slo_violation == pytest.approx(0.0)
        assert score.cluster_name == "c0"

    def test_latency_alignment(self):
        """Each query's latency in the result should match its OWN prediction,
        not a shifted neighbor's prediction — verifying the misalignment fix."""
        incoming, snap, predictions, resolver, bc, bv = self._base_scenario()
        score = RoutingCore.score_placement(
            query=incoming,
            snapshot=snap,
            predictions=predictions,
            current_time_s=incoming.rel_start_time_s,
            slo_resolver=resolver,
            optimize_by_amount=True,
            before_cost=bc,
            before_slo_violation=bv,
        )
        for q in ["a", "b", "new"]:
            assert q in score.latencies
            assert score.latencies[q] == pytest.approx(
                predictions[q].overall_mean_s()
            )

    def test_incoming_query_violation(self):
        """Incoming query prediction exceeds SLO → positive marginal violation."""
        latency_s = 15.0
        slo_s = 10.0
        arrival_time_s = 5.0
        cost_per_second = 1.0

        incoming = _q("new", start=arrival_time_s, latency=-1.0)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[],
            billing_window_start_s=None,
        )
        predictions = {"new": _pred(latency_s)}
        resolver = _resolver(slo_s)
        bc, bv = RoutingCore.compute_before_state(
            snap, incoming.rel_start_time_s, resolver, optimize_by_amount=True
        )

        score = RoutingCore.score_placement(
            incoming,
            snap,
            predictions,
            incoming.rel_start_time_s,
            resolver,
            True,
            bc,
            bv,
        )
        assert score.marginal_slo_violation == pytest.approx(latency_s - slo_s)
        assert score.latencies["new"] == pytest.approx(latency_s)

    def test_marginal_cost_positive(self):
        """Adding a query should increase cost (new billing interval)."""
        latency_s = 15.0
        slo_s = 10.0
        arrival_time_s = 5.0
        cost_per_second = 1.0

        incoming = _q("new", start=arrival_time_s, latency=-1.0)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[],
            billing_window_start_s=None,
        )
        predictions = {"new": _pred(latency_s)}
        resolver = _resolver(slo_s)
        bc, bv = RoutingCore.compute_before_state(
            snap, incoming.rel_start_time_s, resolver, optimize_by_amount=True
        )

        score = RoutingCore.score_placement(
            incoming,
            snap,
            predictions,
            incoming.rel_start_time_s,
            resolver,
            True,
            bc,
            bv,
        )

        assert score.marginal_cost == pytest.approx(
            cost_per_second
            * Billing.billed_s(
                query_intervals=[
                    Interval(arrival_time_s, arrival_time_s + latency_s)
                ]
            )
        )

    def test_binary_slo_mode(self):
        """In binary mode, violation counts are integers (0 or 1 per query)."""
        latency_s = 15.0
        slo_s = 10.0
        arrival_time_s = 5.0
        cost_per_second = 1.0

        incoming = _q("new", start=arrival_time_s, latency=-1.0)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[],
            billing_window_start_s=None,
        )
        predictions = {"new": _pred(latency_s)}
        resolver = _resolver(slo_s)
        bc, bv = RoutingCore.compute_before_state(
            snap, incoming.rel_start_time_s, resolver, optimize_by_amount=True
        )

        score = RoutingCore.score_placement(
            incoming,
            snap,
            predictions,
            incoming.rel_start_time_s,
            resolver,
            False,
            bc,
            bv,
        )
        assert score.marginal_slo_violation == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pick_best
# ---------------------------------------------------------------------------


class TestPickBest:

    def test_single_candidate(self):
        """Single candidate is returned."""
        s = PlacementScore("c0", 1.0, 50.0, {})
        assert RoutingCore.pick_best([s]) is s

    def test_lower_violation_wins(self):
        """Lower marginal SLO violation beats lower cost."""
        a = PlacementScore("a", 0.0, 100.0, {})
        b = PlacementScore("b", 5.0, 1.0, {})
        assert RoutingCore.pick_best([a, b]).cluster_name == "a"
        assert RoutingCore.pick_best([b, a]).cluster_name == "a"

    def test_equal_violation_lower_cost_wins(self):
        """When SLO violations are within tolerance, lower cost wins."""
        a = PlacementScore("a", 1.0, 100.0, {})
        b = PlacementScore("b", 1.0, 50.0, {})
        assert RoutingCore.pick_best([a, b]).cluster_name == "b"

    def test_tolerance_ties_violation(self):
        """Violations within tolerance are treated as equal → cost decides."""
        tolerance = 1e-4
        a = PlacementScore("a", 1.0, 100.0, {})
        b = PlacementScore("b", 1.0 + tolerance / 2, 50.0, {})
        best = RoutingCore.pick_best([a, b], tolerance=tolerance)
        assert best.cluster_name == "b"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            RoutingCore.pick_best([])


# ---------------------------------------------------------------------------
# _slo_cmp_with_tolerance
# ---------------------------------------------------------------------------


class TestSloCmpWithTolerance:

    def test_clearly_less(self):
        assert RoutingCore._slo_cmp_with_tolerance(1.0, 5.0) == -1

    def test_clearly_greater(self):
        assert RoutingCore._slo_cmp_with_tolerance(5.0, 1.0) == 1

    def test_within_tolerance(self):
        tolerance = 1e-4
        assert (
            RoutingCore._slo_cmp_with_tolerance(
                1.0, 1.0 + tolerance / 2, tolerance
            )
            == 0
        )

    def test_at_tolerance_boundary(self):
        tolerance = 1e-4
        assert (
            RoutingCore._slo_cmp_with_tolerance(
                1.0, 1.0 + tolerance + 1e-10, tolerance
            )
            == -1
        )

    def test_symmetric_zero(self):
        assert RoutingCore._slo_cmp_with_tolerance(3.0, 3.0) == 0


# ---------------------------------------------------------------------------
# compute_slo_headroom
# ---------------------------------------------------------------------------


class TestComputeSloHeadroom:

    def test_no_queries(self):
        """No active queries → full headroom (1.0)."""
        assert RoutingCore.compute_slo_headroom([], _resolver()) == 1.0

    def test_half_headroom(self):
        """Latency at 50% of SLO → headroom = 0.5."""
        slo_s = 10.0
        factor = 0.5
        q = _q("a", start=0.0, latency=slo_s * factor)
        h = RoutingCore.compute_slo_headroom([q], _resolver(slo_s))
        assert h == pytest.approx(1.0 - factor)

    def test_at_slo(self):
        """Latency equals SLO → headroom = 0."""
        slo_s = 10.0
        q = _q("a", start=0.0, latency=slo_s)
        h = RoutingCore.compute_slo_headroom([q], _resolver(slo_s))
        assert h == pytest.approx(0.0)

    def test_violated(self):
        """Latency exceeds SLO → headroom < 0."""
        slo_s = 10.0
        factor = 1.5
        q = _q("a", start=0.0, latency=slo_s * factor)
        h = RoutingCore.compute_slo_headroom([q], _resolver(slo_s))
        assert h < 0.0
        assert h == pytest.approx(1.0 - factor)

    def test_min_across_queries(self):
        """Headroom is the minimum across all queries."""
        slo_s = 10.0
        small_factor = 0.2
        large_factor = 0.8
        q1 = _q("a", start=0.0, latency=slo_s * small_factor)
        q2 = _q("b", start=0.0, latency=slo_s * large_factor)
        h = RoutingCore.compute_slo_headroom([q1, q2], _resolver(slo_s))
        assert h == pytest.approx(1.0 - large_factor)


