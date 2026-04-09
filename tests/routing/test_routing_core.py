"""
Tests for :mod:`autoslo.routing.routing_core`.

Covers:
- compute_before_state (empty/populated clusters, with/without billing window)
- score_placement (alignment correctness, SLO violations, billing cost)
- pick_best (lexicographic selection, tolerance handling)
- _slo_cmp_with_tolerance (comparison edge cases)
"""

from __future__ import annotations

import pytest
from intervaltree import Interval

from autoslo.models.model_prediction import ModelPrediction
from autoslo.routing.managed_cluster_pool import ClusterSnapshot
from autoslo.routing.routing_core import PlacementScore, RoutingCore
from autoslo.slo.slo_objective import SloMetric
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query, QueryTextId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(
    query_id: str,
    template: str = "ext_tpcds1000#1#0",
    start: float = 0.0,
) -> Query:
    """Create a minimal Query for testing."""
    return Query(
        query_id=query_id,
        query_text_id=QueryTextId(template),
        rel_start_time_s=start,
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
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies={},
        )
        assert cost == 0.0
        assert violation == 0.0

    def test_single_query_no_violation(self):
        """One active query well under SLO → zero violation, non-zero cost."""
        start_time = 0.0
        end_time = 5.0
        cost_per_second = 2.0

        q = _q("a", start=start_time)
        latencies = {"a": end_time - start_time}
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
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies=latencies,
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

        q = _q("a", start=start_time)  # SLO=10 → 2s violation
        latencies = {"a": end_time - start_time}
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
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies=latencies,
        )
        assert violation == pytest.approx(end_time - start_time - slo_s)

    def test_single_query_with_violation_binary(self):
        """Binary mode: any overshoot → violation count = 1."""
        start_time = 0.0
        end_time = 12.0
        cost_per_second = 1.0
        slo_s = 10.0

        q = _q("a", start=start_time)
        latencies = {"a": end_time - start_time}
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
            slo_metric=SloMetric.BINARY,
            latencies=latencies,
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
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies={},
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

        q_a = _q("a", start=start_time_ab)
        q_b = _q("b", start=start_time_ab)
        incoming = _q("new", start=now)
        latencies = {"a": a_pred, "b": b_pred}

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
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies=latencies,
        )
        return (
            incoming,
            snap,
            predictions,
            resolver,
            before_cost,
            before_violation,
            latencies,
        )

    def test_no_violation_placement(self):
        """All predictions under SLO → marginal SLO violation = 0."""
        incoming, snap, predictions, resolver, bc, bv, latencies = (
            self._base_scenario()
        )
        score = RoutingCore.score_placement(
            query=incoming,
            snapshot=snap,
            predictions=predictions,
            current_time_s=incoming.rel_start_time_s,
            slo_resolver=resolver,
            slo_metric=SloMetric.ABSOLUTE_S,
            before_cost=bc,
            before_slo_violation=bv,
            current_latencies=latencies,
        )
        assert score.marginal_slo_violation == pytest.approx(0.0)
        assert score.cluster_name == "c0"

    def test_latency_alignment(self):
        """Each query's latency in the result should match its OWN prediction,
        not a shifted neighbor's prediction — verifying the misalignment fix."""
        incoming, snap, predictions, resolver, bc, bv, latencies = (
            self._base_scenario()
        )
        score = RoutingCore.score_placement(
            query=incoming,
            snapshot=snap,
            predictions=predictions,
            current_time_s=incoming.rel_start_time_s,
            slo_resolver=resolver,
            slo_metric=SloMetric.ABSOLUTE_S,
            before_cost=bc,
            before_slo_violation=bv,
            current_latencies=latencies,
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

        incoming = _q("new", start=arrival_time_s)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[],
            billing_window_start_s=None,
        )
        predictions = {"new": _pred(latency_s)}
        resolver = _resolver(slo_s)
        bc, bv = RoutingCore.compute_before_state(
            snap,
            incoming.rel_start_time_s,
            resolver,
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies={},
        )

        score = RoutingCore.score_placement(
            incoming,
            snap,
            predictions,
            incoming.rel_start_time_s,
            resolver,
            SloMetric.ABSOLUTE_S,
            bc,
            bv,
            current_latencies={},
        )
        assert score.marginal_slo_violation == pytest.approx(latency_s - slo_s)
        assert score.latencies["new"] == pytest.approx(latency_s)

    def test_marginal_cost_positive(self):
        """Adding a query should increase cost (new billing interval)."""
        latency_s = 15.0
        slo_s = 10.0
        arrival_time_s = 5.0
        cost_per_second = 1.0

        incoming = _q("new", start=arrival_time_s)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[],
            billing_window_start_s=None,
        )
        predictions = {"new": _pred(latency_s)}
        resolver = _resolver(slo_s)
        bc, bv = RoutingCore.compute_before_state(
            snap,
            incoming.rel_start_time_s,
            resolver,
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies={},
        )

        score = RoutingCore.score_placement(
            incoming,
            snap,
            predictions,
            incoming.rel_start_time_s,
            resolver,
            SloMetric.ABSOLUTE_S,
            bc,
            bv,
            current_latencies={},
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

        incoming = _q("new", start=arrival_time_s)
        snap = ClusterSnapshot(
            cluster_name="c0",
            cost_per_second=cost_per_second,
            active_queries=[],
            billing_window_start_s=None,
        )
        predictions = {"new": _pred(latency_s)}
        resolver = _resolver(slo_s)
        bc, bv = RoutingCore.compute_before_state(
            snap,
            incoming.rel_start_time_s,
            resolver,
            slo_metric=SloMetric.ABSOLUTE_S,
            latencies={},
        )

        score = RoutingCore.score_placement(
            incoming,
            snap,
            predictions,
            incoming.rel_start_time_s,
            resolver,
            SloMetric.BINARY,
            bc,
            bv,
            current_latencies={},
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
