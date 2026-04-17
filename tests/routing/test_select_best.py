"""Tests for QueryRouter.select_best with global absolute (violation, cost) tuples."""

import pytest

from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.routing.query_router import QueryRouter, QueryRouterPolicy


def _make_router(
    threshold: float = 0.05,
    metric: SloMetric = SloMetric.RELATIVE,
) -> QueryRouter:
    return QueryRouter(
        slo_resolver=SloResolver(default_slo_s=10.0),
        slo_objective=SloObjective(slo_metric=metric, slo_threshold=threshold),
    )


class TestSelectBestAbsolute:
    """select_best should compare global absolute (violation, cost) tuples."""

    def test_both_feasible_picks_cheaper(self):
        """When both clusters are under the SLO threshold, pick the
        cheaper one even if its violation is slightly higher."""
        router = _make_router(threshold=0.10)
        result = router.select_best({
            "big": (0.05, 100.0),   # lower violation, more expensive
            "small": (0.08, 20.0),  # higher violation, cheaper
        })
        assert result == "small"

    def test_both_infeasible_picks_lower_violation(self):
        """When both clusters exceed the threshold,
        pick the one with lower violation regardless of cost."""
        router = _make_router(threshold=0.05)
        result = router.select_best({
            "A": (0.20, 10.0),   # worse violation, cheap
            "B": (0.10, 100.0),  # better violation, expensive
        })
        assert result == "B"

    def test_one_feasible_one_not_picks_feasible(self):
        """If only one meets the SLO, pick it."""
        router = _make_router(threshold=0.10)
        result = router.select_best({
            "good": (0.05, 200.0),
            "bad": (0.15, 10.0),
        })
        assert result == "good"

    def test_single_cluster_returns_it(self):
        router = _make_router()
        assert router.select_best({"only": (0.5, 50.0)}) == "only"

    def test_no_size_bias(self):
        """Routing should not systematically favour any cluster
        regardless of how many queries it already has. With absolute
        values the comparison is size-agnostic."""
        router = _make_router(threshold=0.10)
        # Scenario: cluster A has global viol 0.08, cluster B has 0.09,
        # both feasible → cheaper wins.
        result = router.select_best({
            "A": (0.08, 50.0),
            "B": (0.09, 30.0),
        })
        assert result == "B"  # cheaper and both feasible

    def test_equal_violation_picks_cheaper(self):
        """Tie in violation (within tolerance) → cheaper wins."""
        router = _make_router(threshold=0.10)
        result = router.select_best({
            "X": (0.05, 100.0),
            "Y": (0.05, 40.0),
        })
        assert result == "Y"

    def test_round_robin_ignores_metrics(self):
        router = _make_router()
        router._routing_policy = QueryRouterPolicy.ROUND_ROBIN
        choices = [
            router.select_best({"A": (0.5, 10.0), "B": (0.1, 5.0)})
            for _ in range(4)
        ]
        # round-robin cycles deterministically through sorted keys
        assert choices == ["A", "B", "A", "B"]
