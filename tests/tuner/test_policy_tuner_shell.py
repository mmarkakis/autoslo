"""Tests for the PolicyTuner shell and shared types."""

from pathlib import Path

import pytest

from autoslo.tuner.config import TunerConfig
from autoslo.tuner.policy_tuner import PolicyTuner
from autoslo.tuner.tuner_utils import (
    AggregatedMetrics,
    PhaseResult,
    ScenarioResult,
    SloObjective,
    aggregate,
    compute_pareto_front,
    is_feasible,
    primary_violation,
    threshold_aware_select,
)


# ---------------------------------------------------------------------------
# PolicyTuner shell
# ---------------------------------------------------------------------------


class TestPolicyTunerShell:
    @pytest.fixture()
    def tuner(self, tmp_path: Path) -> PolicyTuner:
        cfg = TunerConfig()
        initial = {"basic_config": {"schema": "test"}}
        return PolicyTuner(initial, cfg, run_dir=tmp_path / "run")

    def test_run_dir_created(self, tuner: PolicyTuner):
        assert tuner.run_dir.is_dir()

    def test_configs_persisted(self, tuner: PolicyTuner):
        assert (tuner.run_dir / "initial_config.yml").exists()
        assert (tuner.run_dir / "tuner_config.yml").exists()

    def test_run_id_prefix(self, tuner: PolicyTuner):
        assert tuner.run_id.startswith("tuner_")


# ---------------------------------------------------------------------------
# Shared types: aggregate
# ---------------------------------------------------------------------------


class TestAggregate:
    def _results(self, violations: list[float], costs: list[float]) -> list[ScenarioResult]:
        return [
            ScenarioResult(
                scenario_idx=i,
                violation_rate=v,
                violation_amount_s=0.0,
                violation_relative_mean=0.0,
                total_cost=c,
                num_queries=100,
                out_dir=Path("/tmp"),
            )
            for i, (v, c) in enumerate(zip(violations, costs))
        ]

    def test_mean(self):
        results = self._results([0.1, 0.2, 0.3], [10.0, 20.0, 30.0])
        agg = aggregate(results, "mean")
        assert isinstance(agg, AggregatedMetrics)
        assert abs(agg.violation_rate - 0.2) < 1e-9
        assert abs(agg.cost - 20.0) < 1e-9

    def test_p90(self):
        results = self._results(list(range(1, 101)), list(range(1, 101)))
        agg = aggregate(results, "p90")
        assert agg.violation_rate >= 90.0
        assert agg.cost >= 90.0

    def test_empty(self):
        agg = aggregate([], "mean")
        assert agg.violation_rate == 0.0
        assert agg.cost == 0.0

    def test_invalid_metric(self):
        results = self._results([0.1], [1.0])
        with pytest.raises(ValueError, match="Unknown aggregation metric"):
            aggregate(results, "median")

    def test_aggregate_all_three_metrics(self):
        """All 3 violation metrics are aggregated."""
        results = [
            ScenarioResult(
                scenario_idx=0, violation_rate=0.1,
                violation_amount_s=5.0, violation_relative_mean=0.02,
                total_cost=10.0, num_queries=100, out_dir=Path("/tmp"),
            ),
            ScenarioResult(
                scenario_idx=1, violation_rate=0.3,
                violation_amount_s=15.0, violation_relative_mean=0.08,
                total_cost=30.0, num_queries=100, out_dir=Path("/tmp"),
            ),
        ]
        agg = aggregate(results, "mean")
        assert abs(agg.violation_rate - 0.2) < 1e-9
        assert abs(agg.violation_amount_s - 10.0) < 1e-9
        assert abs(agg.violation_relative_mean - 0.05) < 1e-9
        assert abs(agg.cost - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


class TestParetoFront:
    def test_simple(self):
        # (violation, cost) — points 0 and 2 are Pareto-optimal
        points = [(0.1, 20.0), (0.2, 25.0), (0.3, 10.0)]
        front = compute_pareto_front(points)
        assert 0 in front
        assert 2 in front

    def test_empty(self):
        assert compute_pareto_front([]) == []

    def test_single_point(self):
        assert compute_pareto_front([(1.0, 2.0)]) == [0]

    def test_dominated(self):
        # Point 0 dominates point 1 in both objectives
        points = [(1.0, 1.0), (2.0, 2.0)]
        front = compute_pareto_front(points)
        assert front == [0]


# ---------------------------------------------------------------------------
# primary_violation
# ---------------------------------------------------------------------------


class TestPrimaryViolation:
    def _agg(self) -> AggregatedMetrics:
        return AggregatedMetrics(
            violation_rate=0.1,
            violation_amount_s=5.0,
            violation_relative_mean=0.02,
            cost=10.0,
        )

    def test_binary(self):
        assert primary_violation(self._agg(), "binary") == 0.1

    def test_absolute_s(self):
        assert primary_violation(self._agg(), "absolute_s") == 5.0

    def test_relative(self):
        assert primary_violation(self._agg(), "relative") == 0.02

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown slo_metric"):
            primary_violation(self._agg(), "unknown_metric")


# ---------------------------------------------------------------------------
# is_feasible
# ---------------------------------------------------------------------------


class TestIsFeasible:
    def test_below_threshold(self):
        assert is_feasible(0.04, 0.05) is True

    def test_at_threshold(self):
        assert is_feasible(0.05, 0.05) is True

    def test_above_threshold(self):
        assert is_feasible(0.06, 0.05) is False


# ---------------------------------------------------------------------------
# threshold_aware_select
# ---------------------------------------------------------------------------


class TestThresholdAwareSelect:
    def test_all_feasible_picks_cheapest(self):
        # Both feasible (≤0.05); pick the cheaper one (idx 1).
        candidates = [(0.03, 100.0), (0.04, 50.0)]
        assert threshold_aware_select(candidates, 0.05) == 1

    def test_none_feasible_picks_lowest_violation(self):
        # Both infeasible; pick lowest primary violation (idx 1).
        candidates = [(0.10, 50.0), (0.06, 200.0)]
        assert threshold_aware_select(candidates, 0.05) == 1

    def test_none_feasible_tiebreak_cost(self):
        # Both infeasible, same violation; pick cheapest (idx 0).
        candidates = [(0.10, 50.0), (0.10, 100.0)]
        assert threshold_aware_select(candidates, 0.05) == 0

    def test_mixed_feasible_infeasible(self):
        # Only idx 1 is feasible.
        candidates = [(0.10, 50.0), (0.04, 200.0)]
        assert threshold_aware_select(candidates, 0.05) == 1

    def test_exactly_at_threshold(self):
        # Exactly at threshold → feasible → cheapest feasible.
        candidates = [(0.05, 100.0), (0.05, 50.0)]
        assert threshold_aware_select(candidates, 0.05) == 1

    def test_single_candidate(self):
        assert threshold_aware_select([(0.10, 50.0)], 0.05) == 0


# ---------------------------------------------------------------------------
# SloObjective
# ---------------------------------------------------------------------------


class TestSloObjective:
    def test_frozen(self):
        obj = SloObjective(slo_metric="binary", slo_threshold=0.05)
        with pytest.raises(AttributeError):
            obj.slo_metric = "relative"

    def test_fields(self):
        obj = SloObjective(slo_metric="relative", slo_threshold=0.1)
        assert obj.slo_metric == "relative"
        assert obj.slo_threshold == 0.1
