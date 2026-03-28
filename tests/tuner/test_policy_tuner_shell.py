"""Tests for the PolicyTuner shell and shared types."""

from pathlib import Path

import pytest

from autoslo.tuner.config import TunerConfig
from autoslo.tuner.policy_tuner import PolicyTuner
from autoslo.tuner.types import (
    PhaseResult,
    ScenarioResult,
    aggregate,
    compute_pareto_front,
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

    @pytest.mark.parametrize(
        "method,args",
        [
            ("build_reservoir", ([],)),
            ("sample_workloads", (Path("."),)),
            ("tune", ([],)),
        ],
    )
    def test_stubs_raise(self, tuner: PolicyTuner, method: str, args: tuple):
        with pytest.raises(NotImplementedError):
            getattr(tuner, method)(*args)


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
        v, c = aggregate(results, "mean")
        assert abs(v - 0.2) < 1e-9
        assert abs(c - 20.0) < 1e-9

    def test_p90(self):
        results = self._results(list(range(1, 101)), list(range(1, 101)))
        v, c = aggregate(results, "p90")
        assert v >= 90.0
        assert c >= 90.0

    def test_empty(self):
        assert aggregate([], "mean") == (0.0, 0.0)

    def test_invalid_metric(self):
        results = self._results([0.1], [1.0])
        with pytest.raises(ValueError, match="Unknown aggregation metric"):
            aggregate(results, "median")


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
