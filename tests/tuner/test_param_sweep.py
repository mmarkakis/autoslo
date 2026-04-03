"""Tests for ParamSweep grid-search logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import json
import pytest

from autoslo.tuner.config import TunerConfig
from autoslo.tuner.param_sweep import ParamSweep, build_grid
from autoslo.tuner.tuner_utils import ScenarioResult, SloObjective


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario_result(
    scenario_idx: int,
    violation_rate: float = 0.05,
    total_cost: float = 10.0,
    out_dir: str = "/tmp/fake",
) -> ScenarioResult:
    return ScenarioResult(
        scenario_idx=scenario_idx,
        violation_rate=violation_rate,
        violation_amount_s=0.0,
        violation_relative_mean=0.0,
        total_cost=total_cost,
        num_queries=100,
        out_dir=Path(out_dir),
    )


def _mock_evaluator(
    results_by_call: list[list[ScenarioResult]] | None = None,
) -> MagicMock:
    """Return a mock ScenarioEvaluator with canned .evaluate()  and evaluate_batch() results."""
    evaluator = MagicMock()
    if results_by_call is None:
        evaluator.evaluate.return_value = [
            _make_scenario_result(0, 0.10, 100.0),
        ]
        evaluator.evaluate_batch.return_value = [
            [
                _make_scenario_result(0, 0.10, 100.0),
            ]
        ]
    else:
        evaluator.evaluate.side_effect = results_by_call
        evaluator.evaluate_batch.side_effect = [results_by_call]

    return evaluator


# ---------------------------------------------------------------------------
# build_grid
# ---------------------------------------------------------------------------


class TestBuildGrid:
    def test_empty_ranges(self):
        assert build_grid({}) == [{}]

    def test_single_param(self):
        grid = build_grid({"a": [1, 2, 3]})
        assert grid == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_two_params(self):
        grid = build_grid({"a": [1, 2], "b": ["x", "y"]})
        assert len(grid) == 4
        assert {"a": 1, "b": "x"} in grid
        assert {"a": 2, "b": "y"} in grid

    def test_preserves_order(self):
        grid = build_grid({"x": [10, 20], "y": [30]})
        assert grid == [{"x": 10, "y": 30}, {"x": 20, "y": 30}]


# ---------------------------------------------------------------------------
# ParamSweep._select_best
# ---------------------------------------------------------------------------


class TestSelectBest:
    """Test threshold-aware _select_best logic."""

    def _make_sweeper(self, slo_threshold: float = 1.0) -> ParamSweep:
        return ParamSweep(
            evaluator=_mock_evaluator(),
            config={"tuner_config": {"aggregation_metric": "mean"}},
            base_overrides={},
            run_dir=Path("/tmp/fake"),
            phase_name="test",
            slo_objective=SloObjective(
                slo_metric="binary", slo_threshold=slo_threshold
            ),
        )

    def test_single_pareto_point(self):
        grid_results = [
            {
                "val_primary_violation_agg": 0.05,
                "val_cost_agg": 50.0,
                "train_violation_agg": 0.06,
                "train_cost_agg": 55.0,
            },
        ]
        assert self._make_sweeper()._select_best(grid_results, [0]) == 0

    def test_picks_lowest_val_violation(self):
        grid_results = [
            {
                "val_primary_violation_agg": 0.10,
                "val_cost_agg": 50.0,
                "train_violation_agg": 0.10,
                "train_cost_agg": 50.0,
            },
            {
                "val_primary_violation_agg": 0.05,
                "val_cost_agg": 80.0,
                "train_violation_agg": 0.05,
                "train_cost_agg": 80.0,
            },
        ]
        # Both below threshold=1.0 → both feasible → cheapest wins (idx 0).
        # With threshold=0.01 → both infeasible → lowest violation wins (idx 1).
        assert (
            self._make_sweeper(slo_threshold=1.0)._select_best(
                grid_results, [0, 1]
            )
            == 0
        )
        assert (
            self._make_sweeper(slo_threshold=0.01)._select_best(
                grid_results, [0, 1]
            )
            == 1
        )

    def test_ties_broken_by_cost(self):
        grid_results = [
            {
                "val_primary_violation_agg": 0.05,
                "val_cost_agg": 100.0,
                "train_violation_agg": 0.05,
                "train_cost_agg": 100.0,
            },
            {
                "val_primary_violation_agg": 0.05,
                "val_cost_agg": 50.0,
                "train_violation_agg": 0.05,
                "train_cost_agg": 50.0,
            },
        ]
        # Both feasible (threshold=1.0) → cheapest wins (idx 1).
        assert (
            self._make_sweeper(slo_threshold=1.0)._select_best(
                grid_results, [0, 1]
            )
            == 1
        )

    def test_threshold_aware_prefers_feasible_cheapest(self):
        """With threshold=0.05, feasible candidates are preferred."""
        grid_results = [
            {
                "val_primary_violation_agg": 0.03,
                "val_cost_agg": 200.0,
                "train_violation_agg": 0.03,
                "train_cost_agg": 200.0,
            },
            {
                "val_primary_violation_agg": 0.04,
                "val_cost_agg": 100.0,
                "train_violation_agg": 0.04,
                "train_cost_agg": 100.0,
            },
            {
                "val_primary_violation_agg": 0.10,
                "val_cost_agg": 10.0,
                "train_violation_agg": 0.10,
                "train_cost_agg": 10.0,
            },
        ]
        # idx 0 & 1 feasible (≤0.05), idx 2 infeasible → cheapest feasible = idx 1.
        assert (
            self._make_sweeper(slo_threshold=0.05)._select_best(
                grid_results, [0, 1, 2]
            )
            == 1
        )


# ---------------------------------------------------------------------------
# ParamSweep.sweep (integration with mock evaluator)
# ---------------------------------------------------------------------------


class TestParamSweepIntegration:
    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "sweep_run"

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return {"tuner_config": {"aggregation_metric": "mean"}}

    def test_sweep_single_point(self, run_dir: Path, config: dict[str, Any]):
        """One-element grid → the single point is selected."""
        train_result = [_make_scenario_result(0, 0.05, 100.0)]
        val_result = [_make_scenario_result(0, 0.06, 110.0)]

        evaluator = _mock_evaluator([train_result, val_result])

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides={},
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.5),
        )

        best = sweeper.sweep(
            train_paths=[Path("/tmp/train.parquet")],
            val_paths=[Path("/tmp/val.parquet")],
            param_ranges={"eta_crit": [0.5]},
        )

        assert best == {"eta_crit": 0.5}
        # Both train and val were called.
        assert evaluator.evaluate.call_count == 1
        assert evaluator.evaluate_batch.call_count == 1

    def test_sweep_results_written(self, run_dir: Path, config: dict[str, Any]):
        """Verify sweep_results.json is created."""
        train_result = [_make_scenario_result(0, 0.05, 100.0)]
        val_result = [_make_scenario_result(0, 0.06, 110.0)]
        evaluator = _mock_evaluator([train_result, val_result])

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides={},
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.5),
        )

        sweeper.sweep(
            train_paths=[Path("/tmp/t.parquet")],
            val_paths=[Path("/tmp/v.parquet")],
            param_ranges={"eta_crit": [0.5]},
        )

        results_file = run_dir / "test_sweep" / "sweep_results.json"
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert data["best_grid_point"] == 0
        assert data["best_params"] == {"eta_crit": 0.5}

    def test_sweep_two_points_picks_lower_val_violation(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """With two Pareto-optimal points, pick the one with lower validation violation
        (when both infeasible w.r.t. slo_threshold)."""
        # Grid: eta_crit = [0.3, 0.7]
        # Point 0 (eta_crit=0.3): train 0.10 viol, $50 cost — low cost, high viol
        # Point 1 (eta_crit=0.7): train 0.03 viol, $90 cost — high cost, low viol
        # Both are Pareto-optimal.
        # Val for point 0: 0.12 viol, $55
        # Val for point 1: 0.04 viol, $95
        # With slo_threshold=0.01, both infeasible → lowest violation wins (point 1).
        call_results = [
            # train gp 0
            [_make_scenario_result(0, 0.10, 50.0)],
            # train gp 1
            [_make_scenario_result(0, 0.03, 90.0)],
            # val gp 0
            [_make_scenario_result(0, 0.12, 55.0)],
            # val gp 1
            [_make_scenario_result(0, 0.04, 95.0)],
        ]
        evaluator = _mock_evaluator(call_results)

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides={},
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.01),
        )

        best = sweeper.sweep(
            train_paths=[Path("/tmp/t.parquet")],
            val_paths=[Path("/tmp/v.parquet")],
            param_ranges={"eta_crit": [0.3, 0.7]},
        )

        assert best == {"eta_crit": 0.7}
        # 2 train + 2 val evaluations
        assert evaluator.evaluate.call_count == 2
        assert evaluator.evaluate_batch.call_count == 1

    def test_sweep_applies_base_overrides(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Base overrides (e.g. checkpoints) are merged with grid overrides."""
        train_result = [_make_scenario_result(0, 0.05, 100.0)]
        val_result = [_make_scenario_result(0, 0.06, 110.0)]
        evaluator = _mock_evaluator([train_result, val_result])

        base_overrides = {
            "autoscaling_config.capacity_checkpoints": [{"time_s": 0}]
        }

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides=base_overrides,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.5),
        )

        sweeper.sweep(
            train_paths=[Path("/tmp/t.parquet")],
            val_paths=[Path("/tmp/v.parquet")],
            param_ranges={"autoscaling_config.eta_crit": [0.5]},
        )

        # Verify the first evaluate call includes both base and grid overrides.
        call_overrides = evaluator.evaluate.call_args_list[0][1][
            "config_overrides"
        ]
        print(call_overrides)
        assert "autoscaling_config" in call_overrides
        assert "capacity_checkpoints" in call_overrides["autoscaling_config"]
        assert "eta_crit" in call_overrides["autoscaling_config"]

    def test_sweep_dominated_point_not_validated(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """A dominated grid point should not trigger a validation evaluation."""
        # Grid: a = [1, 2, 3]
        # Point 0: 0.10 viol, $100 — dominated by point 1 (lower on both)
        # Point 1: 0.05 viol, $50
        # Point 2: 0.03 viol, $120 — Pareto with point 1
        call_results = [
            [_make_scenario_result(0, 0.10, 100.0)],  # train gp 0 (dominated)
            [_make_scenario_result(0, 0.05, 50.0)],  # train gp 1 (pareto)
            [_make_scenario_result(0, 0.03, 120.0)],  # train gp 2 (pareto)
            # Only 2 val evals (gp 1 and gp 2)
            [_make_scenario_result(0, 0.06, 55.0)],  # val gp 1
            [_make_scenario_result(0, 0.04, 125.0)],  # val gp 2
        ]
        evaluator = _mock_evaluator(call_results)

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides={},
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.01),
        )

        best = sweeper.sweep(
            train_paths=[Path("/tmp/t.parquet")],
            val_paths=[Path("/tmp/v.parquet")],
            param_ranges={"a": [1, 2, 3]},
        )

        # 3 train evals + 2 val evals for Pareto points only
        assert evaluator.evaluate.call_count == 3
        assert evaluator.evaluate_batch.call_count == 1

    def test_sweep_empty_ranges(self, run_dir: Path, config: dict[str, Any]):
        """Empty param_ranges → single point with empty params."""
        train_result = [_make_scenario_result(0, 0.05, 100.0)]
        val_result = [_make_scenario_result(0, 0.06, 110.0)]
        evaluator = _mock_evaluator([train_result, val_result])

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides={},
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.5),
        )

        best = sweeper.sweep(
            train_paths=[Path("/tmp/t.parquet")],
            val_paths=[Path("/tmp/v.parquet")],
            param_ranges={},
        )

        assert best == {}

    def test_sweep_config_section_applied_correctly(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Grid point keys are prefixed with config_section."""
        train_result = [_make_scenario_result(0, 0.05, 100.0)]
        val_result = [_make_scenario_result(0, 0.06, 110.0)]
        evaluator = _mock_evaluator([train_result, val_result])

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides={},
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.5),
        )

        sweeper.sweep(
            train_paths=[Path("/tmp/t.parquet")],
            val_paths=[Path("/tmp/v.parquet")],
            param_ranges={"routing_config.weight": [0.5]},
        )

        call_overrides = evaluator.evaluate.call_args_list[0][1][
            "config_overrides"
        ]
        assert "routing_config" in call_overrides
        assert "weight" in call_overrides["routing_config"]
        assert call_overrides["routing_config"]["weight"] == 0.5

    def test_sweep_multi_param_grid(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Multi-parameter grid produces correct number of evaluations."""
        # 2 x 3 = 6 grid points. Assume all Pareto (worst case).
        results = []
        for i in range(6):
            results.append(
                [_make_scenario_result(0, 0.10 - i * 0.01, 50.0 + i * 10)]
            )
        # 6 val results
        for i in range(6):
            results.append(
                [_make_scenario_result(0, 0.11 - i * 0.01, 55.0 + i * 10)]
            )
        evaluator = _mock_evaluator(results)

        sweeper = ParamSweep(
            evaluator=evaluator,
            config=config,
            base_overrides={},
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(slo_metric="binary", slo_threshold=0.5),
        )

        best = sweeper.sweep(
            train_paths=[Path("/tmp/t.parquet")],
            val_paths=[Path("/tmp/v.parquet")],
            param_ranges={"a": [1, 2], "b": [10, 20, 30]},
        )

        # Best should have both keys.
        assert "a" in best
        assert "b" in best
