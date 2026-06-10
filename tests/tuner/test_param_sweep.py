"""Tests for ParamSweep grid-search logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from autoslo.config.component_configs import (
    ParamSweepConfig,
    SloObjectiveConfig,
    WorkloadConfig,
)
from autoslo.slo.slo_objective import SloObjective
from autoslo.tuner.param_sweep import ParamSweep, build_grid
from autoslo.workload_execution.execution_result import ExecutionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario_result(
    violation_rate: float = 0.05,
    total_cost: float = 10.0,
) -> ExecutionResult:
    return ExecutionResult(
        execution_dir=Path("/tmp/fake"),
        violation_rate=violation_rate,
        violation_amount_s=0.0,
        violation_relative_mean=0.0,
        total_cost=total_cost,
        num_queries=100,
        total_rel_time_s=0.0,
        tail_fraction=1.0,
    )


def _wc(name: str) -> WorkloadConfig:
    """Minimal WorkloadConfig for tests (never read; evaluator is mocked)."""
    return WorkloadConfig(workload_name=name, workload_dir=Path("/tmp"))


def _mock_evaluator(
    results_by_call: list[list[list[ExecutionResult]]] | None = None,
) -> MagicMock:
    """Return a mock ScenarioEvaluator with canned evaluate_batch_from_overrides() results.

    Each element of *results_by_call* is one return value for a successive
    call to ``evaluate_batch_from_overrides``.  The format is
    ``[[SimulationResult, ...], ...]`` (configs × workloads).
    """
    evaluator = MagicMock()
    if results_by_call is None:
        evaluator.evaluate_batch_from_overrides.return_value = [
            [_make_scenario_result(0.10, 100.0)],
        ]
    else:
        evaluator.evaluate_batch_from_overrides.side_effect = results_by_call

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
            initial_config={"tuner_config": {"aggregation_metric": "mean"}},
            run_dir=Path("/tmp/fake"),
            phase_name="test",
            slo_objective=SloObjective(
                SloObjectiveConfig(
                    slo_metric="binary", slo_threshold=slo_threshold
                )
            ),
            agg_method="mean",
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
        evaluator = _mock_evaluator(
            [
                [[_make_scenario_result(0.05, 100.0)]],  # train
                [[_make_scenario_result(0.06, 110.0)]],  # val
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(params={"eta_crit": [0.5]}),
        )

        assert best == {"eta_crit": 0.5} | config
        # Both train and val were called.
        assert evaluator.evaluate_batch_from_overrides.call_count == 2

    def test_sweep_results_written(self, run_dir: Path, config: dict[str, Any]):
        """Verify sweep_results.json is created with train_rank in each entry."""
        evaluator = _mock_evaluator(
            [
                [[_make_scenario_result(0.05, 100.0)]],  # train
                [[_make_scenario_result(0.06, 110.0)]],  # val
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(params={"eta_crit": [0.5]}),
        )

        results_file = run_dir / "test_sweep" / "sweep_results.json"
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert data["best_grid_point"] == 0
        assert data["best_params"] == {"eta_crit": 0.5}
        assert "train_rank" in data["grid_results"][0]
        assert "is_pareto" not in data["grid_results"][0]

    def test_sweep_two_points_picks_lower_val_violation(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """With two candidates in top-k, pick the one with lower validation violation
        (when both infeasible w.r.t. slo_threshold)."""
        # Grid: eta_crit = [0.3, 0.7]
        # Point 0 (eta_crit=0.3): train 0.10 viol, $50 cost — low cost, high viol
        # Point 1 (eta_crit=0.7): train 0.03 viol, $90 cost — high cost, low viol
        # Both in top-k (k=5 ≥ 2).
        # Val for point 0: 0.12 viol, $55
        # Val for point 1: 0.04 viol, $95
        # With slo_threshold=0.01, both infeasible → lowest violation wins (point 1).
        evaluator = _mock_evaluator(
            [
                [  # train: all grid points
                    [_make_scenario_result(0.10, 50.0)],
                    [_make_scenario_result(0.03, 90.0)],
                ],
                [  # val: top-k in train-rank order (eta_crit=0.7 ranked first)
                    [
                        _make_scenario_result(0.04, 95.0)
                    ],  # eta_crit=0.7 (rank 0)
                    [
                        _make_scenario_result(0.12, 55.0)
                    ],  # eta_crit=0.3 (rank 1)
                ],
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.01)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                params={"eta_crit": [0.3, 0.7]}
            ),
        )

        assert best == {"eta_crit": 0.7} | config
        # 1 train batch + 1 val batch
        assert evaluator.evaluate_batch_from_overrides.call_count == 2

    def test_low_ranked_point_not_validated(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """The lowest-ranked grid point should not trigger a validation evaluation."""
        # Grid: a = [1, 2, 3]
        # Point 0: 0.10 viol, $100 — highest violation → ranked last
        # Point 1: 0.05 viol, $50  — mid violation → ranked 2nd
        # Point 2: 0.03 viol, $120 — lowest violation → ranked 1st
        # With val_top_k=2, only points 2 and 1 are validated; point 0 is excluded.
        evaluator = _mock_evaluator(
            [
                [  # train: all 3 grid points
                    [_make_scenario_result(0.10, 100.0)],  # rank 2 (last)
                    [_make_scenario_result(0.05, 50.0)],  # rank 1
                    [_make_scenario_result(0.03, 120.0)],  # rank 0 (best)
                ],
                [  # val: only top-2 (re-indexed 0, 1)
                    [_make_scenario_result(0.04, 125.0)],
                    [_make_scenario_result(0.06, 55.0)],
                ],
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.01)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                params={"a": [1, 2, 3]}, val_top_k=2
            ),
        )

        # 1 train batch + 1 val batch (top-2 only)
        assert evaluator.evaluate_batch_from_overrides.call_count == 2
        val_call = evaluator.evaluate_batch_from_overrides.call_args_list[1]
        assert len(val_call[1]["all_config_overrides"]) == 2

    def test_sweep_empty_ranges(self, run_dir: Path, config: dict[str, Any]):
        """Empty sweep_config → single point with empty params."""
        evaluator = _mock_evaluator(
            [
                [[_make_scenario_result(0.05, 100.0)]],  # train
                [[_make_scenario_result(0.06, 110.0)]],  # val
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(params={}),
        )

        assert best == {} | config

    def test_sweep_config_section_applied_correctly(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Grid point keys are prefixed with config_section."""
        evaluator = _mock_evaluator(
            [
                [[_make_scenario_result(0.05, 100.0)]],  # train
                [[_make_scenario_result(0.06, 110.0)]],  # val
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                params={"routing_config.weight": [0.5]}
            ),
        )

        call_overrides = evaluator.evaluate_batch_from_overrides.call_args_list[
            0
        ][1]["all_config_overrides"][0]
        assert "routing_config.weight" in call_overrides
        assert call_overrides["routing_config.weight"] == 0.5

    def test_sweep_multi_param_grid(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Multi-parameter grid produces correct number of evaluations."""
        # 2 x 3 = 6 grid points. Default val_top_k=5 → 5 validated.
        train_batch = [
            [_make_scenario_result(0.10 - i * 0.01, 50.0 + i * 10)]
            for i in range(6)
        ]
        # Provide 6 val results; only first 5 are consumed.
        val_batch = [
            [_make_scenario_result(0.11 - i * 0.01, 55.0 + i * 10)]
            for i in range(6)
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_sweep",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                params={"a": [1, 2], "b": [10, 20, 30]}
            ),
        )

        # Best should have both keys.
        assert "a" in best
        assert "b" in best


# ---------------------------------------------------------------------------
# Random search strategy
# ---------------------------------------------------------------------------


class TestRandomSweep:
    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "sweep_run"

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return {"tuner_config": {"aggregation_metric": "mean"}}

    def test_random_with_budget_smaller_than_grid(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Random search with budget=2 from a 4-element grid → 2 candidates."""
        # Grid: a=[1,2], b=[10,20] → 4 points.  budget=2 → sample 2.
        # We need train results for 2 configs × 1 workload, then val for ≤2.
        train_batch = [
            [_make_scenario_result(0.05, 50.0)],
            [_make_scenario_result(0.10, 100.0)],
        ]
        val_batch = [
            [_make_scenario_result(0.06, 55.0)],
            [_make_scenario_result(0.11, 105.0)],
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_random",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="random",
                budget=2,
                seed=42,
                params={"a": [1, 2], "b": [10, 20]},
            ),
        )

        # Train call should have exactly 2 overrides.
        train_call = evaluator.evaluate_batch_from_overrides.call_args_list[0]
        assert len(train_call[1]["all_config_overrides"]) == 2
        assert "a" in best
        assert "b" in best

    def test_random_budget_exceeds_grid_uses_full_grid(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Budget ≥ grid size → evaluates full grid."""
        train_batch = [
            [_make_scenario_result(0.05, 50.0)],
            [_make_scenario_result(0.10, 100.0)],
        ]
        val_batch = [
            [_make_scenario_result(0.06, 55.0)],
            [_make_scenario_result(0.11, 105.0)],
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_random",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="random",
                budget=100,
                seed=42,
                params={"a": [1, 2]},
            ),
        )

        train_call = evaluator.evaluate_batch_from_overrides.call_args_list[0]
        assert len(train_call[1]["all_config_overrides"]) == 2

    def test_random_seed_reproducibility(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Same seed produces same sample."""
        train_batch = [[_make_scenario_result(0.05, 50.0)]]
        val_batch = [[_make_scenario_result(0.06, 55.0)]]

        samples = []
        for _ in range(2):
            evaluator = _mock_evaluator([train_batch, val_batch])
            sweeper = ParamSweep(
                evaluator=evaluator,
                initial_config=config,
                run_dir=run_dir / f"run_{_}",
                phase_name="test_random",
                slo_objective=SloObjective(
                    SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
                ),
                agg_method="mean",
            )
            sweeper.sweep(
                train_workload_configs=[_wc("t")],
                val_workload_configs=[_wc("v")],
                param_sweep_config=ParamSweepConfig(
                    strategy="random",
                    budget=1,
                    seed=123,
                    params={"a": [1, 2, 3, 4, 5]},
                ),
            )
            call = evaluator.evaluate_batch_from_overrides.call_args_list[0]
            samples.append(call[1]["all_config_overrides"])

        assert samples[0] == samples[1]


# ---------------------------------------------------------------------------
# Coordinate descent strategy
# ---------------------------------------------------------------------------


class TestCoordinateDescentSweep:
    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "sweep_run"

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return {"tuner_config": {"aggregation_metric": "mean"}}

    def test_cd_basic(self, run_dir: Path, config: dict[str, Any]):
        """Coordinate descent with 2 params × 3 values, 1 cycle."""
        # Params: a=[1,2,3], b=[10,20,30]
        # Starting point: a=2, b=20 (middle values)
        # Cycle 1, sweep a: evaluate a=1,2,3 with b=20
        #   → 3 train evals (1 batch)
        # Then sweep b: evaluate b=10,20,30 with a=best_a
        #   → up to 3 train evals (some may be cached)
        # Then validation for Pareto-optimal points.
        #
        # We'll set up results so a=1 wins (lowest violation) and b=30
        # wins (lowest cost among feasible).
        #
        # Sweep a (b=20 fixed):
        #   a=1: viol=0.02, cost=80
        #   a=2: viol=0.05, cost=60
        #   a=3: viol=0.08, cost=40
        # → a=1 wins (feasible, cheapest feasible)
        # Wait, all are feasible (threshold=0.5), so cheapest wins → a=3
        # Let me use threshold=0.03 so only a=1 is feasible → a=1 wins.
        #
        # Sweep b (a=1 fixed):
        #   b=10: viol=0.01, cost=100
        #   b=20: already evaluated (a=1, b=20 from above), viol=0.02, cost=80
        #   b=30: viol=0.03, cost=50
        # → with threshold=0.03, all feasible → cheapest = b=30.

        # The evaluator will be called multiple times:
        # Call 0: sweep a → 3 configs
        # Call 1: sweep b → 2 new configs (b=10 and b=30; b=20 is cached)
        # Call 2: validation of Pareto-optimal points
        evaluator = _mock_evaluator(
            [
                # Sweep a (a=1,2,3 with b=20):
                [
                    [_make_scenario_result(0.02, 80.0)],  # a=1
                    [_make_scenario_result(0.05, 60.0)],  # a=2
                    [_make_scenario_result(0.08, 40.0)],  # a=3
                ],
                # Sweep b (b=10,30 with a=1; b=20 cached):
                [
                    [_make_scenario_result(0.01, 100.0)],  # b=10
                    [_make_scenario_result(0.03, 50.0)],  # b=30
                ],
                # Validation (Pareto-optimal from all 5 evaluated configs):
                [
                    [_make_scenario_result(0.02, 82.0)],
                    [_make_scenario_result(0.06, 62.0)],
                    [_make_scenario_result(0.09, 42.0)],
                    [_make_scenario_result(0.02, 102.0)],
                    [_make_scenario_result(0.04, 52.0)],
                ],
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_cd",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.03)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="coordinate_descent",
                max_cycles=1,
                params={"a": [1, 2, 3], "b": [10, 20, 30]},
            ),
        )

        # Should have evaluated 5 unique configs (3 + 2 new).
        assert (
            evaluator.evaluate_batch_from_overrides.call_count == 3
        )  # 2 train + 1 val
        assert "a" in best
        assert "b" in best

    def test_cd_converges_early(self, run_dir: Path, config: dict[str, Any]):
        """If nothing changes in a cycle, CD stops early."""
        # Single param a=[1,2,3], starting point a=2.
        # Cycle 1: evaluate a=1,2,3.  a=2 is best → no change → converge.
        evaluator = _mock_evaluator(
            [
                # Sweep a (cycle 0):
                [
                    [_make_scenario_result(0.10, 100.0)],  # a=1
                    [
                        _make_scenario_result(0.05, 50.0)
                    ],  # a=2 (cheapest → best)
                    [_make_scenario_result(0.08, 80.0)],  # a=3
                ],
                # Validation (top-k order: a=2 rank0, a=3 rank1, a=1 rank2):
                [
                    [
                        _make_scenario_result(0.06, 45.0)
                    ],  # a=2 (cheapest val → wins)
                    [_make_scenario_result(0.09, 75.0)],  # a=3
                    [_make_scenario_result(0.11, 95.0)],  # a=1
                ],
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_cd",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="coordinate_descent",
                max_cycles=5,
                params={"a": [1, 2, 3]},
            ),
        )

        # Only 1 train batch (cycle 0, sweep a) + 1 val batch.
        assert evaluator.evaluate_batch_from_overrides.call_count == 2
        assert best["a"] == 2  # middle value was best → unchanged

    def test_cd_deduplication(self, run_dir: Path, config: dict[str, Any]):
        """Configs evaluated in cycle 0 are not re-evaluated in cycle 1."""
        # Param a=[1,2,3], starting a=2.
        # Cycle 0: eval a=1,2,3 → a=1 wins.
        # Cycle 1: eval a=1,2,3 with same context → all cached → no new evals.
        # → converge (unchanged).
        evaluator = _mock_evaluator(
            [
                # Sweep a (cycle 0):
                [
                    [_make_scenario_result(0.02, 80.0)],  # a=1
                    [_make_scenario_result(0.05, 60.0)],  # a=2
                    [_make_scenario_result(0.08, 40.0)],  # a=3
                ],
                # Validation:
                [
                    [_make_scenario_result(0.03, 85.0)],
                    [_make_scenario_result(0.06, 65.0)],
                    [_make_scenario_result(0.09, 45.0)],
                ],
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_cd",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="coordinate_descent",
                max_cycles=3,
                params={"a": [1, 2, 3]},
            ),
        )

        # 1 train batch in cycle 0. Cycle 1: a=1 is still best, all 3 configs
        # are cached, no eval call. Converge. Then 1 val batch.
        assert evaluator.evaluate_batch_from_overrides.call_count == 2

    def test_cd_custom_starting_point(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Custom starting point is used instead of middle value."""
        evaluator = _mock_evaluator(
            [
                # Sweep a (starting from a=3):
                [
                    [_make_scenario_result(0.05, 50.0)],  # a=1
                    [_make_scenario_result(0.10, 100.0)],  # a=2
                    [_make_scenario_result(0.08, 80.0)],  # a=3
                ],
                # Validation:
                [
                    [_make_scenario_result(0.06, 55.0)],
                    [_make_scenario_result(0.11, 105.0)],
                    [_make_scenario_result(0.09, 85.0)],
                ],
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_cd",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="coordinate_descent",
                max_cycles=1,
                starting_point={"a": 3},
                params={"a": [1, 2, 3]},
            ),
        )

        # All 3 are feasible (threshold=0.5), cheapest wins → a=1 ($50).
        assert best["a"] == 1

    def test_cd_results_persisted(self, run_dir: Path, config: dict[str, Any]):
        """Sweep results JSON is written for coordinate descent."""
        evaluator = _mock_evaluator(
            [
                [
                    [_make_scenario_result(0.05, 50.0)],
                    [_make_scenario_result(0.10, 100.0)],
                ],
                [  # val: top-k order (a=1 cheapest → rank 0, a=2 rank 1)
                    [_make_scenario_result(0.06, 55.0)],
                    [_make_scenario_result(0.11, 105.0)],
                ],
            ]
        )

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_cd",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="coordinate_descent",
                max_cycles=1,
                params={"a": [1, 2]},
            ),
        )

        results_file = run_dir / "test_cd" / "sweep_results.json"
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert "best_grid_point" in data
        assert "grid_results" in data


# ---------------------------------------------------------------------------
# Unknown strategy
# ---------------------------------------------------------------------------


class TestUnknownStrategy:
    def test_unknown_strategy_raises(self, tmp_path: Path):
        evaluator = _mock_evaluator()
        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config={"tuner_config": {"aggregation_metric": "mean"}},
            run_dir=tmp_path,
            phase_name="test",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )
        with pytest.raises(ValueError, match="Unknown sweep strategy"):
            sweeper.sweep(
                train_workload_configs=[_wc("t")],
                val_workload_configs=[_wc("v")],
                param_sweep_config=ParamSweepConfig(
                    strategy="nonexistent", params={"a": [1]}
                ),
            )


# ---------------------------------------------------------------------------
# Top-k validation behaviour
# ---------------------------------------------------------------------------


class TestTopKValidation:
    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "sweep_run"

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return {"tuner_config": {"aggregation_metric": "mean"}}

    def test_top_k_limits_validation_calls(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """val_top_k=2 on a 5-point grid → validation call has exactly 2 overrides."""
        train_batch = [
            [_make_scenario_result(0.10 - i * 0.01, 50.0 + i * 5)]
            for i in range(5)
        ]
        val_batch = [
            [_make_scenario_result(0.09 - i * 0.01, 52.0 + i * 5)]
            for i in range(2)
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_topk",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                params={"a": [1, 2, 3, 4, 5]}, val_top_k=2
            ),
        )

        val_call = evaluator.evaluate_batch_from_overrides.call_args_list[1]
        assert len(val_call[1]["all_config_overrides"]) == 2

    def test_top_k_ranks_by_slo_objective(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Feasible candidate is ranked first and included in top-k."""
        # 4 candidates: 1 feasible (violation ≤ threshold=0.05), 3 infeasible.
        # val_top_k=2 → feasible candidate + the least-infeasible one validated.
        train_batch = [
            [_make_scenario_result(0.20, 10.0)],  # infeasible, worst viol
            [_make_scenario_result(0.03, 90.0)],  # feasible (≤0.05)
            [_make_scenario_result(0.15, 20.0)],  # infeasible
            [_make_scenario_result(0.10, 30.0)],  # infeasible
        ]
        val_batch = [
            [_make_scenario_result(0.04, 92.0)],
            [_make_scenario_result(0.11, 32.0)],
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_topk",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.05)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                params={"a": [1, 2, 3, 4]}, val_top_k=2
            ),
        )

        # The feasible candidate (a=2, idx=1) must appear in the val overrides.
        val_call = evaluator.evaluate_batch_from_overrides.call_args_list[1]
        val_overrides = val_call[1]["all_config_overrides"]
        assert len(val_overrides) == 2
        assert {"a": 2} in val_overrides

    def test_top_k_greater_than_grid_validates_all(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """val_top_k larger than grid size → all candidates validated."""
        train_batch = [
            [_make_scenario_result(0.05, 50.0)],
            [_make_scenario_result(0.08, 40.0)],
            [_make_scenario_result(0.03, 70.0)],
        ]
        val_batch = [
            [_make_scenario_result(0.06, 52.0)],
            [_make_scenario_result(0.09, 42.0)],
            [_make_scenario_result(0.04, 72.0)],
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_topk",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                params={"a": [1, 2, 3]}, val_top_k=100
            ),
        )

        val_call = evaluator.evaluate_batch_from_overrides.call_args_list[1]
        assert len(val_call[1]["all_config_overrides"]) == 3

    def test_sweep_results_json_has_train_rank(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Every entry in sweep_results.json has train_rank; none has is_pareto."""
        train_batch = [
            [_make_scenario_result(0.05, 50.0)],
            [_make_scenario_result(0.08, 40.0)],
        ]
        val_batch = [
            [_make_scenario_result(0.06, 52.0)],
            [_make_scenario_result(0.09, 42.0)],
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])

        sweeper = ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name="test_topk",
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(params={"a": [1, 2]}),
        )

        results_file = run_dir / "test_topk" / "sweep_results.json"
        data = json.loads(results_file.read_text())
        for entry in data["grid_results"]:
            assert "train_rank" in entry
            assert "is_pareto" not in entry


# ---------------------------------------------------------------------------
# Adaptive batch strategy
# ---------------------------------------------------------------------------


class TestAdaptiveBatchSweep:
    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "sweep_run"

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        return {"tuner_config": {"aggregation_metric": "mean"}}

    def _sweeper(
        self,
        run_dir: Path,
        config: dict[str, Any],
        evaluator,
        phase: str = "test_ab",
    ) -> ParamSweep:
        return ParamSweep(
            evaluator=evaluator,
            initial_config=config,
            run_dir=run_dir,
            phase_name=phase,
            slo_objective=SloObjective(
                SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
            ),
            agg_method="mean",
        )

    def test_degenerate_param_raises(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Fewer than 2 distinct values per param → ValueError."""
        sweeper = self._sweeper(run_dir, config, _mock_evaluator())
        with pytest.raises(ValueError, match="\u22652 distinct values"):
            sweeper.sweep(
                train_workload_configs=[_wc("t")],
                val_workload_configs=[_wc("v")],
                param_sweep_config=ParamSweepConfig(
                    strategy="adaptive_batch",
                    budget=4,
                    max_rounds=1,
                    params={"a": [5, 5, 5]},
                ),
            )

    def test_round0_evaluates_budget_configs(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """With max_rounds=0, only the LHS round 0 runs before validation."""
        budget = 4
        train_batch = [
            [_make_scenario_result(0.10 - i * 0.01, 50.0 + i * 5)]
            for i in range(budget)
        ]
        val_batch = [
            [_make_scenario_result(0.09 - i * 0.01, 52.0 + i * 5)]
            for i in range(budget)
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])
        sweeper = self._sweeper(run_dir, config, evaluator)

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="adaptive_batch",
                budget=budget,
                seed=42,
                max_rounds=0,
                params={"a": [1.0, 10.0]},
            ),
        )

        # Round 0 train + validation.
        assert evaluator.evaluate_batch_from_overrides.call_count == 2
        train_call = evaluator.evaluate_batch_from_overrides.call_args_list[0]
        assert len(train_call[1]["all_config_overrides"]) == budget

    def test_lhs_samples_within_bounds(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """All round-0 samples lie within [lo, hi]; integer params are rounded."""
        budget = 8
        lo_a, hi_a = 1.0, 10.0
        lo_b, hi_b = 100, 200  # int bounds

        train_batch = [
            [_make_scenario_result(0.05, 50.0)] for _ in range(budget)
        ]
        val_batch = [[_make_scenario_result(0.06, 55.0)] for _ in range(budget)]
        evaluator = _mock_evaluator([train_batch, val_batch])
        sweeper = self._sweeper(run_dir, config, evaluator)

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="adaptive_batch",
                budget=budget,
                seed=99,
                max_rounds=0,
                params={"a": [lo_a, hi_a], "b": [lo_b, hi_b]},
            ),
        )

        overrides = evaluator.evaluate_batch_from_overrides.call_args_list[0][
            1
        ]["all_config_overrides"]
        assert len(overrides) == budget
        for cfg in overrides:
            assert lo_a <= cfg["a"] <= hi_a
            assert lo_b <= cfg["b"] <= hi_b
            assert isinstance(cfg["b"], int)

    def test_early_halt_when_integer_space_exhausted(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Integer param space with only 2 values saturates after round 0."""
        # LHS with budget=2 on an int range [1, 2] always produces exactly
        # {a:1} and {a:2}.  In round 1, every Gaussian proposal rounds to
        # 1 or 2, both already cached → 0 new candidates < ceil(2/2)=1
        # → early halt.  Only round-0 train + val calls are made.
        budget = 2
        train_batch = [
            [_make_scenario_result(0.05, 50.0)],
            [_make_scenario_result(0.08, 40.0)],
        ]
        val_batch = [
            [_make_scenario_result(0.06, 55.0)],
            [_make_scenario_result(0.09, 45.0)],
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])
        sweeper = self._sweeper(run_dir, config, evaluator)

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="adaptive_batch",
                budget=budget,
                seed=42,
                max_rounds=5,
                params={"a": [1, 2]},
            ),
        )

        assert evaluator.evaluate_batch_from_overrides.call_count == 2

    def test_cem_runs_multiple_rounds(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """With float params and max_rounds=2, three training batches are issued."""
        budget = 3

        def _tb(n: int) -> list:
            return [
                [_make_scenario_result(0.10 - i * 0.01, 50.0 + i * 5)]
                for i in range(n)
            ]

        val_batch = [
            [_make_scenario_result(0.05, 55.0)] for _ in range(budget * 3)
        ]
        evaluator = _mock_evaluator(
            [_tb(budget), _tb(budget), _tb(budget), val_batch]
        )
        sweeper = self._sweeper(run_dir, config, evaluator)

        sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="adaptive_batch",
                budget=budget,
                seed=7,
                max_rounds=2,
                params={"x": [0.0, 100.0]},  # wide float range → no collisions
            ),
        )

        # Round 0 + round 1 + round 2 training, then validation.
        assert evaluator.evaluate_batch_from_overrides.call_count == 4

    def test_seed_reproducibility(self, run_dir: Path, config: dict[str, Any]):
        """Same seed produces identical round-0 samples."""
        budget = 4

        def _run(run_id: int) -> list[dict]:
            train_batch = [
                [_make_scenario_result(0.05, 50.0)] for _ in range(budget)
            ]
            val_batch = [
                [_make_scenario_result(0.06, 55.0)] for _ in range(budget)
            ]
            evaluator = _mock_evaluator([train_batch, val_batch])
            sweeper = ParamSweep(
                evaluator=evaluator,
                initial_config={"tuner_config": {"aggregation_metric": "mean"}},
                run_dir=run_dir / f"r{run_id}",
                phase_name="test_ab",
                slo_objective=SloObjective(
                    SloObjectiveConfig(slo_metric="binary", slo_threshold=0.5)
                ),
                agg_method="mean",
            )
            sweeper.sweep(
                train_workload_configs=[_wc("t")],
                val_workload_configs=[_wc("v")],
                param_sweep_config=ParamSweepConfig(
                    strategy="adaptive_batch",
                    budget=budget,
                    seed=42,
                    max_rounds=0,
                    params={"a": [0.0, 1.0], "b": [10.0, 20.0]},
                ),
            )
            return evaluator.evaluate_batch_from_overrides.call_args_list[0][1][
                "all_config_overrides"
            ]

        assert _run(0) == _run(1)

    def test_results_wire_through_val_and_best_selection(
        self, run_dir: Path, config: dict[str, Any]
    ):
        """Best config returned by sweep() has the swept parameter key."""
        budget = 2
        train_batch = [
            [_make_scenario_result(0.05, 50.0)],
            [_make_scenario_result(0.08, 80.0)],
        ]
        val_batch = [
            [_make_scenario_result(0.06, 55.0)],
            [_make_scenario_result(0.09, 85.0)],
        ]
        evaluator = _mock_evaluator([train_batch, val_batch])
        sweeper = self._sweeper(run_dir, config, evaluator)

        best, _, _ = sweeper.sweep(
            train_workload_configs=[_wc("t")],
            val_workload_configs=[_wc("v")],
            param_sweep_config=ParamSweepConfig(
                strategy="adaptive_batch",
                budget=budget,
                seed=42,
                max_rounds=0,
                val_top_k=2,
                params={"x": [1.0, 5.0]},
            ),
        )

        assert "x" in best
        assert 1.0 <= best["x"] <= 5.0
