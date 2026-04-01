"""Tests for CheckpointOptimizer and find_violation_windows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.checkpoint_optimizer import (
    CheckpointOptimizer,
    ViolationWindow,
    _checkpoints_to_config,
    _get_allowed_rpu_sizes,
    _get_spin_up_delay,
    find_violation_windows,
)
from autoslo.tuner.config import TunerConfig
from autoslo.tuner.types import ScenarioResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario_result(
    tmp_path: Path,
    scenario_idx: int,
    latencies: list[float],
    timestamps: list[float] | None = None,
    cost: float = 1.0,
) -> ScenarioResult:
    """Create a ScenarioResult with a synthetic structured_log on disk."""
    out_dir = tmp_path / f"scenario_{scenario_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(latencies)
    if timestamps is None:
        timestamps = [float(i * 10) for i in range(n)]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "source": ["simulator"] * n,
            "event_type": ["completion"] * n,
            "query_id": [f"id_{i}" for i in range(n)],
            "query_text_id": [f"q{i:03d}" for i in range(n)],
            "cluster_name": ["cluster_0"] * n,
            "latency_s": latencies,
        }
    )
    df.to_parquet(out_dir / "structured_log.parquet", index=False)

    # Write billing file.
    billing = {"cluster_0": {"total_billed_cost": cost}}
    with open(out_dir / "billing_interval_analysis.yml", "w") as f:
        yaml.dump(billing, f)

    # Compute violation stats for the result object.
    slo_s = 10.0
    violations = [l > slo_s for l in latencies]
    vr = sum(violations) / n if n else 0.0

    return ScenarioResult(
        scenario_idx=scenario_idx,
        violation_rate=vr,
        violation_amount_s=sum(max(0, l - slo_s) for l in latencies),
        violation_relative_mean=0.0,
        total_cost=cost,
        num_queries=n,
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    def test_checkpoints_to_config_empty(self):
        result = _checkpoints_to_config([])
        assert result == {"autoscaling_config.capacity_checkpoints": []}

    def test_checkpoints_to_config(self):
        cps = [
            CapacityCheckpoint(time_s=60.0, min_rpus=(8,)),
            CapacityCheckpoint(time_s=300.0, min_rpus=(16, 32)),
        ]
        result = _checkpoints_to_config(cps)
        expected = [
            {"time_s": 60.0, "min_rpus": [8]},
            {"time_s": 300.0, "min_rpus": [16, 32]},
        ]
        assert result["autoscaling_config.capacity_checkpoints"] == expected

    def test_get_spin_up_delay_from_config(self):
        cfg = {"managed_cluster_pool_config": {"spin_up_delay_s": 90.0}}
        assert _get_spin_up_delay(cfg) == 90.0

    def test_get_spin_up_delay_default(self):
        assert _get_spin_up_delay({}) == 120.0

    def test_get_allowed_rpu_sizes_from_config(self):
        cfg = {"managed_cluster_pool_config": {"allowed_rpu_sizes": [4, 8]}}
        assert _get_allowed_rpu_sizes(cfg) == [4, 8]

    def test_get_allowed_rpu_sizes_default(self):
        sizes = _get_allowed_rpu_sizes({})
        assert 4 in sizes
        assert 8 in sizes


# ---------------------------------------------------------------------------
# find_violation_windows
# ---------------------------------------------------------------------------


class TestFindViolationWindows:
    def test_no_results(self):
        assert find_violation_windows([], window_s=300.0, slo_s=10.0) == []

    def test_single_scenario_no_violations(self, tmp_path: Path):
        r = _make_scenario_result(
            tmp_path,
            scenario_idx=0,
            latencies=[5.0, 6.0, 7.0],
            timestamps=[10.0, 20.0, 30.0],
        )
        windows = find_violation_windows([r], window_s=300.0, slo_s=10.0)
        assert len(windows) == 1
        assert windows[0].violation_rate == 0.0
        assert windows[0].num_violations == 0
        assert windows[0].num_queries == 3

    def test_single_scenario_with_violations(self, tmp_path: Path):
        # SLO = 10; latencies 5, 15, 20 → 2 violations out of 3
        r = _make_scenario_result(
            tmp_path,
            scenario_idx=0,
            latencies=[5.0, 15.0, 20.0],
            timestamps=[10.0, 20.0, 30.0],
        )
        windows = find_violation_windows([r], window_s=300.0, slo_s=10.0)
        assert len(windows) == 1
        assert windows[0].violation_rate == pytest.approx(2.0 / 3.0)
        assert windows[0].num_violations == 2
        assert windows[0].num_queries == 3

    def test_multiple_windows(self, tmp_path: Path):
        # Queries at t=10 (window 0-300) and t=350 (window 300-600)
        r = _make_scenario_result(
            tmp_path,
            scenario_idx=0,
            latencies=[15.0, 5.0],
            timestamps=[10.0, 350.0],
        )
        windows = find_violation_windows([r], window_s=300.0, slo_s=10.0)
        assert len(windows) == 2
        # First window: 1 violation out of 1
        assert windows[0].start_s == pytest.approx(0.0)
        assert windows[0].violation_rate == pytest.approx(1.0)
        # Second window: 0 violations out of 1
        assert windows[1].start_s == pytest.approx(300.0)
        assert windows[1].violation_rate == pytest.approx(0.0)

    def test_average_across_scenarios(self, tmp_path: Path):
        # Scenario 0: violation in window 0
        r0 = _make_scenario_result(
            tmp_path,
            scenario_idx=0,
            latencies=[15.0],
            timestamps=[10.0],
        )
        # Scenario 1: no violation in window 0
        r1 = _make_scenario_result(
            tmp_path,
            scenario_idx=1,
            latencies=[5.0],
            timestamps=[10.0],
        )
        windows = find_violation_windows([r0, r1], window_s=300.0, slo_s=10.0)
        assert len(windows) == 1
        # Average violation rate: (1.0 + 0.0) / 2 = 0.5
        assert windows[0].violation_rate == pytest.approx(0.5)

    def test_missing_log_file(self, tmp_path: Path):
        # ScenarioResult pointing to empty dir (no log file)
        r = ScenarioResult(
            scenario_idx=0,
            violation_rate=0.0,
            violation_amount_s=0.0,
            violation_relative_mean=0.0,
            total_cost=0.0,
            num_queries=0,
            out_dir=tmp_path / "nonexistent",
        )
        windows = find_violation_windows([r], window_s=300.0, slo_s=10.0)
        assert windows == []

    def test_windows_sorted_chronologically(self, tmp_path: Path):
        r = _make_scenario_result(
            tmp_path,
            scenario_idx=0,
            latencies=[15.0, 15.0, 15.0],
            timestamps=[610.0, 10.0, 310.0],
        )
        windows = find_violation_windows([r], window_s=300.0, slo_s=10.0)
        starts = [w.start_s for w in windows]
        assert starts == sorted(starts)

    def test_per_query_slo_dict(self, tmp_path: Path):
        """Per-query SLO overrides should override the default threshold."""
        # Template 001 has a tight SLO of 3s; template 002 keeps default 10s.
        # Latencies: q001=5s (violates 3s SLO), q002=5s (within 10s SLO).
        out_dir = tmp_path / "scenario_slo"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "timestamp": [10.0, 20.0],
                "source": ["simulator", "simulator"],
                "event_type": ["completion", "completion"],
                "query_id": ["id_0", "id_1"],
                "query_text_id": [
                    "ext_tpcds1000#001#001",
                    "ext_tpcds1000#002#001",
                ],
                "cluster_name": ["c0", "c0"],
                "latency_s": [5.0, 5.0],
            }
        )
        df.to_parquet(out_dir / "structured_log.parquet", index=False)

        r = ScenarioResult(
            scenario_idx=0,
            violation_rate=0.5,
            violation_amount_s=2.0,
            violation_relative_mean=0.0,
            total_cost=1.0,
            num_queries=2,
            out_dir=out_dir,
        )

        # Without per-query SLO: default=10s → 0 violations.
        windows_no_dict = find_violation_windows(
            [r], window_s=300.0, slo_s=10.0, slo_dict=None
        )
        assert windows_no_dict[0].num_violations == 0

        # With per-query SLO: template 001 → 3s → 1 violation.
        windows_with_dict = find_violation_windows(
            [r], window_s=300.0, slo_s=10.0, slo_dict={"001": 3.0}
        )
        assert windows_with_dict[0].num_violations == 1
        assert windows_with_dict[0].violation_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# CheckpointOptimizer — greedy loop
# ---------------------------------------------------------------------------


class TestCheckpointOptimizer:
    """Test the optimizer with a mock evaluator."""

    @pytest.fixture()
    def setup(self, tmp_path: Path):
        """Common setup: mock evaluator, tuner config, file paths."""
        tuner_config = TunerConfig(
            checkpoint_budget=3,
            checkpoint_epsilon=0.01,
            sliding_window_s=300.0,
            violation_threshold=0.1,
            aggregation_metric="mean",
        )
        initial_config: dict[str, Any] = {
            "slo_config": {"slo_s": 10.0, "slo_threshold": 0.5},
            "managed_cluster_pool_config": {
                "allowed_rpu_sizes": [8, 16],
                "spin_up_delay_s": 60.0,
            },
        }

        # Create dummy workload parquet files.
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()
        train_paths = []
        val_paths = []
        for i in range(2):
            p = train_dir / f"t_{i:03d}.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(p)
            train_paths.append(p)
            p = val_dir / f"v_{i:03d}.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(p)
            val_paths.append(p)

        return {
            "tuner_config": tuner_config,
            "initial_config": initial_config,
            "tmp_path": tmp_path,
            "train_paths": train_paths,
            "val_paths": val_paths,
        }

    def _make_mock_evaluator(self) -> MagicMock:
        return MagicMock(spec=["evaluate"])

    def test_no_violating_windows_returns_empty(self, setup):
        """When no window exceeds the threshold, returns empty checkpoints."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]

        # Build scenario results where latencies are all below SLO
        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            results = []
            for i in range(len(workload_paths)):
                r = _make_scenario_result(
                    tmp_path / f"eval_{grid_point}",
                    scenario_idx=i,
                    latencies=[5.0, 6.0, 7.0],
                    timestamps=[10.0, 20.0, 30.0],
                )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=setup["tuner_config"],
            initial_config=setup["initial_config"],
            run_dir=tmp_path / "run",
        )

        checkpoints = optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=0.05,
        )
        assert checkpoints == []

    def test_budget_cap(self, setup):
        """Optimizer stops after checkpoint_budget rounds."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]
        call_count = {"n": 0}

        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            call_count["n"] += 1
            results = []
            for i in range(len(workload_paths)):
                # High violation rate so the loop always finds a window.
                r = _make_scenario_result(
                    tmp_path / f"eval_{call_count['n']}_{grid_point}",
                    scenario_idx=i,
                    latencies=[15.0, 15.0, 15.0],
                    timestamps=[10.0, 20.0, 30.0],
                )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        # Set epsilon very negative so early stopping never triggers.
        tuner_config = replace(
            setup["tuner_config"],
            checkpoint_budget=2,
            checkpoint_epsilon=-999.0,
        )

        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=tuner_config,
            initial_config=setup["initial_config"],
            run_dir=tmp_path / "run",
        )

        checkpoints = optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=1.0,
        )
        assert len(checkpoints) == 2

    def test_early_stopping_epsilon(self, setup):
        """Optimizer stops when validation improvement is below epsilon."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]
        call_count = {"n": 0}

        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            call_count["n"] += 1
            results = []
            for i in range(len(workload_paths)):
                # Round 0 base: high violation → finds a window.
                # RPU trials & val: same high violation → no improvement.
                r = _make_scenario_result(
                    tmp_path / f"eval_{call_count['n']}_{grid_point}",
                    scenario_idx=i,
                    latencies=[15.0, 15.0],
                    timestamps=[10.0, 20.0],
                )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        tuner_config = replace(
            setup["tuner_config"],
            checkpoint_budget=5,
            checkpoint_epsilon=0.01,
        )

        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=tuner_config,
            initial_config=setup["initial_config"],
            run_dir=tmp_path / "run",
        )

        # baseline_val_violation matches what the evaluator returns (1.0) →
        # improvement = 0 < epsilon → should stop after round 0.
        checkpoints = optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=1.0,
        )
        assert len(checkpoints) == 0

    def test_checkpoint_time_accounts_for_spin_up(self, setup):
        """The checkpoint time_s is window_start - spin_up_delay."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]
        call_count = {"n": 0}

        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            call_count["n"] += 1
            is_val = "val" in str(grid_point)
            results = []
            for i in range(len(workload_paths)):
                if is_val:
                    # Val: report low violation so improvement > epsilon.
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[5.0, 5.0],
                        timestamps=[310.0, 320.0],
                    )
                else:
                    # Train: high violation in the window starting at 300.
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[15.0, 15.0],
                        timestamps=[310.0, 320.0],
                    )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        tuner_config = replace(
            setup["tuner_config"],
            checkpoint_budget=1,
            checkpoint_epsilon=-999.0,
        )
        initial_config = dict(setup["initial_config"])
        initial_config["managed_cluster_pool_config"]["spin_up_delay_s"] = 60.0

        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=tuner_config,
            initial_config=initial_config,
            run_dir=tmp_path / "run",
        )

        checkpoints = optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=1.0,
        )
        assert len(checkpoints) == 1
        # Window starts at 300, spin_up_delay = 60 → checkpoint at 240.
        assert checkpoints[0].time_s == pytest.approx(240.0)

    def test_checkpoint_time_floored_at_zero(self, setup):
        """The checkpoint time_s cannot go negative."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]
        call_count = {"n": 0}

        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            call_count["n"] += 1
            is_val = "val" in str(grid_point)
            results = []
            for i in range(len(workload_paths)):
                if is_val:
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[5.0],
                        timestamps=[10.0],
                    )
                else:
                    # Violations in window starting at 0 (t=10 → window [0,300))
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[15.0],
                        timestamps=[10.0],
                    )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        tuner_config = replace(
            setup["tuner_config"],
            checkpoint_budget=1,
            checkpoint_epsilon=-999.0,
        )

        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=tuner_config,
            initial_config=setup["initial_config"],
            run_dir=tmp_path / "run",
        )

        checkpoints = optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=1.0,
        )
        assert len(checkpoints) == 1
        # Window starts at 0, spin_up_delay = 60 → max(0, 0-60) = 0.
        assert checkpoints[0].time_s == 0.0

    def test_picks_best_rpu_by_violation(self, setup):
        """The optimizer picks the RPU size with lowest training violation."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]
        call_count = {"n": 0}

        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            call_count["n"] += 1
            results = []
            for i in range(len(workload_paths)):
                gp_str = str(grid_point)
                if "rpu16" in gp_str:
                    # RPU 16 → lower violation
                    lats = [5.0, 5.0]
                elif "rpu8" in gp_str:
                    # RPU 8 → higher violation
                    lats = [15.0, 15.0]
                elif "val" in gp_str:
                    # Validation: low violation so it passes epsilon check
                    lats = [5.0, 5.0]
                else:
                    # Base round: high violation to trigger the loop
                    lats = [15.0, 15.0]

                r = _make_scenario_result(
                    tmp_path / f"eval_{call_count['n']}_{grid_point}",
                    scenario_idx=i,
                    latencies=lats,
                    timestamps=[10.0, 20.0],
                )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        tuner_config = replace(
            setup["tuner_config"],
            checkpoint_budget=1,
            checkpoint_epsilon=-999.0,
        )

        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=tuner_config,
            initial_config=setup["initial_config"],
            run_dir=tmp_path / "run",
        )

        checkpoints = optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=1.0,
        )
        assert len(checkpoints) == 1
        # Should pick RPU 16 since it has lower violation.
        assert checkpoints[0].min_rpus == (16,)

    def test_writes_selected_checkpoints_yml(self, setup):
        """The optimizer writes selected_checkpoints.yml."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]
        call_count = {"n": 0}

        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            call_count["n"] += 1
            is_val = "val" in str(grid_point)
            results = []
            for i in range(len(workload_paths)):
                if is_val:
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[5.0],
                        timestamps=[10.0],
                    )
                else:
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[15.0],
                        timestamps=[310.0],
                    )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        tuner_config = replace(
            setup["tuner_config"],
            checkpoint_budget=1,
            checkpoint_epsilon=-999.0,
        )

        run_dir = tmp_path / "run"
        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=tuner_config,
            initial_config=setup["initial_config"],
            run_dir=run_dir,
        )

        checkpoints = optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=1.0,
        )

        yml_path = run_dir / "checkpoints" / "selected_checkpoints.yml"
        assert yml_path.exists()
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, list)
        assert len(data) == len(checkpoints)

    def test_writes_round_summary(self, setup):
        """The optimizer writes candidate_results.yml for each round."""
        mock_eval = self._make_mock_evaluator()
        tmp_path = setup["tmp_path"]
        call_count = {"n": 0}

        def evaluate_side_effect(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            call_count["n"] += 1
            is_val = "val" in str(grid_point)
            results = []
            for i in range(len(workload_paths)):
                if is_val:
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[5.0],
                        timestamps=[10.0],
                    )
                else:
                    r = _make_scenario_result(
                        tmp_path / f"eval_{call_count['n']}_{grid_point}",
                        scenario_idx=i,
                        latencies=[15.0],
                        timestamps=[310.0],
                    )
                results.append(r)
            return results

        mock_eval.evaluate.side_effect = evaluate_side_effect

        tuner_config = replace(
            setup["tuner_config"],
            checkpoint_budget=1,
            checkpoint_epsilon=-999.0,
        )

        run_dir = tmp_path / "run"
        optimizer = CheckpointOptimizer(
            evaluator=mock_eval,
            tuner_config=tuner_config,
            initial_config=setup["initial_config"],
            run_dir=run_dir,
        )

        optimizer.optimize(
            train_paths=setup["train_paths"],
            val_paths=setup["val_paths"],
            baseline_val_violation=1.0,
        )

        summary_path = (
            run_dir / "checkpoints" / "round_000" / "candidate_results.yml"
        )
        assert summary_path.exists()
        with open(summary_path) as f:
            data = yaml.safe_load(f)
        assert "candidates" in data
        assert "selected_checkpoint" in data


# ---------------------------------------------------------------------------
# PolicyTuner.evaluate_baseline (integration-style with mock evaluator)
# ---------------------------------------------------------------------------


class TestPolicyTunerBaseline:
    def test_evaluate_baseline_returns_phase_result(self, tmp_path: Path):
        """evaluate_baseline collects results and writes summary.yml."""
        from autoslo.tuner.policy_tuner import PolicyTuner
        from autoslo.tuner.types import PhaseResult

        tuner_config = TunerConfig(
            aggregation_metric="mean",
            parallelism=1,
        )
        initial_config: dict[str, Any] = {
            "slo_config": {"slo_s": 10.0},
            "basic_config": {"schema_name": "test"},
        }

        tuner = PolicyTuner(
            initial_config=initial_config,
            tuner_config=tuner_config,
            run_dir=tmp_path / "run",
        )

        # Create dummy workload files.
        train_paths = []
        val_paths = []
        for i in range(2):
            p = tmp_path / f"t_{i:03d}.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(p)
            train_paths.append(p)
            p = tmp_path / f"v_{i:03d}.parquet"
            pd.DataFrame({"x": [1]}).to_parquet(p)
            val_paths.append(p)

        # Mock the evaluator to return controlled results.
        def mock_evaluate(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            return [
                ScenarioResult(
                    scenario_idx=i,
                    violation_rate=0.1 * (i + 1),
                    violation_amount_s=1.0,
                    violation_relative_mean=0.05,
                    total_cost=2.0 + i,
                    num_queries=10,
                    out_dir=tmp_path / f"out_{i}",
                )
                for i in range(len(workload_paths))
            ]

        tuner._evaluator.evaluate = mock_evaluate

        result = tuner.evaluate_baseline(train_paths, val_paths)

        assert isinstance(result, PhaseResult)
        assert len(result.train_results) == 2
        assert result.val_results is not None
        assert len(result.val_results) == 2
        assert result.train_violation_agg > 0
        assert result.train_cost_agg > 0
        assert result.val_violation_agg > 0
        assert result.val_cost_agg > 0

        # Check summary written.
        summary_path = tmp_path / "run" / "baseline" / "summary.yml"
        assert summary_path.exists()

    def test_evaluate_baseline_calls_evaluator_with_no_overrides(
        self, tmp_path: Path
    ):
        """evaluate_baseline passes empty config_overrides."""
        from autoslo.tuner.policy_tuner import PolicyTuner

        tuner_config = TunerConfig(parallelism=1)
        tuner = PolicyTuner(
            initial_config={"slo_config": {"slo_s": 10.0}},
            tuner_config=tuner_config,
            run_dir=tmp_path / "run",
        )

        calls = []

        def mock_evaluate(
            workload_paths, config_overrides, phase, grid_point, out_subdir
        ):
            calls.append(
                {
                    "config_overrides": config_overrides,
                    "phase": phase,
                    "grid_point": grid_point,
                }
            )
            return [
                ScenarioResult(
                    scenario_idx=0,
                    violation_rate=0.0,
                    violation_amount_s=0.0,
                    violation_relative_mean=0.0,
                    total_cost=1.0,
                    num_queries=10,
                    out_dir=tmp_path,
                )
            ]

        tuner._evaluator.evaluate = mock_evaluate

        p = tmp_path / "w.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(p)

        tuner.evaluate_baseline([p], [p])

        # Should have been called twice (train + val).
        assert len(calls) == 2
        assert calls[0]["config_overrides"] == {}
        assert calls[0]["phase"] == "baseline"
        assert calls[1]["phase"] == "baseline"
