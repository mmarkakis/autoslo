"""Integration tests for the PolicyTuner end-to-end pipeline.

These are smoke tests that verify the wiring between all phases.  Heavy
simulator and model dependencies are mocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.config import TunerConfig
from autoslo.tuner.policy_tuner import PolicyTuner
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.tuner_utils import ScenarioResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace_parquet(path: Path, n: int = 5) -> Path:
    """Write a minimal workload parquet file usable as a trace."""
    base_time = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(n)],
            "abs_start_time": [
                base_time + pd.Timedelta(seconds=i * 600) for i in range(n)
            ],
            "query_text_id": [f"ext_tpcds1000#{(i % 3) + 1}#001" for i in range(n)],
            "repetition_id": [f"rep_{(i % 3) + 1}" for i in range(n)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _make_scenario_result(
    scenario_idx: int,
    violation_rate: float = 0.05,
    total_cost: float = 10.0,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_idx=scenario_idx,
        violation_rate=violation_rate,
        violation_amount_s=0.0,
        violation_relative_mean=0.0,
        total_cost=total_cost,
        num_queries=100,
        out_dir=Path("/tmp/fake"),
    )


def _make_tuner_config() -> TunerConfig:
    """Minimal tuner config for fast integration tests."""
    return TunerConfig(
        num_scenarios=4,
        train_fraction=0.5,
        random_seed=42,
        target_start=datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc),
        target_end=datetime(2024, 6, 3, 11, 0, 0, tzinfo=timezone.utc),
        forecast_policy="uniform",
        aggregation_metric="mean",
        checkpoint_budget=1,
        checkpoint_epsilon=0.01,
        sliding_window_s=300.0,
        violation_threshold=0.1,
        autoscaler_ranges={"eta_crit": [0.5]},
        routing_ranges={"fallback_tightness": [0.7]},
        parallelism=1,
    )


def _make_initial_config() -> dict[str, Any]:
    return {
        "basic_config": {"schema_name": "ext_tpcds1000"},
        "slo_config": {"slo_s": 10.0},
        "autoscaling_config": {},
        "routing_config": {},
        "managed_cluster_pool_config": {"spin_up_delay_s": 0.0},
    }


# ---------------------------------------------------------------------------
# Tests: build_reservoir
# ---------------------------------------------------------------------------


class TestBuildReservoir:
    def test_creates_reservoir_files(self, tmp_path: Path):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")

        reservoir_path = tuner.build_reservoir([trace])

        assert (reservoir_path / "reservoir.parquet").exists()
        assert (reservoir_path / "reservoir_meta.yml").exists()

    def test_reservoir_loadable(self, tmp_path: Path):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")

        reservoir_path = tuner.build_reservoir([trace])
        reservoir = QueryReservoir.load(reservoir_path)
        assert len(reservoir.df) > 0


# ---------------------------------------------------------------------------
# Tests: sample_workloads
# ---------------------------------------------------------------------------


class TestSampleWorkloads:
    def test_produces_train_and_val_paths(self, tmp_path: Path):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")

        reservoir_path = tuner.build_reservoir([trace])
        train_paths, val_paths = tuner.sample_workloads(reservoir_path)

        assert len(train_paths) == cfg.n_train
        assert len(val_paths) == cfg.n_val
        for p in train_paths + val_paths:
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_workloads_readable(self, tmp_path: Path):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")

        reservoir_path = tuner.build_reservoir([trace])
        train_paths, _ = tuner.sample_workloads(reservoir_path)

        df = pd.read_parquet(train_paths[0])
        for col in ["query_id", "abs_start_time", "query_text_id"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Tests: tune() end-to-end (with mocked evaluator)
# ---------------------------------------------------------------------------


class TestTuneEndToEnd:
    """Smoke test that runs the full pipeline with a mocked evaluator."""

    @pytest.fixture()
    def run_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "run"

    def _mock_evaluator_side_effect(self, *args, **kwargs):
        """Return canned ScenarioResult lists for any evaluate() call."""
        workload_paths = kwargs.get("workload_paths", args[0] if args else [])
        return [
            _make_scenario_result(i, violation_rate=0.08, total_cost=50.0)
            for i in range(len(workload_paths))
        ]

    def test_tune_produces_final_config(self, tmp_path: Path, run_dir: Path):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        initial = _make_initial_config()
        tuner = PolicyTuner(initial, cfg, run_dir=run_dir)

        # Mock the evaluator so we don't need real simulators.
        tuner._evaluator.evaluate = MagicMock(
            side_effect=self._mock_evaluator_side_effect
        )

        # Mock find_violation_windows to return a window above threshold
        # for the first call (so checkpoint optimizer does one round),
        # then no violations.
        from autoslo.tuner.checkpoint_optimizer import ViolationWindow

        violation_window = ViolationWindow(
            start_s=100.0,
            end_s=400.0,
            violation_rate=0.15,
            num_violations=10,
            num_queries=100,
        )

        call_count = {"n": 0}
        def mock_find_violation_windows(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return [violation_window]
            return []

        with patch(
            "autoslo.tuner.checkpoint_optimizer.find_violation_windows",
            side_effect=mock_find_violation_windows,
        ):
            final_path = tuner.tune([trace])

        assert final_path.exists()
        assert final_path.name == "final_config.yml"

        # Verify the final config is valid YAML.
        with open(final_path) as f:
            final_cfg = yaml.safe_load(f)
        assert isinstance(final_cfg, dict)
        assert "autoscaling_config" in final_cfg

    def test_tune_creates_expected_dirs(self, tmp_path: Path, run_dir: Path):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=run_dir)

        tuner._evaluator.evaluate = MagicMock(
            side_effect=self._mock_evaluator_side_effect
        )

        with patch(
            "autoslo.tuner.checkpoint_optimizer.find_violation_windows",
            return_value=[],
        ):
            tuner.tune([trace])

        # Key directories should exist.
        assert (run_dir / "reservoir").is_dir()
        assert (run_dir / "sampled_workloads" / "train").is_dir()
        assert (run_dir / "sampled_workloads" / "val").is_dir()
        assert (run_dir / "initial_config.yml").exists()
        assert (run_dir / "tuner_config.yml").exists()
        assert (run_dir / "final_config.yml").exists()

    def test_tune_with_no_violation_windows(self, tmp_path: Path, run_dir: Path):
        """When no violations, checkpoint optimizer exits immediately."""
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=run_dir)

        tuner._evaluator.evaluate = MagicMock(
            side_effect=self._mock_evaluator_side_effect
        )

        with patch(
            "autoslo.tuner.checkpoint_optimizer.find_violation_windows",
            return_value=[],
        ):
            final_path = tuner.tune([trace])

        with open(final_path) as f:
            final_cfg = yaml.safe_load(f)
        # With no violation windows and a 1-point grid, config should be set.
        assert isinstance(final_cfg, dict)


# ---------------------------------------------------------------------------
# Tests: _write_final_config
# ---------------------------------------------------------------------------


class TestWriteFinalConfig:
    def test_overlays_checkpoints(self, tmp_path: Path):
        cfg = _make_tuner_config()
        initial = _make_initial_config()
        tuner = PolicyTuner(initial, cfg, run_dir=tmp_path / "run")

        checkpoints = [CapacityCheckpoint(time_s=60.0, min_rpus=(8,))]
        autoscaler_params = {"eta_crit": 0.5}
        routing_params = {"fallback_tightness": 0.7}

        path = tuner._write_final_config(
            checkpoints, autoscaler_params, routing_params
        )

        with open(path) as f:
            result = yaml.safe_load(f)

        assert "capacity_checkpoints" in result["autoscaling_config"]
        assert result["autoscaling_config"]["eta_crit"] == 0.5
        assert result["routing_config"]["fallback_tightness"] == 0.7

# ---------------------------------------------------------------------------
# Tests: __main__ CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_cli_module_importable(self):
        """Verify the CLI module can be imported."""
        from autoslo.tuner import __main__
        assert hasattr(__main__, "main")
