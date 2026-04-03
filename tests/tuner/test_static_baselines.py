"""Tests for Phase 9: static baselines, skip-retuning, and related helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.config import TunerConfig, load_tuner_config
from autoslo.tuner.policy_tuner import PolicyTuner
from autoslo.tuner.scenario_evaluator import EvalSpec
from autoslo.tuner.tuner_utils import ScenarioResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace_parquet(path: Path, n: int = 5) -> Path:
    base_time = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(n)],
            "abs_start_time": [
                base_time + pd.Timedelta(seconds=i * 600) for i in range(n)
            ],
            "query_text_id": [
                f"ext_tpcds1000#{(i % 3) + 1}#001" for i in range(n)
            ],
            "repetition_id": [f"rep_{(i % 3) + 1}" for i in range(n)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _make_scenario_result(
    scenario_idx: int,
    violation_rate: float = 0.08,
    total_cost: float = 50.0,
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


def _make_tuner_config(**overrides: Any) -> TunerConfig:
    defaults = dict(
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
    defaults.update(overrides)
    return TunerConfig(**defaults)


def _make_initial_config() -> dict[str, Any]:
    return {
        "basic_config": {"schema_name": "ext_tpcds1000"},
        "slo_config": {"slo_s": 10.0},
        "autoscaling_config": {},
        "routing_config": {},
        "managed_cluster_pool_config": {"spin_up_delay_s": 0.0},
    }


def _evaluator_side_effect(
    violation_rate: float = 0.08, total_cost: float = 50.0
):
    def _side_effect(*args, **kwargs):
        workload_paths = kwargs.get("workload_paths", args[0] if args else [])
        return [
            _make_scenario_result(i, violation_rate=violation_rate, total_cost=total_cost)
            for i in range(len(workload_paths))
        ]
    return _side_effect


def _evaluator_batch_side_effect(
    violation_rate: float = 0.08, total_cost: float = 50.0
):
    """Mock for evaluate_batch — returns list[list[ScenarioResult]]."""
    def _side_effect(*args, **kwargs):
        workload_paths = kwargs.get("workload_paths", args[0] if args else [])
        specs = kwargs.get("specs", args[1] if len(args) > 1 else [])
        return [
            [
                _make_scenario_result(
                    i, violation_rate=violation_rate, total_cost=total_cost
                )
                for i in range(len(workload_paths))
            ]
            for _ in specs
        ]
    return _side_effect


# ---------------------------------------------------------------------------
# TunerConfig: static_baselines and force_retuning fields
# ---------------------------------------------------------------------------


class TestTunerConfigStaticBaselines:
    def test_defaults_none_and_false(self):
        cfg = TunerConfig()
        assert cfg.static_baselines is None
        assert cfg.force_retuning is False

    def test_static_baselines_from_yaml(self, tmp_path: Path):
        raw = {
            "static_baselines": [
                {
                    "label": "4 RPU",
                    "overrides": {
                        "managed_cluster_pool_config.initial_rpus": [4],
                        "autoscaling_config.autoscaling_policy": "noop",
                    },
                }
            ],
            "force_retuning": True,
        }
        p = tmp_path / "cfg.yml"
        with open(p, "w") as f:
            yaml.dump(raw, f)
        cfg = load_tuner_config(p)
        assert len(cfg.static_baselines) == 1
        assert cfg.static_baselines[0]["label"] == "4 RPU"
        assert cfg.force_retuning is True


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    @pytest.mark.parametrize(
        "label, expected",
        [
            ("4 RPU", "static_4_rpu"),
            ("16 RPU", "static_16_rpu"),
            ("Hello World!", "static_hello_world"),
            ("a-b.c", "static_abc"),
            ("ALL CAPS 123", "static_all_caps_123"),
        ],
    )
    def test_slugify(self, label: str, expected: str):
        assert PolicyTuner._slugify(label) == expected


# ---------------------------------------------------------------------------
# _prepare_holdout_workloads
# ---------------------------------------------------------------------------


class TestPrepareHoldoutWorkloads:
    def test_creates_parquets(self, tmp_path: Path):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")

        paths = tuner._prepare_holdout_workloads([trace])
        assert len(paths) > 0
        for p in paths:
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_empty_when_no_matching_period(self, tmp_path: Path):
        # Target period is June 3 09:00–11:00 UTC, but make trace in Jan 2020
        base_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        df = pd.DataFrame(
            {
                "query_id": ["q0"],
                "abs_start_time": [base_time],
                "query_text_id": ["ext_tpcds1000#1#001"],
                "repetition_id": ["rep_1"],
            }
        )
        trace_path = tmp_path / "traces" / "t.parquet"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(trace_path, index=False)

        cfg = _make_tuner_config()
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")
        paths = tuner._prepare_holdout_workloads([trace_path])
        assert paths == []


# ---------------------------------------------------------------------------
# _evaluate_holdout with static baselines
# ---------------------------------------------------------------------------


class TestEvaluateHoldoutStaticBaselines:
    @pytest.fixture()
    def tuner_with_statics(self, tmp_path: Path) -> PolicyTuner:
        cfg = _make_tuner_config(
            holdout_evaluation=True,
            static_baselines=[
                {
                    "label": "4 RPU",
                    "overrides": {
                        "managed_cluster_pool_config.initial_rpus": [4],
                        "autoscaling_config.autoscaling_policy": "noop",
                    },
                },
                {
                    "label": "8 RPU",
                    "overrides": {
                        "managed_cluster_pool_config.initial_rpus": [8],
                        "autoscaling_config.autoscaling_policy": "noop",
                    },
                },
            ],
        )
        return PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")

    def test_static_baselines_in_summary(self, tmp_path: Path, tuner_with_statics: PolicyTuner):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        tuner_with_statics._evaluator.evaluate_batch = MagicMock(
            side_effect=_evaluator_batch_side_effect(0.06, 80.0)
        )

        tuner_with_statics._evaluate_holdout([trace], tuned_overrides={"eta_crit": 0.5})

        summary_path = tuner_with_statics.run_dir / "holdout" / "summary.yml"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = yaml.safe_load(f)

        assert "static_baselines" in summary
        assert len(summary["static_baselines"]) == 2
        assert summary["static_baselines"][0]["label"] == "4 RPU"
        assert summary["static_baselines"][1]["label"] == "8 RPU"

    def test_static_baseline_dirs_created(self, tmp_path: Path, tuner_with_statics: PolicyTuner):
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        tuner_with_statics._evaluator.evaluate_batch = MagicMock(
            side_effect=_evaluator_batch_side_effect()
        )

        tuner_with_statics._evaluate_holdout([trace], tuned_overrides={})

        # evaluate_batch was called with specs containing slugified out_subdirs.
        call_args = tuner_with_statics._evaluator.evaluate_batch.call_args
        specs = call_args.kwargs["specs"]
        out_subdirs = [str(spec.out_subdir) for spec in specs]
        assert any("static_4_rpu" in d for d in out_subdirs)
        assert any("static_8_rpu" in d for d in out_subdirs)

    def test_no_static_baselines_still_works(self, tmp_path: Path):
        cfg = _make_tuner_config(holdout_evaluation=True)
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        tuner._evaluator.evaluate_batch = MagicMock(
            side_effect=_evaluator_batch_side_effect()
        )

        tuner._evaluate_holdout([trace], tuned_overrides={})

        summary_path = tuner.run_dir / "holdout" / "summary.yml"
        with open(summary_path) as f:
            summary = yaml.safe_load(f)

        assert "static_baselines" not in summary
        # Single evaluate_batch call with 2 specs (baseline + tuned).
        assert tuner._evaluator.evaluate_batch.call_count == 1
        specs = tuner._evaluator.evaluate_batch.call_args.kwargs["specs"]
        assert len(specs) == 2


# ---------------------------------------------------------------------------
# Skip-retuning logic
# ---------------------------------------------------------------------------


class TestSkipRetuning:
    def _run_initial_tune(self, tmp_path: Path, cfg: TunerConfig) -> Path:
        """Run tune() once to produce final_config.yml + tuned_overrides.yml."""
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        tuner = PolicyTuner(_make_initial_config(), cfg, run_dir=tmp_path / "run")
        tuner._evaluator.evaluate = MagicMock(
            side_effect=_evaluator_side_effect()
        )
        tuner._evaluator.evaluate_batch = MagicMock(
            side_effect=_evaluator_batch_side_effect()
        )

        with patch(
            "autoslo.tuner.checkpoint_optimizer.find_violation_windows",
            return_value=[],
        ):
            return tuner.tune([trace])

    def test_skip_retuning_when_artifacts_exist(self, tmp_path: Path):
        cfg = _make_tuner_config(holdout_evaluation=True, force_retuning=False)
        final_path = self._run_initial_tune(tmp_path, cfg)
        run_dir = final_path.parent

        assert (run_dir / "final_config.yml").exists()
        assert (run_dir / "tuned_overrides.yml").exists()

        # Second run should skip phases 1–7.
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        tuner2 = PolicyTuner(_make_initial_config(), cfg, run_dir=run_dir)
        tuner2._evaluator.evaluate = MagicMock(
            side_effect=_evaluator_side_effect()
        )
        tuner2._evaluator.evaluate_batch = MagicMock(
            side_effect=_evaluator_batch_side_effect()
        )

        final_path2 = tuner2.tune([trace])
        assert final_path2 == final_path

        # No phase 3–6 evaluate() calls; one evaluate_batch() for holdout.
        assert tuner2._evaluator.evaluate.call_count == 0
        assert tuner2._evaluator.evaluate_batch.call_count == 1

    def test_force_retuning_reruns_all_phases(self, tmp_path: Path):
        cfg = _make_tuner_config(holdout_evaluation=True, force_retuning=False)
        final_path = self._run_initial_tune(tmp_path, cfg)
        run_dir = final_path.parent

        # Now create a tuner with force_retuning=True.
        cfg_force = _make_tuner_config(holdout_evaluation=True, force_retuning=True)
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        tuner2 = PolicyTuner(_make_initial_config(), cfg_force, run_dir=run_dir)
        tuner2._evaluator.evaluate = MagicMock(
            side_effect=_evaluator_side_effect()
        )
        tuner2._evaluator.evaluate_batch = MagicMock(
            side_effect=_evaluator_batch_side_effect()
        )

        with patch(
            "autoslo.tuner.checkpoint_optimizer.find_violation_windows",
            return_value=[],
        ):
            tuner2.tune([trace])

        # With force_retuning, full pipeline runs: baseline, checkpoint,
        # autoscaler sweep, routing sweep, final eval → evaluate().
        # Holdout → evaluate_batch().
        assert tuner2._evaluator.evaluate.call_count > 0
        assert tuner2._evaluator.evaluate_batch.call_count == 1

    def test_tuned_overrides_persisted(self, tmp_path: Path):
        cfg = _make_tuner_config(holdout_evaluation=False)
        final_path = self._run_initial_tune(tmp_path, cfg)
        run_dir = final_path.parent

        overrides_path = run_dir / "tuned_overrides.yml"
        assert overrides_path.exists()
        with open(overrides_path) as f:
            overrides = yaml.safe_load(f)
        assert isinstance(overrides, dict)

    def test_skip_retuning_rebuilds_overrides_when_missing(self, tmp_path: Path):
        """Legacy runs that have final_config.yml but no tuned_overrides.yml."""
        cfg = _make_tuner_config(holdout_evaluation=True, force_retuning=False)
        final_path = self._run_initial_tune(tmp_path, cfg)
        run_dir = final_path.parent

        # Remove tuned_overrides.yml to simulate a legacy run.
        overrides_path = run_dir / "tuned_overrides.yml"
        overrides_path.unlink()
        assert not overrides_path.exists()

        # Second run should still skip phases 1–7 and reconstruct overrides.
        trace = _make_trace_parquet(tmp_path / "traces" / "t.parquet")
        tuner2 = PolicyTuner(_make_initial_config(), cfg, run_dir=run_dir)
        tuner2._evaluator.evaluate = MagicMock(
            side_effect=_evaluator_side_effect()
        )
        tuner2._evaluator.evaluate_batch = MagicMock(
            side_effect=_evaluator_batch_side_effect()
        )

        final_path2 = tuner2.tune([trace])
        assert final_path2 == final_path

        # tuned_overrides.yml should have been reconstructed.
        assert overrides_path.exists()
        with open(overrides_path) as f:
            rebuilt = yaml.safe_load(f)
        assert isinstance(rebuilt, dict)

        # No phase 3–6 calls; one batch call for holdout.
        assert tuner2._evaluator.evaluate.call_count == 0
        assert tuner2._evaluator.evaluate_batch.call_count == 1
