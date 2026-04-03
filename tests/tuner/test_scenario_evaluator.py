"""Tests for ScenarioEvaluator and extract_scenario_result."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from autoslo.tuner.config import TunerConfig
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator, _run_scenario
from autoslo.tuner.tuner_utils import ScenarioResult, extract_scenario_result
from autoslo.utils.structured_log import StructuredLogHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workload_parquet(path: Path, n: int = 1) -> Path:
    """Write a minimal workload parquet file and return its path."""
    df = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(n)],
            "abs_start_time": [float(i * 1000) for i in range(n)],
            "query_text_id": [f"{i:03d}" for i in range(n)],
            "repetition_id": [f"r{i}" for i in range(n)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_billing(out_dir: Path, cost_per_cluster: dict[str, float]) -> None:
    """Write a minimal billing_interval_analysis.yml."""
    billing = {}
    for name, cost in cost_per_cluster.items():
        billing[name] = {"total_billed_cost": cost}
    with open(out_dir / "billing_interval_analysis.yml", "w") as f:
        yaml.dump(billing, f)


def _write_structured_log(
    out_dir: Path,
    latencies: list[float],
    query_text_ids: list[str] | None = None,
) -> None:
    """Write a minimal structured_log.parquet with completion events."""
    n = len(latencies)
    if query_text_ids is None:
        query_text_ids = [f"q{i:03d}" for i in range(n)]
    df = pd.DataFrame(
        {
            "timestamp": range(n),
            "source": ["simulator"] * n,
            "event_type": ["completion"] * n,
            "query_id": [f"id_{i}" for i in range(n)],
            "query_text_id": query_text_ids,
            "cluster_name": ["cluster_0"] * n,
            "latency_s": latencies,
        }
    )
    df.to_parquet(out_dir / "structured_log.parquet", index=False)


# ---------------------------------------------------------------------------
# extract_scenario_result
# ---------------------------------------------------------------------------


class TestExtractScenarioResult:
    def test_basic(self, tmp_path: Path):
        _write_billing(tmp_path, {"c0": 5.0, "c1": 3.0})
        # SLO = 10.0; latencies: 8.0 (ok), 12.0 (violation), 10.0 (ok)
        _write_structured_log(tmp_path, [8.0, 12.0, 10.0])

        result = extract_scenario_result(tmp_path, scenario_idx=0, slo_s=10.0)
        assert result.scenario_idx == 0
        assert result.total_cost == pytest.approx(8.0)
        assert result.num_queries == 3
        # 1 of 3 queries violated
        assert result.violation_rate == pytest.approx(1.0 / 3.0)
        # violation amount = max(0, 12-10) = 2.0
        assert result.violation_amount_s == pytest.approx(2.0)
        assert result.out_dir == tmp_path

    def test_no_violations(self, tmp_path: Path):
        _write_billing(tmp_path, {"c0": 1.0})
        _write_structured_log(tmp_path, [5.0, 6.0])

        result = extract_scenario_result(tmp_path, scenario_idx=1, slo_s=10.0)
        assert result.violation_rate == 0.0
        assert result.violation_amount_s == 0.0
        assert result.total_cost == pytest.approx(1.0)
        assert result.num_queries == 2

    def test_empty_dir(self, tmp_path: Path):
        result = extract_scenario_result(tmp_path, scenario_idx=0, slo_s=10.0)
        assert result.total_cost == 0.0
        assert result.num_queries == 0
        assert result.violation_rate == 0.0

    def test_per_template_slo(self, tmp_path: Path):
        _write_billing(tmp_path, {"c0": 1.0})
        # q001 has SLO override of 5.0; latency 6.0 → violation
        # q002 uses default SLO 10.0; latency 6.0 → no violation
        _write_structured_log(
            tmp_path, [6.0, 6.0], query_text_ids=["001", "002"]
        )

        result = extract_scenario_result(
            tmp_path, scenario_idx=0, slo_s=10.0, slo_dict={"001": 5.0}
        )
        assert result.violation_rate == pytest.approx(0.5)
        assert result.num_queries == 2


# ---------------------------------------------------------------------------
# ScenarioEvaluator internals
# ---------------------------------------------------------------------------


class TestScenarioEvaluatorInternals:
    @pytest.fixture()
    def evaluator(self, tmp_path: Path) -> ScenarioEvaluator:
        handler = StructuredLogHandler(
            out_dir=str(tmp_path / "logs"),
            filename="evolution.parquet",
        )
        return ScenarioEvaluator(
            initial_config={
                "basic_config": {"schema_name": "test", "iconq_model_id": "m1"},
                "workload_config": {"workload_name": "w1"},
                "slo_config": {"slo_s": 10.0},
            },
            tuner_config=TunerConfig(parallelism=2),
            tuner_run_id="tuner_test_123",
            evolution_logger=handler,
        )

    def test_resolve_parallelism_explicit(self, evaluator: ScenarioEvaluator):
        assert evaluator._resolve_parallelism() == 2

    def test_resolve_parallelism_auto(self, tmp_path: Path):
        handler = StructuredLogHandler(out_dir=str(tmp_path), filename="evo.parquet")
        ev = ScenarioEvaluator(
            initial_config={},
            tuner_config=TunerConfig(parallelism="auto"),
            tuner_run_id="t",
            evolution_logger=handler,
        )
        assert ev._resolve_parallelism() >= 1

    def test_build_work_units(self, evaluator: ScenarioEvaluator, tmp_path: Path):
        # Create two workload parquet files on disk.
        wl0_path = _make_workload_parquet(tmp_path / "train" / "t_000.parquet")
        wl1_path = _make_workload_parquet(tmp_path / "train" / "t_001.parquet")

        units = evaluator._build_work_units(
            workload_paths=[wl0_path, wl1_path],
            config_overrides={"autoscaling_config.eta_crit": 0.5},
            phase="baseline",
            grid_point=0,
            out_subdir=tmp_path / "out",
            schema_name="test",
        )

        assert len(units) == 2
        # Check run ID format.
        assert units[0]["config_dict"]["basic_config"]["simulator_run_id"] == (
            "tuner_test_123_baseline_0_000"
        )
        assert units[1]["config_dict"]["basic_config"]["simulator_run_id"] == (
            "tuner_test_123_baseline_0_001"
        )
        # Check overrides applied.
        assert units[0]["config_dict"]["autoscaling_config"]["eta_crit"] == 0.5
        # Check verbose forced on.
        assert units[0]["config_dict"]["output_config"]["verbose"] is True
        # Check workload path stored (not bytes).
        assert units[0]["workload_path"] == str(wl0_path)
        assert units[1]["workload_path"] == str(wl1_path)
        # Check workload name derived from filename stem.
        assert units[0]["workload_name"] == "t_000"
        assert units[1]["workload_name"] == "t_001"

    def test_run_id_format(self, evaluator: ScenarioEvaluator, tmp_path: Path):
        """Run IDs encode tuner_run_id, phase, grid_point, and scenario idx."""
        wl_path = _make_workload_parquet(tmp_path / "train" / "t_000.parquet")
        units = evaluator._build_work_units(
            workload_paths=[wl_path],
            config_overrides={},
            phase="ckpt",
            grid_point="rpu32",
            out_subdir=tmp_path,
            schema_name="s",
        )
        rid = units[0]["config_dict"]["basic_config"]["simulator_run_id"]
        assert rid == "tuner_test_123_ckpt_rpu32_000"


# ---------------------------------------------------------------------------
# Worker env-var setup
# ---------------------------------------------------------------------------


class TestRunScenarioEnvVars:
    """Verify parallelism constants used by _run_scenario."""

    def test_inner_level_cpus_positive(self):
        from autoslo.utils.paralellism import inner_level_num_cpus

        assert inner_level_num_cpus() >= 1

    def test_deg_of_parallelism_positive(self):
        from autoslo.utils.paralellism import deg_of_paralellism

        assert deg_of_paralellism() >= 1
