"""Tests for ExecutionResult and Trace.aborted_query_ids_from_dir."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from autoslo.config.component_configs import SloResolverConfig
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_execution.execution_result import ExecutionResult
from autoslo.workload_execution.trace import Trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLO_S = 5.0

_EXECUTION_CONFIG = {
    "slo_resolver_config": {
        "slo_s": _SLO_S,
        "slo_dict_filename": None,
    },
    "slo_objective_config": {
        "slo_metric": "binary",
        "slo_threshold": 0.1,
    },
}

_RESOLVER = SloResolver(SloResolverConfig(slo_s=_SLO_S, slo_dict_filename=None))


def _write_execution_config(d: Path) -> None:
    with open(d / "execution_config.yml", "w") as f:
        yaml.dump(_EXECUTION_CONFIG, f)


def _write_structured_log(
    d: Path,
    queries: list[dict],
) -> None:
    """Write a minimal structured_log.parquet.

    Each entry in *queries* is a dict with keys:
        query_id, query_text_id, arrival_s, completion_s
    Two rows are written per query: one ARRIVAL, one COMPLETION.
    """
    rows = []
    for q in queries:
        rows.append(
            {
                "query_id": q["query_id"],
                "query_text_id": q["query_text_id"],
                "rel_time_s": q["arrival_s"],
                "event_type": "arrival",
                "source": "test",
                "wall_clock_s": 0.0,
            }
        )
        rows.append(
            {
                "query_id": q["query_id"],
                "query_text_id": q["query_text_id"],
                "rel_time_s": q["completion_s"],
                "event_type": "completion",
                "source": "test",
                "wall_clock_s": 0.0,
            }
        )
    pd.DataFrame(rows).to_parquet(d / "structured_log.parquet", index=False)


def _write_billing(d: Path, total_billed_cost: float) -> None:
    data = {
        "cluster_a": {
            "total_billed_cost": total_billed_cost,
        }
    }
    with open(d / "billing_interval_analysis.yml", "w") as f:
        yaml.dump(data, f)


def _write_sys_serverless_usage(
    d: Path,
    charged_seconds: float,
    cluster_name: str = "cluster_a",
) -> None:
    df = pd.DataFrame(
        {
            "start_time": [pd.Timestamp("2024-01-01")],
            "end_time": [pd.Timestamp("2024-01-01 00:01:00")],
            "charged_seconds": [charged_seconds],
            "charged_extra_compute_for_automatic_optimization_seconds": [0.0],
        }
    )
    df.to_parquet(
        d / f"sys_serverless_usage+{cluster_name}.parquet", index=False
    )


def _write_sys_query_history(
    d: Path,
    rows: list[dict],
    cluster_name: str = "cluster_a",
) -> None:
    """Write sys_query_history+<cluster>.parquet.

    Each dict in *rows* needs at least ``query_id`` and ``status``.
    """
    pd.DataFrame(rows).to_parquet(
        d / f"sys_query_history+{cluster_name}.parquet", index=False
    )


# ---------------------------------------------------------------------------
# ExecutionResult.load — simulation
# ---------------------------------------------------------------------------


def test_load_simulation_no_violations(tmp_path: Path) -> None:
    _write_execution_config(tmp_path)
    _write_billing(tmp_path, total_billed_cost=1.23)
    # Both queries complete well within SLO (latency = 1 s, SLO = 5 s)
    _write_structured_log(
        tmp_path,
        [
            {"query_id": "q1", "query_text_id": "s#001#001", "arrival_s": 0.0, "completion_s": 1.0},
            {"query_id": "q2", "query_text_id": "s#001#002", "arrival_s": 1.0, "completion_s": 2.0},
        ],
    )

    r = ExecutionResult.load(tmp_path)

    assert r.total_cost == pytest.approx(1.23)
    assert r.num_queries == 2
    assert r.violation_rate == pytest.approx(0.0)
    assert r.violation_amount_s == pytest.approx(0.0)
    assert r.violation_relative_mean == pytest.approx(0.0)


def test_load_simulation_all_violations(tmp_path: Path) -> None:
    _write_execution_config(tmp_path)
    _write_billing(tmp_path, total_billed_cost=0.5)
    # Both queries take 10 s > 5 s SLO
    _write_structured_log(
        tmp_path,
        [
            {"query_id": "q1", "query_text_id": "s#001#001", "arrival_s": 0.0, "completion_s": 10.0},
            {"query_id": "q2", "query_text_id": "s#001#002", "arrival_s": 0.0, "completion_s": 10.0},
        ],
    )

    r = ExecutionResult.load(tmp_path)

    assert r.violation_rate == pytest.approx(1.0)
    assert r.violation_amount_s == pytest.approx(5.0)  # mean(10-5, 10-5)
    assert r.violation_relative_mean == pytest.approx(1.0)  # mean((10-5)/5, ...)


def test_load_simulation_empty_log(tmp_path: Path) -> None:
    _write_execution_config(tmp_path)
    _write_billing(tmp_path, total_billed_cost=0.0)
    # Write an empty structured log (no ARRIVAL/COMPLETION rows)
    pd.DataFrame(
        columns=["query_id", "query_text_id", "rel_time_s", "event_type", "source", "wall_clock_s"]
    ).to_parquet(tmp_path / "structured_log.parquet", index=False)

    r = ExecutionResult.load(tmp_path)

    assert r.num_queries == 0
    assert r.violation_rate == pytest.approx(0.0)


def test_load_simulation_no_log_file(tmp_path: Path) -> None:
    _write_execution_config(tmp_path)
    _write_billing(tmp_path, total_billed_cost=2.0)
    # No structured_log.parquet at all — should not raise, just zero violations

    r = ExecutionResult.load(tmp_path)

    assert r.num_queries == 0
    assert r.total_cost == pytest.approx(2.0)


def test_load_simulation_cost_sums_multiple_clusters(tmp_path: Path) -> None:
    _write_execution_config(tmp_path)
    data = {
        "cluster_a": {"total_billed_cost": 1.0},
        "cluster_b": {"total_billed_cost": 2.5},
    }
    with open(tmp_path / "billing_interval_analysis.yml", "w") as f:
        yaml.dump(data, f)

    r = ExecutionResult.load(tmp_path)

    assert r.total_cost == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# ExecutionResult.load — live run: cost
# ---------------------------------------------------------------------------


def test_load_live_run_missing_usage_raises(tmp_path: Path) -> None:
    _write_execution_config(tmp_path)
    # No sys_serverless_usage file — should raise

    with pytest.raises(FileNotFoundError, match="sys_serverless_usage"):
        ExecutionResult.load(tmp_path)


def test_load_live_run_cost_from_usage(tmp_path: Path) -> None:
    from autoslo.clusters.cluster import Cluster

    _write_execution_config(tmp_path)
    charged_s = 3600.0  # 1 RPU-hour
    _write_sys_serverless_usage(tmp_path, charged_seconds=charged_s)
    _write_structured_log(
        tmp_path,
        [{"query_id": "q1", "query_text_id": "s#001#001", "arrival_s": 0.0, "completion_s": 1.0}],
    )

    r = ExecutionResult.load(tmp_path)

    expected_cost = charged_s / 3600 * Cluster.US_EAST_1_COST_PER_RPU_HOUR
    assert r.total_cost == pytest.approx(expected_cost)


def test_load_live_run_cost_sums_multiple_clusters(tmp_path: Path) -> None:
    from autoslo.clusters.cluster import Cluster

    _write_execution_config(tmp_path)
    _write_sys_serverless_usage(tmp_path, charged_seconds=1800.0, cluster_name="c1")
    _write_sys_serverless_usage(tmp_path, charged_seconds=1800.0, cluster_name="c2")
    _write_structured_log(
        tmp_path,
        [{"query_id": "q1", "query_text_id": "s#001#001", "arrival_s": 0.0, "completion_s": 1.0}],
    )

    r = ExecutionResult.load(tmp_path)

    expected_cost = 3600.0 / 3600 * Cluster.US_EAST_1_COST_PER_RPU_HOUR
    assert r.total_cost == pytest.approx(expected_cost)


# ---------------------------------------------------------------------------
# ExecutionResult.load — live run: aborted queries (warning only)
# ---------------------------------------------------------------------------


def test_load_live_run_aborted_queries_not_counted_as_violations(
    tmp_path: Path, capsys
) -> None:
    """Aborted queries trigger a printed warning but do NOT affect violation metrics."""
    _write_execution_config(tmp_path)
    _write_sys_serverless_usage(tmp_path, charged_seconds=0.0)

    # One successful query well within SLO (1 s vs 5 s)
    _write_structured_log(
        tmp_path,
        [{"query_id": "q_ok", "query_text_id": "s#001#001", "arrival_s": 0.0, "completion_s": 1.0}],
    )
    # One aborted query visible only in sys_query_history
    _write_sys_query_history(
        tmp_path,
        [{"query_id": "q_abort", "status": "failed", "elapsed_time": 1000}],
    )

    r = ExecutionResult.load(tmp_path)

    # Only the completed query is counted; the aborted one is not in the metrics
    assert r.num_queries == 1
    assert r.violation_rate == pytest.approx(0.0)
    # A warning must be printed
    assert "aborted" in capsys.readouterr().out.lower()


def test_load_live_run_no_aborted_queries_no_warning(
    tmp_path: Path, capsys
) -> None:
    """When all queries succeed, no warning is printed."""
    _write_execution_config(tmp_path)
    _write_sys_serverless_usage(tmp_path, charged_seconds=0.0)
    _write_structured_log(
        tmp_path,
        [{"query_id": "q1", "query_text_id": "s#001#001", "arrival_s": 0.0, "completion_s": 1.0}],
    )
    _write_sys_query_history(
        tmp_path,
        [{"query_id": "q1", "status": "success", "elapsed_time": 1000000}],
    )

    r = ExecutionResult.load(tmp_path)

    assert r.num_queries == 1
    assert r.violation_rate == pytest.approx(0.0)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Trace.aborted_query_ids_from_dir
# ---------------------------------------------------------------------------


def test_aborted_query_ids_no_files(tmp_path: Path) -> None:
    result = Trace.aborted_query_ids_from_dir(tmp_path)
    assert result == set()


def test_aborted_query_ids_all_success(tmp_path: Path) -> None:
    _write_sys_query_history(
        tmp_path,
        [
            {"query_id": "q1", "status": "success"},
            {"query_id": "q2", "status": "success   "},  # trailing whitespace OK
        ],
    )
    result = Trace.aborted_query_ids_from_dir(tmp_path)
    assert result == set()


def test_aborted_query_ids_mixed(tmp_path: Path) -> None:
    _write_sys_query_history(
        tmp_path,
        [
            {"query_id": "q1", "status": "success"},
            {"query_id": "q2", "status": "failed"},
            {"query_id": "q3", "status": "error"},
        ],
    )
    result = Trace.aborted_query_ids_from_dir(tmp_path)
    assert result == {"q2", "q3"}


def test_aborted_query_ids_multiple_clusters(tmp_path: Path) -> None:
    _write_sys_query_history(
        tmp_path,
        [{"query_id": "q1", "status": "success"}, {"query_id": "q2", "status": "failed"}],
        cluster_name="c1",
    )
    _write_sys_query_history(
        tmp_path,
        [{"query_id": "q3", "status": "failed"}],
        cluster_name="c2",
    )
    result = Trace.aborted_query_ids_from_dir(tmp_path)
    assert result == {"q2", "q3"}
