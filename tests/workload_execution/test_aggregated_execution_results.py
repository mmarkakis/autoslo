"""Tests for AggregatedExecutionResults."""
from __future__ import annotations

from pathlib import Path

import pytest

from autoslo.workload_execution.aggregated_execution_results import (
    AggregatedExecutionResults,
)
from autoslo.workload_execution.execution_result import ExecutionResult
from autoslo.slo.slo_metric import SloMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    violation_rate: float = 0.0,
    violation_amount_s: float = 0.0,
    violation_relative_mean: float = 0.0,
    cost: float = 0.0,
    num_queries: int = 10,
) -> ExecutionResult:
    return ExecutionResult(
        execution_dir=Path("/fake/dir"),
        violation_rate=violation_rate,
        violation_amount_s=violation_amount_s,
        violation_relative_mean=violation_relative_mean,
        total_cost=cost,
        num_queries=num_queries,
    )


# ---------------------------------------------------------------------------
# aggregate_from — empty list
# ---------------------------------------------------------------------------


def test_aggregate_from_empty_returns_zeros() -> None:
    agg = AggregatedExecutionResults.aggregate_from([])
    assert agg.violation_rate == pytest.approx(0.0)
    assert agg.violation_amount_s == pytest.approx(0.0)
    assert agg.violation_relative_mean == pytest.approx(0.0)
    assert agg.cost == pytest.approx(0.0)
    assert agg.scenario_results == []


# ---------------------------------------------------------------------------
# aggregate_from — mean
# ---------------------------------------------------------------------------


def test_aggregate_mean() -> None:
    results = [
        _make_result(violation_rate=0.2, violation_amount_s=1.0, violation_relative_mean=0.5, cost=1.0),
        _make_result(violation_rate=0.4, violation_amount_s=3.0, violation_relative_mean=1.5, cost=3.0),
    ]
    agg = AggregatedExecutionResults.aggregate_from(results, metric="mean")
    assert agg.violation_rate == pytest.approx(0.3)
    assert agg.violation_amount_s == pytest.approx(2.0)
    assert agg.violation_relative_mean == pytest.approx(1.0)
    assert agg.cost == pytest.approx(2.0)
    assert agg.scenario_results is results


# ---------------------------------------------------------------------------
# aggregate_from — max
# ---------------------------------------------------------------------------


def test_aggregate_max() -> None:
    results = [
        _make_result(violation_rate=0.1, violation_amount_s=1.0, violation_relative_mean=0.2, cost=10.0),
        _make_result(violation_rate=0.3, violation_amount_s=5.0, violation_relative_mean=0.8, cost=5.0),
        _make_result(violation_rate=0.2, violation_amount_s=3.0, violation_relative_mean=0.5, cost=8.0),
    ]
    agg = AggregatedExecutionResults.aggregate_from(results, metric="max")
    assert agg.violation_rate == pytest.approx(0.3)
    assert agg.violation_amount_s == pytest.approx(5.0)
    assert agg.violation_relative_mean == pytest.approx(0.8)
    assert agg.cost == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# aggregate_from — p90
# ---------------------------------------------------------------------------


def test_aggregate_p90() -> None:
    """p90 over [0.0, 0.1, 0.2, ..., 0.9] should be 0.81 (numpy linear interp)."""
    import numpy as np

    values = [i / 10 for i in range(10)]
    results = [_make_result(violation_rate=v, cost=v) for v in values]
    agg = AggregatedExecutionResults.aggregate_from(results, metric="p90")
    expected = float(np.quantile(values, 0.9))
    assert agg.violation_rate == pytest.approx(expected)
    assert agg.cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# aggregate_from — pNN (p50 = median)
# ---------------------------------------------------------------------------


def test_aggregate_p50_is_median() -> None:
    import numpy as np

    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = [_make_result(violation_rate=v) for v in values]
    agg = AggregatedExecutionResults.aggregate_from(results, metric="p50")
    expected = float(np.quantile(values, 0.50))
    assert agg.violation_rate == pytest.approx(expected)


# ---------------------------------------------------------------------------
# aggregate_from — unknown metric
# ---------------------------------------------------------------------------


def test_aggregate_unknown_metric_raises() -> None:
    results = [_make_result()]
    with pytest.raises(ValueError, match="Unknown aggregation metric"):
        AggregatedExecutionResults.aggregate_from(results, metric="average")


# ---------------------------------------------------------------------------
# primary_violation
# ---------------------------------------------------------------------------


def test_primary_violation_binary() -> None:
    agg = AggregatedExecutionResults.aggregate_from(
        [_make_result(violation_rate=0.25)], metric="mean"
    )
    assert agg.primary_violation(SloMetric.BINARY) == pytest.approx(0.25)
    assert agg.primary_violation("binary") == pytest.approx(0.25)


def test_primary_violation_absolute_s() -> None:
    agg = AggregatedExecutionResults.aggregate_from(
        [_make_result(violation_amount_s=3.5)], metric="mean"
    )
    assert agg.primary_violation(SloMetric.ABSOLUTE_S) == pytest.approx(3.5)
    assert agg.primary_violation("absolute_s") == pytest.approx(3.5)


def test_primary_violation_relative() -> None:
    agg = AggregatedExecutionResults.aggregate_from(
        [_make_result(violation_relative_mean=0.7)], metric="mean"
    )
    assert agg.primary_violation(SloMetric.RELATIVE) == pytest.approx(0.7)
    assert agg.primary_violation("relative") == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# print_comparison — smoke test (no assertion on output, just no crash)
# ---------------------------------------------------------------------------


def test_print_comparison_smoke() -> None:
    from io import StringIO
    from rich.console import Console

    results_a = [_make_result(violation_rate=0.1, cost=1.0)]
    results_b = [_make_result(violation_rate=0.2, cost=2.0)]
    agg_a = AggregatedExecutionResults.aggregate_from(results_a, metric="mean")
    agg_b = AggregatedExecutionResults.aggregate_from(results_b, metric="mean")

    buf = StringIO()
    console = Console(file=buf, highlight=False, no_color=True)
    AggregatedExecutionResults.print_comparison(
        ("Config A", agg_a),
        ("Config B", agg_b),
        console=console,
        agg_method="mean",
        slo_metric="binary",
    )
    output = buf.getvalue()
    assert "Config A" in output
    assert "Config B" in output
    assert "Viol. Rate" in output
