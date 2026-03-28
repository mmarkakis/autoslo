"""Shared data types for the policy tuner."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from autoslo.blueprint_selection.slo_resolver import SloResolver

# ---------------------------------------------------------------------------
# Per-scenario result
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Metrics from a single ``simulate_one`` run."""

    scenario_idx: int
    violation_rate: float
    violation_amount_s: float
    violation_relative_mean: float
    total_cost: float
    num_queries: int
    out_dir: Path


# ---------------------------------------------------------------------------
# Aggregated result for one parameter combination
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Aggregated metrics across scenarios for one parameter combination."""

    params: dict = field(default_factory=dict)
    train_results: list[ScenarioResult] = field(default_factory=list)
    val_results: list[ScenarioResult] | None = None
    train_violation_agg: float = 0.0
    train_cost_agg: float = 0.0
    val_violation_agg: float | None = None
    val_cost_agg: float | None = None


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def aggregate(
    results: list[ScenarioResult], metric: str = "p90"
) -> tuple[float, float]:
    """Compute a summary statistic over scenario results.

    Parameters
    ----------
    results :
        Per-scenario results to aggregate.
    metric :
        ``"mean"``, ``"p90"``, ``"p99"``, or any ``"pNN"`` quantile.

    Returns
    -------
    (violation_agg, cost_agg)
    """
    if not results:
        return (0.0, 0.0)

    violations = [r.violation_rate for r in results]
    costs = [r.total_cost for r in results]

    if metric == "mean":
        return (statistics.mean(violations), statistics.mean(costs))

    # pNN quantile
    if metric.startswith("p") and metric[1:].isdigit():
        q = int(metric[1:]) / 100.0
        return (
            float(np.quantile(violations, q)),
            float(np.quantile(costs, q)),
        )

    raise ValueError(f"Unknown aggregation metric: {metric!r}")


# ---------------------------------------------------------------------------
# Pareto front computation
# ---------------------------------------------------------------------------


def compute_pareto_front(
    points: list[tuple[float, float]],
) -> list[int]:
    """Return indices of Pareto-optimal points (both objectives minimised).

    Parameters
    ----------
    points :
        List of ``(violation, cost)`` pairs.

    Returns
    -------
    Sorted list of indices into *points* that lie on the Pareto front.
    """
    if not points:
        return []

    # Sort by first objective; break ties by second.
    indexed = sorted(enumerate(points), key=lambda t: (t[1][0], t[1][1]))
    front: list[int] = []
    best_cost = float("inf")
    for idx, (_, cost) in indexed:
        if cost <= best_cost:
            front.append(idx)
            best_cost = cost
    front.sort()
    return front


# ---------------------------------------------------------------------------
# Result extraction from simulator output
# ---------------------------------------------------------------------------


def extract_scenario_result(
    out_dir: str | Path,
    scenario_idx: int,
    slo_s: float,
    slo_dict: dict[str, float] | None = None,
) -> ScenarioResult:
    """Build a :class:`ScenarioResult` from files written by ``simulate_one``.

    Reads ``billing_interval_analysis.yml`` for cost and
    ``structured_log.parquet`` for violation statistics — the same logic
    used by :meth:`WorkloadSimulator._write_experiment_meta`.
    """

    out_dir = Path(out_dir)

    # -- cost --
    total_cost = 0.0
    billing_path = out_dir / "billing_interval_analysis.yml"
    if billing_path.exists():
        with open(billing_path) as f:
            billing: dict[str, Any] = yaml.safe_load(f) or {}
        for cluster_data in billing.values():
            total_cost += cluster_data.get("total_billed_cost", 0.0)

    # -- violations --
    violation_rate = 0.0
    violation_amount_s = 0.0
    violation_relative_mean = 0.0
    num_queries = 0

    log_path = out_dir / "structured_log.parquet"
    if log_path.exists():
        log = pd.read_parquet(log_path)
        completions = log[log["event_type"] == "completion"].copy()
        num_queries = len(completions)
        if num_queries > 0:
            resolver = SloResolver.from_dict(slo_s, slo_dict or {})
            durations = completions["latency_s"].fillna(0.0)
            per_row_slo = (
                completions["query_text_id"].map(resolver.resolve).fillna(slo_s)
            )
            violations = durations > per_row_slo
            violation_rate = float(violations.mean())
            violation_amount_s = float(
                (durations - per_row_slo).clip(lower=0.0).sum()
            )
            relative = ((durations - per_row_slo) / per_row_slo).clip(lower=0.0)
            violation_relative_mean = float(relative.mean())

    return ScenarioResult(
        scenario_idx=scenario_idx,
        violation_rate=violation_rate,
        violation_amount_s=violation_amount_s,
        violation_relative_mean=violation_relative_mean,
        total_cost=total_cost,
        num_queries=num_queries,
        out_dir=out_dir,
    )
