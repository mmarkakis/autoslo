"""Shared data types for the policy tuner."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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
