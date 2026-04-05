"""Shared data types for the policy tuner."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from autoslo.blueprint_selection.slo_resolver import SloResolver


# ---------------------------------------------------------------------------
# Aggregated result for one config over one collection of workloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregatedSimulationResults:
    """All three violation metrics plus cost, aggregated across scenarios.

    Optionally stores the per-scenario :class:`SimulationResult` objects
    that were aggregated, so downstream code can inspect individual
    scenario values (e.g. for min–max range display).
    """

    violation_rate: float
    violation_amount_s: float
    violation_relative_mean: float
    cost: float
    scenario_results: tuple[SimulationResult, ...] = ()

@dataclass
class SimulationResult:
    """Metrics from a single simulation."""

    simulation_dir: Path
    violation_rate: float
    violation_amount_s: float
    violation_relative_mean: float
    total_cost: float
    num_queries: int

    @staticmethod
    def load(
        simulation_dir: str | Path,
    ) -> SimulationResult:
        """Load a SimulationResult from the output directory of a single
        simulation.

        Reads ``billing_interval_analysis.yml`` for cost and
        ``structured_log.parquet`` for violation statistics — the same logic
        used by :meth:`WorkloadSimulator._write_experiment_meta`.
        """
        simulation_dir = Path(simulation_dir)

        # -- build slo resolver for this scenario from its config.yml --
        config_path = simulation_dir / "config.yml"
        config: dict[str, Any] = {}
        with open(config_path) as f:
            config = yaml.safe_load(f)
        slo_resolver = SloResolver.from_config(config)

        # -- cost --
        total_cost = 0.0
        billing_path = simulation_dir / "billing_interval_analysis.yml"
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

        log_path = simulation_dir / "structured_log.parquet"
        if log_path.exists():
            log = pd.read_parquet(log_path)
            completions = log[log["event_type"] == "completion"].copy()
            num_queries = len(completions)
            if num_queries > 0:
                durations = completions["latency_s"].fillna(0.0)
                per_row_slo = (
                    completions["query_text_id"]
                    .map(slo_resolver.resolve)
                    .fillna(0.0)
                )
                violations = durations > per_row_slo
                violation_rate = float(violations.mean())
                violation_amount_s = float(
                    (durations - per_row_slo).clip(lower=0.0).sum()
                )
                relative = ((durations - per_row_slo) / per_row_slo).clip(
                    lower=0.0
                )
                violation_relative_mean = float(relative.mean())

        return SimulationResult(
            simulation_dir=simulation_dir,
            violation_rate=violation_rate,
            violation_amount_s=violation_amount_s,
            violation_relative_mean=violation_relative_mean,
            total_cost=total_cost,
            num_queries=num_queries,
        )

    @staticmethod
    def load_batch(
        batch_dir: str | Path,
    ) -> list[SimulationResult]:
        """Load all simulation results from the given directory."""
        batch_dir = Path(batch_dir)
        results: list[SimulationResult] = []
        for simulation_dir in batch_dir.iterdir():
            if simulation_dir.is_dir():
                result = SimulationResult.load(
                    simulation_dir=simulation_dir,
                )
                results.append(result)
        return results


    @staticmethod
    def aggregate(
        results: list[SimulationResult], metric: str = "p90"
    ) -> AggregatedSimulationResults:
        """Compute a summary statistic over scenario results.

        Parameters
        ----------
        results :
            Per-scenario results to aggregate.
        metric :
            ``"mean"``, ``"max"``, ``"p90"``, ``"p99"``, or any ``"pNN"`` quantile.

        Returns
        -------
        AggregatedSimulationResults with all three violation metrics and cost.
        """
        if not results:
            return AggregatedSimulationResults(
                violation_rate=0.0,
                violation_amount_s=0.0,
                violation_relative_mean=0.0,
                cost=0.0,
            )

        scenario_results = tuple(results)
        rates = [r.violation_rate for r in results]
        amounts = [r.violation_amount_s for r in results]
        relatives = [r.violation_relative_mean for r in results]
        costs = [r.total_cost for r in results]

        if metric == "mean":
            return AggregatedSimulationResults(
                violation_rate=statistics.mean(rates),
                violation_amount_s=statistics.mean(amounts),
                violation_relative_mean=statistics.mean(relatives),
                cost=statistics.mean(costs),
                scenario_results=scenario_results,
            )
        if metric == "max":
            return AggregatedSimulationResults(
                violation_rate=max(rates),
                violation_amount_s=max(amounts),
                violation_relative_mean=max(relatives),
                cost=max(costs),
                scenario_results=scenario_results,
            )

        # pNN quantile
        if metric.startswith("p") and metric[1:].isdigit():
            q = int(metric[1:]) / 100.0
            return AggregatedSimulationResults(
                violation_rate=float(np.quantile(rates, q)),
                violation_amount_s=float(np.quantile(amounts, q)),
                violation_relative_mean=float(np.quantile(relatives, q)),
                cost=float(np.quantile(costs, q)),
                scenario_results=scenario_results,
            )

        raise ValueError(f"Unknown aggregation metric: {metric!r}")




# ---------------------------------------------------------------------------
# SLO objective bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SloObjective:
    """Bundles the SLO metric name and feasibility threshold."""

    slo_metric: str  # "binary", "absolute_s", or "relative"
    slo_threshold: float


# ---------------------------------------------------------------------------
# Helpers for metric routing and threshold-aware selection
# ---------------------------------------------------------------------------

_METRIC_TO_FIELD = {
    "binary": "violation_rate",
    "absolute_s": "violation_amount_s",
    "relative": "violation_relative_mean",
}


def primary_violation(agg: AggregatedSimulationResults, slo_metric: str) -> float:
    """Extract the primary violation value for the given SLO metric."""
    field_name = _METRIC_TO_FIELD.get(slo_metric)
    if field_name is None:
        raise ValueError(f"Unknown slo_metric: {slo_metric!r}")
    return getattr(agg, field_name)


def is_feasible(primary_val: float, slo_threshold: float) -> bool:
    """Return True if *primary_val* satisfies the SLO threshold."""
    return primary_val <= slo_threshold


def threshold_aware_select(
    candidates: list[tuple[float, float]],
    slo_threshold: float,
) -> int:
    """Return the index of the best candidate under lexicographic selection.

    1. Partition into feasible (primary ≤ threshold) and infeasible.
    2. If any feasible: return the one with lowest cost.
    3. If none feasible: return the one with lowest primary violation
       (tiebreak on cost).
    """
    feasible = [
        (i, pv, cost)
        for i, (pv, cost) in enumerate(candidates)
        if pv <= slo_threshold
    ]
    if feasible:
        return min(feasible, key=lambda t: t[2])[0]
    # None feasible — pick lowest primary violation, tiebreak cost.
    return min(
        range(len(candidates)),
        key=lambda i: (candidates[i][0], candidates[i][1]),
    )




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
