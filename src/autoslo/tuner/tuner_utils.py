"""Shared data types for the policy tuner."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from autoslo.slo.slo_metric import LatencySlo, SloMetric
from autoslo.slo.slo_resolver import SloResolver

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

    def primary_violation(self, slo_metric: str | SloMetric) -> float:
        """Extract the primary violation value for the given SLO metric."""
        slo_metric_obj = (
            slo_metric
            if isinstance(slo_metric, SloMetric)
            else SloMetric(slo_metric)
        )

        if slo_metric_obj == SloMetric.BINARY:
            return self.violation_rate
        elif slo_metric_obj == SloMetric.ABSOLUTE_S:
            return self.violation_amount_s
        elif slo_metric_obj == SloMetric.RELATIVE:
            return self.violation_relative_mean
        else:
            raise ValueError(f"Unknown SLO metric: {slo_metric}")

    @staticmethod
    def _fmt_cell(
        agg_val: float,
        scenario_vals: list[float],
    ) -> str:
        """Format a metric cell: aggregated value with dim min–max range."""
        main = f"{agg_val:.4f}"
        if len(scenario_vals) >= 2:
            lo, hi = min(scenario_vals), max(scenario_vals)
            main += f"\n[dim]{lo:.4f} … {hi:.4f}[/dim]"
        return main

    @staticmethod
    def print_comparison(
        *entries: tuple[str, AggregatedSimulationResults],
        console: Console,
        agg_metric: str = "p90",
        slo_metric: str | SloMetric = "binary",
        highlight_best: bool = True,
    ) -> None:
        """Print a table comparing multiple AggregatedSimulationResults.

        Parameters
        ----------
        *entries :
            ``(label, agg)`` pairs.  Each gets one row.  Use labels like
            ``"Initial (train)"`` / ``"Initial (val)"`` to distinguish splits.
        agg_metric :
            Aggregation metric shown in the title.
        slo_metric :
            The SLO metric that was actually optimised (``"binary"``,
            ``"absolute_s"``, or ``"relative"``).  The best cell in this
            column and in the Cost column is highlighted green; the best
            cell in the other two violation columns is highlighted yellow.
        highlight_best :
            Whether to highlight the best values in the table.
        """
        # Map slo_metric → column index (0-2 are the three violation metrics).
        slo_metric_str = (
            slo_metric.value
            if isinstance(slo_metric, SloMetric)
            else slo_metric
        )
        _SLO_TO_COL = {"binary": 0, "absolute_s": 1, "relative": 2}
        targeted_col = _SLO_TO_COL.get(slo_metric_str, 0)
        cost_col = 3  # always

        table = Table(
            title=f"Comparison  [dim](agg: {agg_metric})[/dim]",
            show_lines=True,
        )
        table.add_column("Config", justify="left")

        metric_labels = [
            "Viol. Rate",
            "Viol. Amt (s)",
            "Viol. Rel.",
            "Cost ($)",
        ]
        for ml in metric_labels:
            table.add_column(ml, justify="right")

        fmt = AggregatedSimulationResults._fmt_cell

        def _extract(
            agg: AggregatedSimulationResults,
        ) -> tuple[list[float], list[list[float]]]:
            aggs = [
                agg.violation_rate,
                agg.violation_amount_s,
                agg.violation_relative_mean,
                agg.cost,
            ]
            if agg.scenario_results:
                per = [
                    [r.violation_rate for r in agg.scenario_results],
                    [r.violation_amount_s for r in agg.scenario_results],
                    [r.violation_relative_mean for r in agg.scenario_results],
                    [r.total_cost for r in agg.scenario_results],
                ]
            else:
                per = [[], [], [], []]
            return aggs, per

        # Build row data.
        row_data: list[tuple[str, list[float], list[list[float]]]] = []
        for label, agg in entries:
            a, p = _extract(agg)
            row_data.append((label, a, p))

        # Find the best (lowest) aggregated value per column.
        n_metric = 4
        best_per_col: list[int] = []
        if row_data:
            for c in range(n_metric):
                best_per_col.append(
                    min(
                        range(len(row_data)),
                        key=lambda i, _c=c: row_data[i][1][_c],  # type: ignore
                    )
                )

        for row_idx, (label, aggs, per) in enumerate(row_data):
            cells: list[str] = []
            for c in range(n_metric):
                cell = fmt(aggs[c], per[c])
                if (
                    len(row_data) > 1
                    and (row_idx == best_per_col[c])
                    and highlight_best
                ):
                    if c == targeted_col or c == cost_col:
                        cell = f"[green]{cell}[/green]"
                    else:
                        cell = f"[yellow]{cell}[/yellow]"
                cells.append(cell)
            table.add_row(label, *cells)

        console.print(table)


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
            log = pd.read_parquet(
                log_path,
                columns=[
                    "rel_time_s",
                    "event_type",
                    "query_id",
                    "query_text_id",
                ],
            )
            log = log[log["event_type"].isin({"arrival", "completion"})]
            if not log.empty:

                pivoted = log.pivot(
                    index=["query_id", "query_text_id"],
                    columns="event_type",
                    values="rel_time_s",
                )
                latencies = (
                    pivoted["completion"] - pivoted["arrival"]
                ).tolist()
                per_row_slo = (
                    pivoted.index.get_level_values("query_text_id")
                    .map(slo_resolver.resolve)
                    .fillna(0.0)
                )

                ## TODO: Deal with failed queries. Not super needed here because
                ## in the sumulator all queries succeed, but needed in principle.
                lat_and_slos = [
                    LatencySlo(lat, slo)
                    for lat, slo in zip(latencies, per_row_slo)
                ]
                violation_rate = SloMetric.BINARY.aggregate_batch(lat_and_slos)
                violation_amount_s = SloMetric.ABSOLUTE_S.aggregate_batch(
                    lat_and_slos
                )
                violation_relative_mean = SloMetric.RELATIVE.aggregate_batch(
                    lat_and_slos
                )

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
# Helpers for metric routing and threshold-aware selection
# ---------------------------------------------------------------------------


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
