from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
from rich.table import Table

from autoslo.slo.slo_metric import SloMetric
from autoslo.workload_execution.execution_result import ExecutionResult


@dataclass(frozen=True)
class AggregatedExecutionResults:
    """All three violation metrics plus cost, aggregated across scenarios.

    Optionally stores the per-scenario :class:`ExecutionResult` objects
    that were aggregated, so downstream code can inspect individual
    scenario values (e.g. for min–max range display).
    """

    violation_rate: float
    violation_amount_s: float
    violation_relative_mean: float
    cost: float
    scenario_results: list[ExecutionResult]

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
        *entries: tuple[str, AggregatedExecutionResults],
        console: Any,
        agg_method: str = "p90",
        slo_metric: str | SloMetric = "binary",
        highlight_best: bool = True,
    ) -> None:
        """Print a table comparing multiple AggregatedExecutionResults.

        Parameters
        ----------
        *entries :
            ``(label, agg)`` pairs.  Each gets one row.  Use labels like
            ``"Initial (train)"`` / ``"Initial (val)"`` to distinguish splits.
        agg_method :
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
            title=f"Comparison  [dim](agg: {agg_method})[/dim]",
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

        fmt = AggregatedExecutionResults._fmt_cell

        def _extract(
            agg: AggregatedExecutionResults,
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

    @staticmethod
    def aggregate_from(
        results: list[ExecutionResult], metric: str = "p90"
    ) -> AggregatedExecutionResults:
        """Compute a summary statistic over scenario results.

        Parameters
        ----------
        results :
            Per-scenario results to aggregate.
        metric :
            ``"mean"``, ``"max"``, or any ``"pNN"`` quantile.

        Returns
        -------
        AggregatedExecutionResults with all three violation metrics and cost.
        """
        if not results:
            return AggregatedExecutionResults(
                violation_rate=0.0,
                violation_amount_s=0.0,
                violation_relative_mean=0.0,
                cost=0.0,
                scenario_results=[],
            )

        rates = [r.violation_rate for r in results]
        amounts = [r.violation_amount_s for r in results]
        relatives = [r.violation_relative_mean for r in results]
        costs = [r.total_cost for r in results]

        if metric == "mean":
            return AggregatedExecutionResults(
                violation_rate=statistics.mean(rates),
                violation_amount_s=statistics.mean(amounts),
                violation_relative_mean=statistics.mean(relatives),
                cost=statistics.mean(costs),
                scenario_results=results,
            )
        if metric == "max":
            return AggregatedExecutionResults(
                violation_rate=max(rates),
                violation_amount_s=max(amounts),
                violation_relative_mean=max(relatives),
                cost=max(costs),
                scenario_results=results,
            )

        # pNN quantile
        if metric.startswith("p") and metric[1:].isdigit():
            q = int(metric[1:]) / 100.0
            return AggregatedExecutionResults(
                violation_rate=float(np.quantile(rates, q)),
                violation_amount_s=float(np.quantile(amounts, q)),
                violation_relative_mean=float(np.quantile(relatives, q)),
                cost=float(np.quantile(costs, q)),
                scenario_results=results,
            )

        raise ValueError(f"Unknown aggregation metric: {metric!r}")
