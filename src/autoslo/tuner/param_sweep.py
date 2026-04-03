"""Shared grid-sweep logic for autoscaler and routing parameter tuning.

Both the autoscaler sweep (design step 5) and the routing sweep (design
step 6) follow the same pattern: build a grid, evaluate each point on
the training set, compute the Pareto front over (violation, cost),
validate Pareto-optimal points, and select the best.  :class:`ParamSweep`
encapsulates this logic.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from autoslo.tuner.config import TunerConfig
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.tuner_utils import (
    AggregatedMetrics,
    PhaseResult,
    SloObjective,
    aggregate,
    compute_pareto_front,
    primary_violation,
    threshold_aware_select,
)
import autoslo.utils.config as cfgu

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def build_grid(param_ranges: dict[str, list]) -> list[dict[str, Any]]:
    """Cartesian product of *param_ranges* as a list of dicts.

    >>> build_grid({"a": [1, 2], "b": ["x"]})
    [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}]
    """
    if not param_ranges:
        return [{}]
    keys = list(param_ranges.keys())
    return [
        dict(zip(keys, vals))
        for vals in itertools.product(*param_ranges.values())
    ]


# ---------------------------------------------------------------------------
# ParamSweep
# ---------------------------------------------------------------------------


class ParamSweep:
    """Grid search with Pareto-front analysis and validation-set selection.

    Parameters
    ----------
    evaluator :
        Shared scenario evaluator for running simulations.
    config :
        Configuration for this tuner run, including the aggregation metric and
        other hyperparameters.
    base_overrides :
        Config overrides that are applied to *every* grid point (e.g.
        optimised checkpoints from a previous phase).
    run_dir :
        Root directory for this tuner run.
    phase_name :
        Label for the current phase (e.g. ``"autoscaler"`` or
        ``"routing"``).  Used for directory and log naming.
    """

    def __init__(
        self,
        evaluator: ScenarioEvaluator,
        config: dict[str, Any],
        base_overrides: dict[str, Any],
        run_dir: Path,
        phase_name: str,
        slo_objective: SloObjective,
    ) -> None:
        self._evaluator = evaluator
        self._config = config
        self._base_overrides = base_overrides
        self._run_dir = run_dir
        self._phase_name = phase_name
        self._slo_objective = slo_objective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sweep(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        param_ranges: dict[str, list],
    ) -> dict[str, Any]:
        """Run a full grid sweep and return the best parameter dict.

        Parameters
        ----------
        train_paths :
            Parquet workload paths for training scenarios.
        val_paths :
            Parquet workload paths for validation scenarios.
        param_ranges :
            Parameter names → candidate values.  The Cartesian product
            forms the grid.

        Returns
        -------
        The best parameter dict (keys without the section prefix).
        """
        metric = cfgu.cfg_getd(
            self._config,
            "tuner_config.forecast_config.aggregation_metric",
            "p90",
        )
        grid = build_grid(param_ranges)
        phase_dir = self._run_dir / self._phase_name

        # ── Pre-flight summary ──────────────────────────────────────
        self._print_preflight(param_ranges, grid, len(train_paths))

        # ── Training sweep ──────────────────────────────────────────
        grid_results: list[dict[str, Any]] = []

        for gp_idx, point in enumerate(grid):
            overrides = dict(self._base_overrides)
            overrides = cfgu.apply_overrides(overrides, point)

            train_results = self._evaluator.evaluate(
                workload_paths=train_paths,
                config_overrides=overrides,
                phase=self._phase_name,
                grid_point=gp_idx,
                out_subdir=phase_dir / f"grid_point_{gp_idx:03d}" / "train",
            )
            train_agg = aggregate(train_results, metric)
            train_primary = primary_violation(
                train_agg, self._slo_objective.slo_metric
            )

            grid_results.append(
                {
                    "grid_point": gp_idx,
                    "params": point,
                    "train_violation_agg": train_primary,
                    "train_cost_agg": train_agg.cost,
                    "train_metrics": train_agg,
                    "is_pareto": False,
                    "val_violation_agg": None,
                    "val_cost_agg": None,
                    "val_metrics": None,
                }
            )

            slo_label = self._slo_objective.slo_metric
            slo_thresh = self._slo_objective.slo_threshold
            console.print(
                f"  [dim]gp {gp_idx:>3d}/{len(grid)}[/]  "
                f"{slo_label}={train_primary:.4f} (threshold={slo_thresh:.2f})  "
                f"cost=${train_agg.cost:.4f}  "
                f"params={point}"
            )

        # ── Pareto front ───────────────────────────────────────────
        points = [
            (r["train_violation_agg"], r["train_cost_agg"])
            for r in grid_results
        ]
        pareto_indices = compute_pareto_front(points)
        for idx in pareto_indices:
            grid_results[idx]["is_pareto"] = True

        console.print(
            f"\n  [cyan]Pareto front:[/] {len(pareto_indices)} of "
            f"{len(grid)} points"
        )

        # ── Validate Pareto-optimal points ─────────────────────────
        for idx in pareto_indices:
            point = grid_results[idx]["params"]
            overrides = dict(self._base_overrides)
            overrides = cfgu.apply_overrides(overrides, point)

            val_results = self._evaluator.evaluate(
                workload_paths=val_paths,
                config_overrides=overrides,
                phase=self._phase_name,
                grid_point=f"{idx}_val",
                out_subdir=phase_dir / f"grid_point_{idx:03d}" / "val",
            )
            val_agg = aggregate(val_results, metric)
            val_primary = primary_violation(
                val_agg, self._slo_objective.slo_metric
            )
            grid_results[idx]["val_violation_agg"] = val_primary
            grid_results[idx]["val_cost_agg"] = val_agg.cost
            grid_results[idx]["val_metrics"] = val_agg

        # ── Select best ───────────────────────────────────────────
        best_idx = self._select_best(grid_results, pareto_indices)
        best_params = grid_results[best_idx]["params"]

        # ── Rich summary table ─────────────────────────────────────
        self._print_pareto_table(grid_results, pareto_indices, best_idx)

        # ── Persist results ────────────────────────────────────────
        self._write_sweep_results(phase_dir, grid_results, best_idx)

        return best_params

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def _select_best(
        self,
        grid_results: list[dict[str, Any]],
        pareto_indices: list[int],
    ) -> int:
        """Pick the best Pareto-optimal point by threshold-aware selection.

        Strategy: among validated Pareto points, apply lexicographic
        (feasibility-first, then cheapest) selection.
        """
        # Among Pareto points that have been validated.
        validated = [
            idx
            for idx in pareto_indices
            if grid_results[idx]["val_violation_agg"] is not None
        ]
        if not validated:
            # Fallback: pick the Pareto point with lowest training violation.
            return min(
                pareto_indices,
                key=lambda i: (
                    grid_results[i]["train_violation_agg"],
                    grid_results[i]["train_cost_agg"],
                ),
            )
        candidates = [
            (
                grid_results[i]["val_violation_agg"],
                grid_results[i]["val_cost_agg"],
            )
            for i in validated
        ]
        best_local_idx = threshold_aware_select(
            candidates,
            self._slo_objective.slo_threshold,
        )
        return validated[best_local_idx]

    # ------------------------------------------------------------------
    # Rich output
    # ------------------------------------------------------------------

    def _print_preflight(
        self,
        param_ranges: dict[str, list],
        grid: list[dict[str, Any]],
        n_train: int,
    ) -> None:
        console.rule(f"[bold cyan]{self._phase_name} sweep")

        table = Table(title="Parameter Ranges", show_lines=True)
        table.add_column("Parameter", justify="left")
        table.add_column("Values", justify="left")
        table.add_column("Count", justify="right")
        for name, values in param_ranges.items():
            table.add_row(name, str(values), str(len(values)))
        console.print(table)

        console.print(
            f"  Grid size: [bold]{len(grid)}[/] combinations  |  "
            f"Training scenarios: [bold]{n_train}[/]  |  "
            f"Total evaluations: [bold]{len(grid) * n_train}[/]"
        )

    def _print_pareto_table(
        self,
        grid_results: list[dict[str, Any]],
        pareto_indices: list[int],
        best_idx: int,
    ) -> None:
        table = Table(
            title=f"{self._phase_name} — Pareto Front Results",
            show_lines=True,
        )
        slo_label = self._slo_objective.slo_metric
        slo_thresh = self._slo_objective.slo_threshold
        table.add_column("GP", justify="right")
        table.add_column("Params", justify="left")
        table.add_column(
            f"Train {slo_label} (≤{slo_thresh:.2f})", justify="right"
        )
        table.add_column("Train Cost", justify="right")
        table.add_column(
            f"Val {slo_label} (≤{slo_thresh:.2f})", justify="right"
        )
        table.add_column("Val Cost", justify="right")
        table.add_column("Best", justify="center")

        for idx in pareto_indices:
            r = grid_results[idx]
            is_best = idx == best_idx
            style = "bold green" if is_best else ""
            val_v = (
                f"{r['val_violation_agg']:.4f}"
                if r["val_violation_agg"] is not None
                else "—"
            )
            val_c = (
                f"{r['val_cost_agg']:.4f}"
                if r["val_cost_agg"] is not None
                else "—"
            )
            table.add_row(
                str(idx),
                str(r["params"]),
                f"{r['train_violation_agg']:.4f}",
                f"{r['train_cost_agg']:.4f}",
                val_v,
                val_c,
                "✓" if is_best else "",
                style=style,
            )
        console.print(table)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _write_sweep_results(
        phase_dir: Path,
        grid_results: list[dict[str, Any]],
        best_idx: int,
    ) -> None:
        phase_dir.mkdir(parents=True, exist_ok=True)
        # Serialize grid_results, converting AggregatedMetrics to dicts.
        serializable = []
        for r in grid_results:
            entry = {
                k: v
                for k, v in r.items()
                if k not in ("train_metrics", "val_metrics")
            }
            tm = r.get("train_metrics")
            if tm is not None:
                entry["train_violation_rate"] = tm.violation_rate
                entry["train_violation_amount_s"] = tm.violation_amount_s
                entry["train_violation_relative_mean"] = (
                    tm.violation_relative_mean
                )
            vm = r.get("val_metrics")
            if vm is not None:
                entry["val_violation_rate"] = vm.violation_rate
                entry["val_violation_amount_s"] = vm.violation_amount_s
                entry["val_violation_relative_mean"] = (
                    vm.violation_relative_mean
                )
            serializable.append(entry)
        output = {
            "best_grid_point": best_idx,
            "best_params": grid_results[best_idx]["params"],
            "grid_results": serializable,
        }
        with open(phase_dir / "sweep_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
