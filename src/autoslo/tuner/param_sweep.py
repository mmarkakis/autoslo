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
from autoslo.tuner.types import PhaseResult, aggregate, compute_pareto_front

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
    tuner_config :
        Tuner hyper-parameters (aggregation metric, etc.).
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
        tuner_config: TunerConfig,
        base_overrides: dict[str, Any],
        run_dir: Path,
        phase_name: str,
    ) -> None:
        self._evaluator = evaluator
        self._tuner_config = tuner_config
        self._base_overrides = base_overrides
        self._run_dir = run_dir
        self._phase_name = phase_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sweep(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        param_ranges: dict[str, list],
        config_section: str,
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
        config_section :
            Config section prefix for dot-key overrides (e.g.
            ``"autoscaling_config"`` or ``"routing_config"``).

        Returns
        -------
        The best parameter dict (keys without the section prefix).
        """
        metric = self._tuner_config.aggregation_metric
        grid = build_grid(param_ranges)
        phase_dir = self._run_dir / self._phase_name

        # ── Pre-flight summary ──────────────────────────────────────
        self._print_preflight(param_ranges, grid, len(train_paths))

        # ── Training sweep ──────────────────────────────────────────
        grid_results: list[dict[str, Any]] = []

        for gp_idx, point in enumerate(grid):
            overrides = dict(self._base_overrides)
            for k, v in point.items():
                overrides[f"{config_section}.{k}"] = v

            train_results = self._evaluator.evaluate(
                workload_paths=train_paths,
                config_overrides=overrides,
                phase=self._phase_name,
                grid_point=gp_idx,
                out_subdir=phase_dir / f"grid_point_{gp_idx:03d}" / "train",
            )
            viol_agg, cost_agg = aggregate(train_results, metric)

            grid_results.append(
                {
                    "grid_point": gp_idx,
                    "params": point,
                    "train_violation_agg": viol_agg,
                    "train_cost_agg": cost_agg,
                    "is_pareto": False,
                    "val_violation_agg": None,
                    "val_cost_agg": None,
                }
            )

            console.print(
                f"  [dim]gp {gp_idx:>3d}/{len(grid)}[/]  "
                f"violation={viol_agg:.4f}  cost=${cost_agg:.4f}  "
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
            for k, v in point.items():
                overrides[f"{config_section}.{k}"] = v

            val_results = self._evaluator.evaluate(
                workload_paths=val_paths,
                config_overrides=overrides,
                phase=self._phase_name,
                grid_point=f"{idx}_val",
                out_subdir=phase_dir / f"grid_point_{idx:03d}" / "val",
            )
            val_viol, val_cost = aggregate(val_results, metric)
            grid_results[idx]["val_violation_agg"] = val_viol
            grid_results[idx]["val_cost_agg"] = val_cost

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

    @staticmethod
    def _select_best(
        grid_results: list[dict[str, Any]],
        pareto_indices: list[int],
    ) -> int:
        """Pick the best Pareto-optimal point by validation performance.

        Strategy: lowest validation violation; ties broken by lowest
        validation cost.
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
        return min(
            validated,
            key=lambda i: (
                grid_results[i]["val_violation_agg"],
                grid_results[i]["val_cost_agg"],
            ),
        )

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
        table.add_column("GP", justify="right")
        table.add_column("Params", justify="left")
        table.add_column("Train Viol", justify="right")
        table.add_column("Train Cost", justify="right")
        table.add_column("Val Viol", justify="right")
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
        output = {
            "best_grid_point": best_idx,
            "best_params": grid_results[best_idx]["params"],
            "grid_results": grid_results,
        }
        with open(phase_dir / "sweep_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
