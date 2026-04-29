"""Shared sweep logic for autoscaler and routing parameter tuning.

Both the autoscaler sweep (design step 5) and the routing sweep (design
step 6) follow the same pattern: generate candidate configurations,
evaluate them on the training set, rank them via
:meth:`SloObjective.rank_indices`, validate the top-k, and select the
best.  :class:`ParamSweep` encapsulates this logic with pluggable search
strategies (``grid``, ``random``, ``coordinate_descent``).
"""

from __future__ import annotations

import itertools
import json
import logging
import random as stdlib_random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

import autoslo.utils.config as cfgu
from autoslo.config.component_configs import WorkloadConfig
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.tuner_utils import (
    AggregatedSimulationResults,
    SimulationResult,
)
from autoslo.utils.yaml_helpers import dump_yaml

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
    """Grid search with top-k training ranking and validation-set selection.

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
        initial_config: dict[str, Any],
        run_dir: Path,
        phase_name: str,
        slo_objective: SloObjective,
        agg_method: str,
    ) -> None:
        self._evaluator = evaluator
        self._config = initial_config
        self._run_dir = run_dir
        self._phase_name = phase_name
        self._slo_objective = slo_objective
        self._agg_method = agg_method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sweep(
        self,
        train_workload_configs: list[WorkloadConfig],
        val_workload_configs: list[WorkloadConfig],
        sweep_config: dict[str, Any],
    ) -> tuple[
        dict[str, Any], AggregatedSimulationResults, AggregatedSimulationResults
    ]:
        """Run a parameter sweep and return the best config with its metrics.

        *sweep_config* format::

            {"strategy": "random", "seed": 42, "budget": 20, "val_top_k": 5,
             "params": {"param.name": [v1, v2], ...}}

        Both ``strategy`` (default ``"grid"``) and ``params`` (default
        ``{}``) are optional.  Supported strategies: ``"grid"``,
        ``"random"``, ``"coordinate_descent"``.

        Parameters
        ----------
        train_workload_configs :
            Workload configurations for training scenarios.
        val_workload_configs :
            Workload configurations for validation scenarios.
        sweep_config :
            Sweep configuration (see above).

        Returns
        -------
        The full config dict with the best parameter values applied.
        """
        strategy = sweep_config.get("strategy", "grid")
        param_ranges = sweep_config.get("params", {})
        options = {
            k: v
            for k, v in sweep_config.items()
            if k not in ("strategy", "params")
        }
        phase_dir = self._run_dir / self._phase_name

        # Empty params → nothing to sweep; fall back to grid (evaluates
        # the base config once).
        if not param_ranges:
            strategy = "grid"

        dump_yaml(self._config, phase_dir / "initial_config.yml")

        # ── Generate & evaluate candidates (strategy-specific) ─────
        if strategy == "grid":
            candidates, grid_results = self._sweep_grid(
                train_workload_configs, param_ranges, phase_dir
            )
        elif strategy == "random":
            candidates, grid_results = self._sweep_random(
                train_workload_configs, param_ranges, options, phase_dir
            )
        elif strategy == "coordinate_descent":
            candidates, grid_results = self._sweep_coordinate_descent(
                train_workload_configs, param_ranges, options, phase_dir
            )
        else:
            raise ValueError(f"Unknown sweep strategy: {strategy!r}")

        # ── Rank training candidates and select top-k for validation ─
        val_top_k: int = sweep_config.get("val_top_k", 5)
        points = [
            ViolationCost(r["train_violation_agg"], r["train_cost_agg"])
            for r in grid_results
        ]
        ranked_indices = self._slo_objective.rank_indices(points)
        for rank, idx in enumerate(ranked_indices):
            grid_results[idx]["train_rank"] = rank
        top_k_indices = ranked_indices[:val_top_k]

        console.print(
            f"\n  [cyan]Top-k validation:[/] {len(top_k_indices)} of "
            f"{len(candidates)} points (k={val_top_k})"
        )

        # ── Validate top-k candidates ──────────────────────────────
        console.print(f"\n[bold cyan]Validation sweep:[/]")
        val_overrides = [candidates[idx] for idx in top_k_indices]
        all_val_results = self._evaluator.evaluate_batch_from_overrides(
            progress_bar_label=self._phase_name,
            workload_configs=val_workload_configs,
            base_config=self._config,
            all_config_overrides=val_overrides,
            out_dir=phase_dir / "val",
        )
        for i, idx in enumerate(top_k_indices):
            val_results = all_val_results[i]
            val_agg = SimulationResult.aggregate(val_results, self._agg_method)
            val_primary = val_agg.primary_violation(
                self._slo_objective.slo_metric
            )
            grid_results[idx]["val_primary_violation_agg"] = val_primary
            grid_results[idx]["val_cost_agg"] = val_agg.cost
            grid_results[idx]["val_metrics"] = val_agg

        # ── Select best ───────────────────────────────────────────
        best_idx = self._select_best(grid_results, top_k_indices)
        best_params = candidates[best_idx]

        # ── Rich summary table ─────────────────────────────────────
        self._print_top_k_table(grid_results, top_k_indices, best_idx)

        # ── Persist results ────────────────────────────────────────
        self._write_sweep_results(phase_dir, grid_results, best_idx)
        final_config = cfgu.copy_and_apply_overrides(self._config, best_params)
        dump_yaml(final_config, phase_dir / "final_config.yml")

        best_train_agg = grid_results[best_idx]["train_metrics"]
        best_val_agg = grid_results[best_idx]["val_metrics"]
        return final_config, best_train_agg, best_val_agg

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _evaluate_candidates(
        self,
        train_workload_configs: list[WorkloadConfig],
        candidates: list[dict[str, Any]],
        out_dir: Path,
    ) -> list[dict[str, Any]]:
        """Evaluate *candidates* on training scenarios and return result dicts."""
        all_train_results = self._evaluator.evaluate_batch_from_overrides(
            progress_bar_label=self._phase_name,
            workload_configs=train_workload_configs,
            base_config=self._config,
            all_config_overrides=candidates,
            out_dir=out_dir,
        )
        grid_results: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            train_agg = SimulationResult.aggregate(
                all_train_results[idx], self._agg_method
            )
            train_primary = train_agg.primary_violation(
                self._slo_objective.slo_metric
            )
            grid_results.append(
                {
                    "grid_point": idx,
                    "params": candidate,
                    "train_violation_agg": train_primary,
                    "train_cost_agg": train_agg.cost,
                    "train_metrics": train_agg,
                    "train_rank": None,
                    "val_primary_violation_agg": None,
                    "val_cost_agg": None,
                    "val_metrics": None,
                }
            )
        return grid_results

    def _sweep_grid(
        self,
        train_workload_configs: list[WorkloadConfig],
        param_ranges: dict[str, list],
        phase_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Exhaustive grid search (original strategy)."""
        grid = build_grid(param_ranges)
        self._print_preflight(param_ranges, grid, len(train_workload_configs))
        console.print(f"\n[bold cyan]Training sweep:[/]")
        grid_results = self._evaluate_candidates(
            train_workload_configs, grid, phase_dir / "train"
        )
        return grid, grid_results

    def _sweep_random(
        self,
        train_workload_configs: list[WorkloadConfig],
        param_ranges: dict[str, list],
        options: dict[str, Any],
        phase_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Random search: sample *budget* configs from the grid."""
        full_grid = build_grid(param_ranges)
        budget = options.get("budget", len(full_grid))
        seed = options.get("seed", 42)

        if budget >= len(full_grid):
            grid = full_grid
        else:
            rng = stdlib_random.Random(seed)
            grid = rng.sample(full_grid, budget)

        self._print_preflight(
            param_ranges,
            grid,
            len(train_workload_configs),
            strategy_label=f"Random (budget={budget}, seed={seed})",
        )
        console.print(f"\n[bold cyan]Training sweep:[/]")
        grid_results = self._evaluate_candidates(
            train_workload_configs, grid, phase_dir / "train"
        )
        return grid, grid_results

    def _sweep_coordinate_descent(
        self,
        train_workload_configs: list[WorkloadConfig],
        param_ranges: dict[str, list],
        options: dict[str, Any],
        phase_dir: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Coordinate descent: optimise one parameter at a time."""
        max_cycles = options.get("max_cycles", 3)
        starting_point = options.get("starting_point", None)

        # Default starting point: middle value for each parameter.
        if starting_point is None:
            starting_point = {
                name: values[len(values) // 2]
                for name, values in param_ranges.items()
            }

        current_best = dict(starting_point)
        all_candidates: list[dict[str, Any]] = []
        all_grid_results: list[dict[str, Any]] = []
        evaluated_cache: dict[frozenset, int] = {}

        def _config_key(cfg: dict[str, Any]) -> frozenset:
            return frozenset(sorted(cfg.items()))

        total_values = sum(len(v) for v in param_ranges.values())
        console.print(
            f"\n[bold cyan]Coordinate descent[/]  "
            f"max_cycles={max_cycles}  |  "
            f"params={len(param_ranges)}  |  "
            f"total values={total_values}  |  "
            f"training scenarios={len(train_workload_configs)}  |  "
            f"max evals={max_cycles * total_values * len(train_workload_configs)}"
        )
        console.print(f"  Starting point: {current_best}")

        for cycle in range(max_cycles):
            console.print(f"\n  [bold cyan]Cycle {cycle + 1}/{max_cycles}:[/]")
            changed = False

            for param_name, param_values in param_ranges.items():
                # Candidate configs: vary only this param.
                candidates_for_param = []
                for val in param_values:
                    point = dict(current_best)
                    point[param_name] = val
                    candidates_for_param.append(point)

                # Identify configs not yet evaluated.
                new_candidates = [
                    c
                    for c in candidates_for_param
                    if _config_key(c) not in evaluated_cache
                ]

                # Evaluate new candidates.
                if new_candidates:
                    out_dir = (
                        phase_dir
                        / "train"
                        / f"cycle_{cycle}"
                        / param_name.replace(".", "_")
                    )
                    batch_results = self._evaluate_candidates(
                        train_workload_configs, new_candidates, out_dir
                    )
                    for j, c in enumerate(new_candidates):
                        idx = len(all_candidates)
                        batch_results[j]["grid_point"] = idx
                        all_candidates.append(c)
                        all_grid_results.append(batch_results[j])
                        evaluated_cache[_config_key(c)] = idx

                # Select best value for this parameter.
                indices = [
                    evaluated_cache[_config_key(c)]
                    for c in candidates_for_param
                ]
                cd_candidates = [
                    ViolationCost(
                        all_grid_results[i]["train_violation_agg"],
                        all_grid_results[i]["train_cost_agg"],
                    )
                    for i in indices
                ]
                best_local = self._slo_objective.idx_of_best(cd_candidates)
                best_val = param_values[best_local]

                if best_val != current_best[param_name]:
                    console.print(
                        f"    {param_name}: "
                        f"{current_best[param_name]} → {best_val}"
                    )
                    current_best[param_name] = best_val
                    changed = True
                else:
                    console.print(
                        f"    {param_name}: unchanged "
                        f"({current_best[param_name]})"
                    )

            if not changed:
                console.print(
                    f"\n  [dim]Converged after {cycle + 1} cycle(s).[/dim]"
                )
                break

        return all_candidates, all_grid_results

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def _select_best(
        self,
        grid_results: list[dict[str, Any]],
        candidate_indices: list[int],
    ) -> int:
        """
        Pick the best validated candidate.

        """
        candidates = [
            ViolationCost(
                grid_results[i]["val_primary_violation_agg"],
                grid_results[i]["val_cost_agg"],
            )
            for i in candidate_indices
        ]
        best_local_idx = self._slo_objective.idx_of_best(candidates)
        return candidate_indices[best_local_idx]

    # ------------------------------------------------------------------
    # Rich output
    # ------------------------------------------------------------------

    def _print_preflight(
        self,
        param_ranges: dict[str, list],
        grid: list[dict[str, Any]],
        n_train: int,
        strategy_label: str = "Grid",
    ) -> None:
        table = Table(
            title=f"Parameter Ranges — {strategy_label}", show_lines=True
        )
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

    def _print_top_k_table(
        self,
        grid_results: list[dict[str, Any]],
        top_k_indices: list[int],
        best_idx: int,
    ) -> None:
        table = Table(
            title=f"{self._phase_name} — Top-K Validation Results",
            show_lines=True,
        )
        slo_label = self._slo_objective.slo_metric
        slo_thresh = self._slo_objective.slo_threshold
        table.add_column("GP", justify="right")
        table.add_column("Train Rank", justify="right")
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

        for idx in top_k_indices:
            r = grid_results[idx]
            is_best = idx == best_idx
            style = "bold green" if is_best else ""
            val_v = (
                f"{r['val_primary_violation_agg']:.4f}"
                if r["val_primary_violation_agg"] is not None
                else "—"
            )
            val_c = (
                f"{r['val_cost_agg']:.4f}"
                if r["val_cost_agg"] is not None
                else "—"
            )
            table.add_row(
                str(idx),
                str(r["train_rank"]),
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
