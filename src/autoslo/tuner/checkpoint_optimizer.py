"""Greedy checkpoint optimiser — design step 4.

Iteratively places :class:`CapacityCheckpoint` at the earliest
sliding-window with a violation rate above the configured threshold,
tries every allowed RPU size, picks the best on training data, and
validates on held-out scenarios.  Stops when the budget is exhausted,
no violating window remains, or validation improvement is below
epsilon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from autoslo.blueprints.cluster import Cluster
from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.config import TunerConfig
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.types import (
    AggregatedMetrics,
    ScenarioResult,
    SloObjective,
    aggregate,
    is_feasible,
    primary_violation,
    threshold_aware_select,
)

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Violation window dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViolationWindow:
    """A time window with aggregated violation statistics."""

    start_s: float
    end_s: float
    violation_rate: float
    num_violations: int
    num_queries: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_SPIN_UP_DELAY_S = 120.0


def _checkpoints_to_config(
    checkpoints: list[CapacityCheckpoint],
) -> dict[str, Any]:
    """Serialise checkpoints into a config-override dict."""
    return {
        "autoscaling_config.capacity_checkpoints": [
            {"time_s": cp.time_s, "min_rpus": list(cp.min_rpus)}
            for cp in checkpoints
        ]
    }


def _get_spin_up_delay(initial_config: dict[str, Any]) -> float:
    """Read spin_up_delay_s from the managed_cluster_pool_config section."""
    mcp = initial_config.get("managed_cluster_pool_config") or {}
    return float(mcp.get("spin_up_delay_s", DEFAULT_SPIN_UP_DELAY_S))


def _get_allowed_rpu_sizes(initial_config: dict[str, Any]) -> list[int]:
    """Read allowed_rpu_sizes from the managed_cluster_pool_config section."""
    mcp = initial_config.get("managed_cluster_pool_config") or {}
    sizes = mcp.get("allowed_rpu_sizes", Cluster.ALL_ALLOWED_RPU_SIZES)
    return [int(s) for s in sizes]


def find_violation_windows(
    results: list[ScenarioResult],
    window_s: float,
    slo_s: float,
    slo_dict: dict[str, float] | None = None,
) -> list[ViolationWindow]:
    """Compute sliding-window violation rates across training scenarios.

    For each scenario, read its ``structured_log.parquet``, filter
    completion events, bin them into non-overlapping windows of
    *window_s* seconds, compute per-window violation rates, then
    average across scenarios.

    Returns windows sorted chronologically.
    """
    from autoslo.blueprint_selection.slo_resolver import SloResolver

    # Collect per-scenario window stats: {window_start: [rate, n_viol, n_q]}
    per_scenario: list[dict[float, tuple[float, int, int]]] = []

    for result in results:
        log_path = result.out_dir / "structured_log.parquet"
        if not log_path.exists():
            continue
        log = pd.read_parquet(log_path)
        completions = log[log["event_type"] == "completion"].copy()
        if completions.empty:
            continue

        resolver = SloResolver.from_dict(slo_s, slo_dict or {})
        durations = completions["latency_s"].fillna(0.0)
        per_row_slo = (
            completions["query_text_id"].map(resolver.resolve).fillna(slo_s)
        )
        completions["violated"] = durations > per_row_slo

        # Use the completion timestamp (= end_time_s) to assign windows.
        completions["window_start"] = (
            np.floor(completions["timestamp"].astype(float) / window_s) * window_s
        )

        windows: dict[float, tuple[float, int, int]] = {}
        for ws, grp in completions.groupby("window_start"):
            n_q = len(grp)
            n_v = int(grp["violated"].sum())
            vr = n_v / n_q if n_q else 0.0
            windows[float(ws)] = (vr, n_v, n_q)
        per_scenario.append(windows)

    if not per_scenario:
        return []

    # Gather all window starts across scenarios.
    all_starts = sorted(
        {ws for scenario in per_scenario for ws in scenario}
    )

    # Average across scenarios.
    aggregated: list[ViolationWindow] = []
    for ws in all_starts:
        rates, viols, queries = [], [], []
        for scenario in per_scenario:
            if ws in scenario:
                r, v, q = scenario[ws]
                rates.append(r)
                viols.append(v)
                queries.append(q)
        aggregated.append(
            ViolationWindow(
                start_s=ws,
                end_s=ws + window_s,
                violation_rate=float(np.mean(rates)) if rates else 0.0,
                num_violations=int(np.sum(viols)),
                num_queries=int(np.sum(queries)),
            )
        )

    return aggregated


# ---------------------------------------------------------------------------
# CheckpointOptimizer
# ---------------------------------------------------------------------------


class CheckpointOptimizer:
    """Greedy capacity-checkpoint placement.

    Parameters
    ----------
    evaluator :
        The shared scenario evaluator.
    tuner_config :
        Tuner hyper-parameters (budget, epsilon, window size, threshold).
    initial_config :
        Base simulator config (used to read RPU sizes, spin-up delay, SLO).
    run_dir :
        Root directory for the current tuner run.
    """

    def __init__(
        self,
        evaluator: ScenarioEvaluator,
        tuner_config: TunerConfig,
        initial_config: dict[str, Any],
        run_dir: Path,
        slo_objective: SloObjective | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._tuner_config = tuner_config
        self._initial_config = initial_config
        self._run_dir = run_dir

        self._allowed_rpu_sizes = _get_allowed_rpu_sizes(initial_config)
        self._spin_up_delay_s = _get_spin_up_delay(initial_config)
        self._slo_s = float(
            (initial_config.get("slo_config") or {}).get("slo_s", 10.0)
        )
        self._slo_dict: dict[str, float] | None = None
        slo_dict_filename = (initial_config.get("slo_config") or {}).get(
            "slo_dict_filename"
        )
        if slo_dict_filename:
            try:
                from autoslo.blueprint_selection.slo_resolver import SloResolver

                self._slo_dict = SloResolver(
                    self._slo_s, slo_dict_filename
                ).slo_dict
            except Exception:
                logger.warning("Could not pre-load SLO dict; using default SLO.")

        # SLO objective for threshold-aware candidate selection.
        if slo_objective is not None:
            self._slo_objective = slo_objective
        else:
            slo_cfg = initial_config.get("slo_config") or {}
            self._slo_objective = SloObjective(
                slo_metric=str(slo_cfg.get("slo_metric", "binary")),
                slo_threshold=float(slo_cfg.get("slo_threshold", 1.0)),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        baseline_val_violation: float,
    ) -> list[CapacityCheckpoint]:
        """Run the greedy checkpoint placement loop.

        Parameters
        ----------
        train_paths :
            Parquet workload paths for training scenarios.
        val_paths :
            Parquet workload paths for validation scenarios.
        baseline_val_violation :
            Baseline validation violation rate (used as the starting
            ``best_val_violation`` for early stopping).

        Returns
        -------
        The selected list of capacity checkpoints.
        """
        metric = self._tuner_config.aggregation_metric
        current_checkpoints: list[CapacityCheckpoint] = []
        best_val_violation = baseline_val_violation
        ckpt_dir = self._run_dir / "checkpoints"

        for round_idx in range(self._tuner_config.checkpoint_budget):
            console.rule(f"[bold cyan]Checkpoint round {round_idx}")

            # 1. Simulate training scenarios with current checkpoints.
            overrides = _checkpoints_to_config(current_checkpoints)
            train_results = self._evaluator.evaluate(
                workload_paths=train_paths,
                config_overrides=overrides,
                phase="checkpoints",
                grid_point=f"round_{round_idx:03d}_base",
                out_subdir=ckpt_dir / f"round_{round_idx:03d}" / "base",
            )

            # 2. Find violation windows.
            windows = find_violation_windows(
                train_results,
                window_s=self._tuner_config.sliding_window_s,
                slo_s=self._slo_s,
                slo_dict=self._slo_dict,
            )

            # 3. Find earliest window above threshold.
            target_window = next(
                (
                    w
                    for w in windows
                    if w.violation_rate > self._tuner_config.violation_threshold
                ),
                None,
            )
            if target_window is None:
                console.print(
                    "[green]No violation window above threshold — stopping."
                )
                break

            console.print(
                f"  Target window: [{target_window.start_s:.0f}s, "
                f"{target_window.end_s:.0f}s)  "
                f"violation_rate={target_window.violation_rate:.3f}  "
                f"({target_window.num_violations}/{target_window.num_queries} queries)"
            )

            # 4. Try each RPU size.
            candidates: list[
                tuple[CapacityCheckpoint, list[ScenarioResult], AggregatedMetrics, float]
            ] = []
            for rpu in self._allowed_rpu_sizes:
                checkpoint = CapacityCheckpoint(
                    time_s=max(
                        0.0, target_window.start_s - self._spin_up_delay_s
                    ),
                    min_rpus=(rpu,),
                )
                trial_checkpoints = current_checkpoints + [checkpoint]
                trial_overrides = _checkpoints_to_config(trial_checkpoints)
                trial_results = self._evaluator.evaluate(
                    workload_paths=train_paths,
                    config_overrides=trial_overrides,
                    phase="checkpoints",
                    grid_point=f"round_{round_idx:03d}_rpu{rpu}",
                    out_subdir=ckpt_dir
                    / f"round_{round_idx:03d}"
                    / f"rpu{rpu}",
                )
                agg = aggregate(trial_results, metric)
                candidates.append(
                    (checkpoint, trial_results, agg, agg.cost)
                )

            # 5. Pick best on training set (threshold-aware selection).
            best_idx = threshold_aware_select(
                [
                    (primary_violation(agg, self._slo_objective.slo_metric), cost)
                    for _, _, agg, cost in candidates
                ],
                self._slo_objective.slo_threshold,
            )
            best_cp, best_train_results, best_agg, best_cost = candidates[best_idx]
            best_viol = primary_violation(best_agg, self._slo_objective.slo_metric)

            self._print_candidate_table(round_idx, candidates, best_cp)

            # 6. Validate.
            val_overrides = _checkpoints_to_config(
                current_checkpoints + [best_cp]
            )
            val_results = self._evaluator.evaluate(
                workload_paths=val_paths,
                config_overrides=val_overrides,
                phase="checkpoints",
                grid_point=f"round_{round_idx:03d}_val",
                out_subdir=ckpt_dir / f"round_{round_idx:03d}" / "val",
            )
            val_agg = aggregate(val_results, metric)
            val_primary = primary_violation(val_agg, self._slo_objective.slo_metric)

            slo_metric = self._slo_objective.slo_metric
            slo_thresh = self._slo_objective.slo_threshold
            console.print(
                f"  Validation {slo_metric}={val_primary:.4f}  "
                f"(threshold={slo_thresh:.2f})  "
                f"cost=${val_agg.cost:.4f}  "
                f"(prev best={best_val_violation:.4f})"
            )

            # Threshold-aware early stopping (D-M12).
            if is_feasible(val_primary, self._slo_objective.slo_threshold):
                console.print(
                    f"[green]SLO satisfied ({slo_metric} "
                    f"{val_primary:.4f} ≤ {slo_thresh:.2f}) "
                    f"— stopping checkpoint placement."
                )
                current_checkpoints.append(best_cp)
                best_val_violation = val_primary
                self._write_round_summary(
                    round_idx, candidates, best_cp, val_agg,
                )
                break
            if best_val_violation - val_primary < self._tuner_config.checkpoint_epsilon:
                console.print(
                    "[yellow]Improvement below epsilon — early stopping."
                )
                break

            # 7. Accept.
            current_checkpoints.append(best_cp)
            best_val_violation = val_primary
            console.print(
                f"  [green]Accepted checkpoint: time_s={best_cp.time_s:.0f}  "
                f"min_rpus={best_cp.min_rpus}"
            )

            # Write round summary.
            self._write_round_summary(
                round_idx, candidates, best_cp, val_agg,
            )

        # Write the final selected checkpoints.
        self._write_selected_checkpoints(current_checkpoints)
        return current_checkpoints

    # ------------------------------------------------------------------
    # Rich output helpers
    # ------------------------------------------------------------------

    def _print_candidate_table(
        self,
        round_idx: int,
        candidates: list[
            tuple[CapacityCheckpoint, list[ScenarioResult], AggregatedMetrics, float]
        ],
        best_cp: CapacityCheckpoint,
    ) -> None:
        table = Table(
            title=f"Round {round_idx} — Candidate RPU Sizes",
            show_lines=True,
        )
        table.add_column("RPU", justify="right")
        table.add_column("time_s", justify="right")
        table.add_column("Viol. Rate", justify="right")
        table.add_column("Viol. Amount (s)", justify="right")
        table.add_column("Viol. Relative", justify="right")
        table.add_column("Train Cost ($)", justify="right")
        table.add_column("Selected", justify="center")

        for cp, _, agg, cost in candidates:
            is_best = cp == best_cp
            style = "bold green" if is_best else ""
            table.add_row(
                str(cp.min_rpus[0]),
                f"{cp.time_s:.0f}",
                f"{agg.violation_rate:.4f}",
                f"{agg.violation_amount_s:.4f}",
                f"{agg.violation_relative_mean:.4f}",
                f"{cost:.4f}",
                "✓" if is_best else "",
                style=style,
            )
        console.print(table)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _write_round_summary(
        self,
        round_idx: int,
        candidates: list[
            tuple[CapacityCheckpoint, list[ScenarioResult], AggregatedMetrics, float]
        ],
        best_cp: CapacityCheckpoint,
        val_agg: AggregatedMetrics,
    ) -> None:
        round_dir = self._run_dir / "checkpoints" / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "round_idx": round_idx,
            "selected_checkpoint": {
                "time_s": best_cp.time_s,
                "min_rpus": list(best_cp.min_rpus),
            },
            "val_violation": primary_violation(val_agg, self._slo_objective.slo_metric),
            "val_cost": val_agg.cost,
            "val_violation_rate": val_agg.violation_rate,
            "val_violation_amount_s": val_agg.violation_amount_s,
            "val_violation_relative_mean": val_agg.violation_relative_mean,
            "candidates": [
                {
                    "rpu": cp.min_rpus[0],
                    "time_s": cp.time_s,
                    "train_violation": primary_violation(agg, self._slo_objective.slo_metric),
                    "train_cost": cost,
                    "train_violation_rate": agg.violation_rate,
                    "train_violation_amount_s": agg.violation_amount_s,
                    "train_violation_relative_mean": agg.violation_relative_mean,
                }
                for cp, _, agg, cost in candidates
            ],
        }
        with open(round_dir / "candidate_results.yml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

    def _write_selected_checkpoints(
        self, checkpoints: list[CapacityCheckpoint]
    ) -> None:
        ckpt_dir = self._run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        data = [
            {"time_s": cp.time_s, "min_rpus": list(cp.min_rpus)}
            for cp in checkpoints
        ]
        with open(ckpt_dir / "selected_checkpoints.yml", "w") as f:
            yaml.dump(data, f, default_flow_style=False)
