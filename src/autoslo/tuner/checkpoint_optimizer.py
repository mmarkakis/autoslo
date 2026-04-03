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
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from autoslo.blueprints.cluster import Cluster
from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.config import TunerConfig
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator, EvalSpec
from autoslo.tuner.tuner_utils import (
    AggregatedMetrics,
    ScenarioResult,
    SloObjective,
    aggregate,
    is_feasible,
    primary_violation,
    threshold_aware_select,
)
from autoslo.blueprint_selection.slo_resolver import (
    SloResolver,
    slo_relative_violation,
)
import autoslo.utils.config as cfgu
from collections import defaultdict

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CheckpointOptimizerResult:
    """Output of :meth:`CheckpointOptimizer.optimize`.

    Checkpoints whose ``time_s`` was clamped to zero are absorbed into
    :attr:`additional_initial_rpus` so that the caller can merge them
    into ``managed_cluster_pool_config.initial_rpus`` (instant start)
    rather than treating them as timed checkpoints subject to spin-up
    delay.
    """

    checkpoints: list[CapacityCheckpoint]
    """Checkpoints with ``time_s > 0``."""

    additional_initial_rpus: tuple[int, ...]
    """RPU sizes extracted from time-zero checkpoints."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def result_to_config(
    result: CheckpointOptimizerResult,
    existing_initial_rpus: tuple[int, ...],
) -> dict[str, Any]:
    """Serialise a :class:`CheckpointOptimizerResult` into config overrides.

    Merges :attr:`~CheckpointOptimizerResult.additional_initial_rpus`
    with *existing_initial_rpus* and emits both the checkpoint list and
    the updated ``initial_rpus``.
    """
    overrides = _checkpoints_to_config(result.checkpoints)
    merged_rpus = list(existing_initial_rpus) + list(
        result.additional_initial_rpus
    )
    if result.additional_initial_rpus:
        overrides["managed_cluster_pool_config.initial_rpus"] = merged_rpus
    return overrides


def find_next_checkpoint_time(
    results: list[ScenarioResult],  # TODO: Feed dfs for easier testing
    slo_resolver: SloResolver,
    threshold: float,
    spin_up_delay_s: float,
) -> Optional[float]:
    """
    Find the next time at which to insert a capacity checkpoint, or None
    if no such time can be found through the process below. The process is:

    1. For each of the scenarios in *results*, compute the start and end time
        of each query from the structured logs. Also label each query by its
        relative SLO violation rate. Union the query start and query end events
        across all scenarios into a single timeline.
    2.  For each interval in this timeline, calculate the
        mean relative SLO violation rate for active queries, first taking the
        mean within each scenario and then averaging across scenarios to get a
        single value.
    3. Find the earliest interval where the mean relative SLO violation rate
        exceeds *threshold*. If no such interval exists, return None.
    4. Among the running queries in this interval, find the earliest query start
        time.
    5. Return this time minus *spin_up_delay_s*.

    """
    completion_structured_logs = []
    for result in results:
        log_path = result.out_dir / "structured_log.parquet"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")

        # Read in log and compute violations.
        log = pd.read_parquet(
            log_path,
            columns=[
                "timestamp",
                "event_type",
                "query_id",
                "query_text_id",
                "latency_s",
            ],
        )
        completions = log[log["event_type"] == "completion"].copy()
        if completions.empty:
            raise ValueError(f"No completion events in log: {log_path}")
        completion_structured_logs.append(completions)

    return find_next_checkpoint_time_df(
        completion_structured_logs=completion_structured_logs,
        slo_resolver=slo_resolver,
        threshold=threshold,
        spin_up_delay_s=spin_up_delay_s,
    )


def find_next_checkpoint_time_df(
    completion_structured_logs: list[pd.DataFrame],
    slo_resolver: SloResolver,
    threshold: float,
    spin_up_delay_s: float,
) -> Optional[float]:
    """
    Internal helper for find_next_checkpoint_time that takes pre-loaded
    structured logs, for easier testing.
    """

    # 1. Create events per scenario and aggregate.
    events = []
    for scenario_id, completions in enumerate(completion_structured_logs):

        completions["latency_s"] = completions["latency_s"].fillna(0.0)
        completions["start_time"] = (
            completions["timestamp"] - completions["latency_s"]
        )
        completions["slo_s"] = (
            completions["query_text_id"].map(slo_resolver.resolve).fillna(0.0)
        )
        completions["slo_relative_violation"] = completions.apply(
            lambda row: slo_relative_violation(row["latency_s"], row["slo_s"]),
            axis=1,
        )

        # 1. Create events per scenario.
        for _, row in completions.iterrows():
            events.append(
                (
                    row["start_time"],
                    "start",
                    row["slo_relative_violation"],
                    scenario_id,
                    row["query_id"],
                )
            )
            events.append(
                (
                    row["timestamp"],
                    "end",
                    row["slo_relative_violation"],
                    scenario_id,
                    row["query_id"],
                )
            )
    events.sort(key=lambda x: x[0])  # Sort by timestamp

    # 2. Compute mean violation rate in each interval and
    # 3. Find earliest interval with mean violation above threshold.
    active_queries: dict[int, dict[str, tuple[float, float]]] = defaultdict(
        dict
    )  # scenario_id -> query_id -> (start_time, violation)
    mean_relative_violation_per_scenario: dict[int, float] = {
        scenario_id: 0.0
        for scenario_id in range(len(completion_structured_logs))
    }
    for i in range(len(events) - 1):
        timestamp, event_type, violation, scenario_id, query_id = events[i]
        if event_type == "start":
            active_queries[scenario_id][query_id] = (timestamp, violation)
        else:
            active_queries[scenario_id].pop(query_id, None)

        scenario_active_queries = active_queries[scenario_id].values()
        if len(scenario_active_queries) > 0:
            mean_relative_violation_per_scenario[scenario_id] = float(
                np.mean([tup[1] for tup in scenario_active_queries])
            )
        else:
            mean_relative_violation_per_scenario[scenario_id] = 0.0
        mean_relative_violation_across_scenarios = float(
            np.mean(list(mean_relative_violation_per_scenario.values()))
        )

        # Don't make decisions based on zero-length intervals.
        next_event_time = events[i + 1][0]
        if next_event_time == timestamp:
            continue

        if mean_relative_violation_across_scenarios > threshold:
            min_seen = np.inf
            for scenario_queries in active_queries.values():
                for query_id, (timestamp, _) in scenario_queries.items():
                    min_seen = min(min_seen, timestamp)
            spinup_time = min_seen - spin_up_delay_s
            console.print(
                f"[green]Found violating interval: "
                f"{timestamp:.0f}s to {events[i+1][0]:.0f}s  "
                f"with mean relative violation "
                f"{mean_relative_violation_across_scenarios:.4f} "
                f"(threshold={threshold:.2f})  "
                f"returning checkpoint time {spinup_time:.0f}s"
            )
            return max(0, min_seen - spin_up_delay_s)

    console.print(
        f" [green]No interval with mean relative violation above {threshold} "
        f"found."
    )

    return None


# ---------------------------------------------------------------------------
# CheckpointOptimizer
# ---------------------------------------------------------------------------


class CheckpointOptimizer:
    """Greedy capacity-checkpoint placement.

    Parameters
    ----------
    evaluator :
        The shared scenario evaluator.
    config :
        The configuration dict loaded from the YAML file for this tuner run,
        which is used to configure the simulator and tuner parameters.
    run_dir :
        Root directory for the current tuner run.
    """

    def __init__(
        self,
        evaluator: ScenarioEvaluator,
        config: dict[str, Any],
        run_dir: Path,
    ) -> None:
        self._evaluator = evaluator
        self._config = config
        self._run_dir = run_dir

        self._allowed_rpu_sizes = self._cfgd(
            "managed_cluster_pool_config.allowed_rpu_sizes",
            Cluster.ALL_ALLOWED_RPU_SIZES,
        )
        self._spin_up_delay_s = self._cfgd(
            "managed_cluster_pool_config.spin_up_delay_s",
            Cluster.DEFAULT_SPIN_UP_DELAY_S,
        )

        # SLO Resolver
        slo_s = self._cfgd("slo_config.slo_s", 30.0)
        slo_dict_filename = self._cfgd("slo_config.slo_dict_filename", None)
        self._slo_resolver = SloResolver(slo_s, slo_dict_filename)

        # SLO objective for threshold-aware candidate selection.
        slo_metric = self._cfgd("slo_config.slo_metric", "binary")
        slo_threshold = self._cfgd("slo_config.slo_threshold", 1.0)
        self._slo_objective = SloObjective(
            slo_metric=str(slo_metric),
            slo_threshold=float(slo_threshold),
        )

    def _cfgd(self, dot_delimited_key: str, default: Any = None) -> Any:
        """Helper to read from the config dict with dot-delimited keys."""
        return cfgu.cfg_getd(self._config, dot_delimited_key, default)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        baseline_val_violation: float,
    ) -> CheckpointOptimizerResult:
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
        A :class:`CheckpointOptimizerResult` containing the selected
        checkpoints (``time_s > 0``) and any RPU sizes that should be
        added to ``initial_rpus`` (from time-zero checkpoints).
        """
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "p90"
        )
        checkpoint_budget = self._cfgd(
            "tuner_config.checkpoint_phase.checkpoint_budget", 5
        )
        violation_threshold = self._cfgd(
            "tuner_config.checkpoint_phase.violation_threshold", 0.1
        )
        checkpoint_epsilon = self._cfgd(
            "tuner_config.checkpoint_phase.checkpoint_epsilon", 0.01
        )

        current_checkpoints: list[CapacityCheckpoint] = []
        best_val_violation = baseline_val_violation
        ckpt_dir = self._run_dir / "checkpoints"

        for round_idx in range(checkpoint_budget):
            console.rule(f"[bold cyan]Checkpoint round {round_idx}")

            # 1. Simulate training scenarios with current checkpoints. 
            # TODO: Later can copy these over.
            overrides = _checkpoints_to_config(current_checkpoints)
            train_results = self._evaluator.evaluate(
                workload_paths=train_paths,
                config_overrides=overrides,
                phase="checkpoints",
                grid_point=f"round_{round_idx:03d}_base",
                out_subdir=ckpt_dir / f"round_{round_idx:03d}" / "base",
            )

            # 2. Find earliest promising checkpoint time.
            next_checkpoint_time = find_next_checkpoint_time(
                train_results,
                slo_resolver=self._slo_resolver,
                threshold=violation_threshold,
                spin_up_delay_s=self._spin_up_delay_s,
            )

            if next_checkpoint_time is None:
                s = "[green]No promising checkpoint time found — stopping."
                console.print(s)
                break

            # 3. Try each RPU size.
            cands: list[tuple[CapacityCheckpoint, AggregatedMetrics]] = []
            checkpoints = []
            specs: list[EvalSpec] = []
            phase = "checkpoints"
            for rpu in self._allowed_rpu_sizes:
                checkpoint = CapacityCheckpoint(
                    time_s=max(0.0, next_checkpoint_time),
                    min_rpus=(rpu,),
                )
                checkpoints.append(checkpoint)
                trial_checkpoints = current_checkpoints + [checkpoint]
                trial_overrides = _checkpoints_to_config(trial_checkpoints)
                grid_point = f"round_{round_idx:03d}_rpu{rpu}"
                spec = EvalSpec(
                    label=f"{phase} gp={grid_point}",
                    config_overrides=trial_overrides,
                    grid_point=str(grid_point),
                    out_subdir=(
                        ckpt_dir / f"round_{round_idx:03d}" / f"rpu{rpu}"
                    ),
                )
                specs.append(spec)
            all_trial_results = self._evaluator.evaluate_batch(
                workload_paths=train_paths, specs=specs, phase=phase
            )
            for i in range(len(self._allowed_rpu_sizes)):
                checkpoint = checkpoints[i]
                trial_result = all_trial_results[i]
                agg = aggregate(trial_result, metric)
                cands.append((checkpoint, agg))

            # 5. Pick best on training set (threshold-aware selection).
            sm = self._slo_objective.slo_metric
            st = self._slo_objective.slo_threshold
            best_idx = threshold_aware_select(
                [(primary_violation(agg, sm), agg.cost) for _, agg in cands],
                st,
            )
            best_cp, _ = cands[best_idx]

            self._print_candidate_table(round_idx, cands, best_cp)

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
            val_primary = primary_violation(val_agg, sm)

            console.print(
                f"  Validation {sm}={val_primary:.4f}  "
                f"(threshold={st:.2f})  "
                f"cost=${val_agg.cost:.4f}  "
                f"(prev best={best_val_violation:.4f})"
            )

            # Threshold-aware early stopping.
            if is_feasible(val_primary, st):
                console.print(
                    f"[green]SLO satisfied on validation set ({sm} "
                    f"{val_primary:.4f} ≤ {st:.2f}) "
                    f"— stopping checkpoint placement."
                )
                current_checkpoints.append(best_cp)
                best_val_violation = val_primary
                self._write_round_summary(
                    round_idx,
                    cands,
                    best_cp,
                    val_agg,
                )
                break
            if best_val_violation - val_primary < checkpoint_epsilon:
                console.print(
                    "[yellow]Improvement below epsilon on validation set — early stopping."
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
                round_idx,
                cands,
                best_cp,
                val_agg,
            )

        # Partition: absorb time-zero checkpoints into initial_rpus.
        real_checkpoints = [
            cp for cp in current_checkpoints if cp.time_s > 0
        ]
        zero_rpus: list[int] = []
        for cp in current_checkpoints:
            if cp.time_s == 0:
                zero_rpus.extend(cp.min_rpus)

        if zero_rpus:
            console.print(
                f"  [yellow]Absorbed {len(zero_rpus)} time-zero "
                f"checkpoint(s) into additional_initial_rpus: "
                f"{tuple(zero_rpus)}"
            )

        # Write the final selected checkpoints.
        self._write_selected_checkpoints(current_checkpoints)
        return CheckpointOptimizerResult(
            checkpoints=real_checkpoints,
            additional_initial_rpus=tuple(zero_rpus),
        )

    # ------------------------------------------------------------------
    # Rich output helpers
    # ------------------------------------------------------------------

    def _print_candidate_table(
        self,
        round_idx: int,
        candidates: list[
            tuple[
                CapacityCheckpoint,
                AggregatedMetrics,
            ]
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

        for cp, agg in candidates:
            is_best = cp == best_cp
            style = "bold green" if is_best else ""
            table.add_row(
                str(cp.min_rpus[0]),
                f"{cp.time_s:.0f}",
                f"{agg.violation_rate:.4f}",
                f"{agg.violation_amount_s:.4f}",
                f"{agg.violation_relative_mean:.4f}",
                f"{agg.cost:.4f}",
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
            tuple[
                CapacityCheckpoint,
                AggregatedMetrics,
            ]
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
            "val_violation": primary_violation(
                val_agg, self._slo_objective.slo_metric
            ),
            "val_cost": val_agg.cost,
            "val_violation_rate": val_agg.violation_rate,
            "val_violation_amount_s": val_agg.violation_amount_s,
            "val_violation_relative_mean": val_agg.violation_relative_mean,
            "candidates": [
                {
                    "rpu": cp.min_rpus[0],
                    "time_s": cp.time_s,
                    "train_violation": primary_violation(
                        agg, self._slo_objective.slo_metric
                    ),
                    "train_cost": agg.cost,
                    "train_violation_rate": agg.violation_rate,
                    "train_violation_amount_s": agg.violation_amount_s,
                    "train_violation_relative_mean": agg.violation_relative_mean,
                }
                for cp, agg in candidates
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
