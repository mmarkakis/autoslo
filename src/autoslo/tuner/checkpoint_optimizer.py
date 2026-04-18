"""Greedy checkpoint optimiser — design step 4.

Iteratively places :class:`CapacityCheckpoint` at the earliest
sliding-window with a violation rate above the configured threshold,
tries every allowed RPU size, picks the best on training data, and
validates on held-out scenarios.  Stops when the budget is exhausted,
no violating window remains, or validation improvement is below
epsilon.
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

import autoslo.utils.config as cfgu
from autoslo.clusters.capacity_checkpoint import CapacityCheckpoint
from autoslo.clusters.cluster import Cluster
from autoslo.slo.slo_metric import LatencySlo, SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.tuner_utils import (
    AggregatedSimulationResults,
    SimulationResult,
    threshold_aware_select,
)
from autoslo.utils.yaml_helpers import dump

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_checkpoints_to_config(
    config: dict[str, Any],
    new_checkpoints: list[CapacityCheckpoint],
) -> dict[str, Any]:

    initial_rpus_dot_delimited_key = "managed_cluster_pool_config.initial_rpus"
    initial_rpus = copy.deepcopy(
        cfgu.getd(config, initial_rpus_dot_delimited_key, [])
    )
    nonzero_checkpoints: list[dict[str, Any]] = []

    for cp in new_checkpoints:
        if cp.rel_time_s == 0:
            initial_rpus.extend(cp.min_rpus)
        else:
            nonzero_checkpoints.append(
                {
                    "rel_time_s": cp.rel_time_s,
                    "min_rpus": list(cp.min_rpus),
                }
            )

    overrides = cfgu.copy_and_apply_overrides(
        config,
        {initial_rpus_dot_delimited_key: sorted(initial_rpus)},
    )

    if nonzero_checkpoints:
        dot_delimited_key = "capacity_checkpoints"
        existing_checkpoints = cfgu.getd(config, dot_delimited_key, [])
        merged_checkpoints = existing_checkpoints + nonzero_checkpoints
        overrides = cfgu.copy_and_apply_overrides(
            overrides, {dot_delimited_key: merged_checkpoints}
        )
    return overrides


def find_next_checkpoint_time(
    results: list[SimulationResult],
    slo_resolver: SloResolver,
    slo_objective: SloObjective,
    min_delinquent_workloads: int,
    spin_up_delay_s: float,
) -> Optional[float]:
    """
    Find the next time at which to insert a capacity checkpoint, or None
    if no such time can be found through the process below. The process is:

    1. For each of the scenarios in *results*, compute the start and end time
        of each query from the structured logs. Also get its SLO from the
        **slo-resolver**. Union the query start and query end events
        across all scenarios into a single timeline.
    2.  For each interval in this timeline, calculate whether each of the
        workloads is in "SLO delinquency", based on its queries active during
        that interval and the configured **slo_objective**. Then compute the
        number of delinquent workloads.
    3. Find the earliest interval where the number of delinquent workloads
        exceeds *min_delinquent_workloads*. If no such interval exists,
        return None.
    4. Return the start time of that interval minus *spin_up_delay_s*.

    """
    completion_structured_logs = []
    for result in results:
        log_path = result.simulation_dir / "structured_log.parquet"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")

        # Read in log and compute violations.
        log = pd.read_parquet(
            log_path,
            columns=[
                "rel_time_s",
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
        slo_objective=slo_objective,
        min_delinquent_workloads=min_delinquent_workloads,
        spin_up_delay_s=spin_up_delay_s,
    )


def find_next_checkpoint_time_df(
    completion_structured_logs: list[pd.DataFrame],
    slo_resolver: SloResolver,
    slo_objective: SloObjective,
    min_delinquent_workloads: int,
    spin_up_delay_s: float,
) -> Optional[float]:
    """
    Internal helper for find_next_checkpoint_time that takes pre-loaded
    structured logs, for easier testing.
    """

    # Create events per scenario and form a single timeline.
    events = []
    for scenario_id, completions in enumerate(completion_structured_logs):

        completions["latency_s"] = completions["latency_s"].fillna(0.0)
        completions["start_time"] = (
            completions["rel_time_s"] - completions["latency_s"]
        )
        completions["slo_s"] = (
            completions["query_text_id"].map(slo_resolver.resolve).fillna(0.0)
        )

        # 1. Create events per scenario.
        for _, row in completions.iterrows():
            events.append(
                {
                    "time": row["start_time"],
                    "event_type": "start",
                    "latency_s": row["latency_s"],
                    "slo_s": row["slo_s"],
                    "scenario_id": scenario_id,
                    "query_id": row["query_id"],
                }
            )
            events.append(
                {
                    "time": row["rel_time_s"],
                    "event_type": "end",
                    "latency_s": row["latency_s"],
                    "slo_s": row["slo_s"],
                    "scenario_id": scenario_id,
                    "query_id": row["query_id"],
                }
            )
    events.sort(key=lambda x: x["time"])  # Sort by timestamp

    # Process the intervals in the timeline.
    active_queries: dict[int, dict[str, LatencySlo]] = defaultdict(
        dict
    )  # scenario_id -> query_id -> LatencySlo
    delinquency_per_workload: dict[int, bool] = {
        scenario_id: False
        for scenario_id in range(len(completion_structured_logs))
    }
    for i in range(len(events) - 1):
        event = events[i]
        if event["event_type"] == "start":
            active_queries[event["scenario_id"]][event["query_id"]] = (
                LatencySlo(
                    event["latency_s"],
                    event["slo_s"],
                )
            )
        else:
            active_queries[event["scenario_id"]].pop(event["query_id"], None)

        scenario_active_queries = list(
            active_queries[event["scenario_id"]].values()
        )

        if len(scenario_active_queries) > 0:
            delinquency_per_workload[event["scenario_id"]] = (
                not slo_objective.is_met(
                    per_query_latency_slo=scenario_active_queries
                )
            )
        else:
            delinquency_per_workload[event["scenario_id"]] = False
        num_delinquent_workloads = sum(delinquency_per_workload.values())

        # Don't make decisions based on zero-length intervals.
        this_event_time = event["time"]
        next_event_time = events[i + 1]["time"]
        if next_event_time == this_event_time:
            continue

        if num_delinquent_workloads >= min_delinquent_workloads:
            spinup_time = this_event_time - spin_up_delay_s
            console.print(
                f"[green]Found violating interval: "
                f"{this_event_time:.2f}s to {next_event_time:.2f}s  "
                f"with {num_delinquent_workloads} delinquent workloads. "
                f"Suggesting checkpoint at {spinup_time:.2f}s."
            )
            return max(0, spinup_time)

    console.print(
        f" [green]No violating interval found with at least "
        f"{min_delinquent_workloads} delinquent workloads."
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
        agg_metric: str,
    ) -> None:
        self._evaluator = evaluator
        self._config = config
        self._run_dir = run_dir
        self._agg_metric = agg_metric

        self._allowed_rpu_sizes = self._cfgd(
            "autoscaling_config.allowed_rpu_sizes",
            Cluster.ALL_ALLOWED_RPU_SIZES,
        )
        self._spin_up_delay_s = self._cfgd(
            "provisioner_config.spin_up_delay_s",
            Cluster.DEFAULT_SPIN_UP_DELAY_S,
        )

        # SLO Resolver
        slo_s = self._cfgd("slo_config.slo_s", 30.0)
        slo_dict_filename = self._cfgd("slo_config.slo_dict_filename", None)
        self._slo_resolver = SloResolver(slo_s, slo_dict_filename)

        # SLO objective for threshold-aware candidate selection.
        slo_metric = SloMetric(self._cfgd("slo_config.slo_metric", "binary"))
        slo_threshold = self._cfgd("slo_config.slo_threshold", 1.0)
        self._slo_objective = SloObjective(
            slo_metric=slo_metric,
            slo_threshold=float(slo_threshold),
        )

    def _cfgd(self, dot_delimited_key: str, default: Any = None) -> Any:
        """Helper to read from the config dict with dot-delimited keys."""
        return cfgu.getd(self._config, dot_delimited_key, default)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        train_paths: list[Path],
    ) -> tuple[dict[str, Any], AggregatedSimulationResults]:
        """Run the greedy checkpoint placement loop.

        Checkpoints are placed greedily using training data only.
        Validation is deferred to the caller.

        Parameters
        ----------
        train_paths :
            Parquet workload paths for training scenarios.

        Returns
        -------
        A tuple of ``(config, train_agg)`` where *config* is the full
        config dict with the best checkpoint schedule applied and
        *train_agg* is the aggregated training results for that config.
        """
        max_checkpoints = self._cfgd(
            "tuner_config.checkpoint_phase.max_checkpoints", 5
        )
        min_delinquent_workload_fraction = self._cfgd(
            "tuner_config.checkpoint_phase.min_delinquent_workload_fraction",
            0.5,
        )
        min_delinquent_workloads = int(
            min_delinquent_workload_fraction * len(train_paths)
        )

        ckpt_dir = self._run_dir / "checkpoints"

        current_config = copy.deepcopy(self._config)
        dump(current_config, ckpt_dir / "initial_config.yml")

        # Track evaluation results for the current config to avoid
        # re-evaluation across rounds.  When a candidate is accepted,
        # its results carry forward as the next round's baseline.
        current_train_results: list[SimulationResult] | None = None
        current_train_agg: AggregatedSimulationResults | None = None

        for round_idx in range(max_checkpoints):
            console.rule(f"[bold cyan]Checkpoint round {round_idx}")
            round_dir = ckpt_dir / f"round_{round_idx:03d}"
            dump(current_config, round_dir / "initial_config.yml")

            # 1. Get baseline results (reuse from previous round
            #    if available).
            if current_train_results is not None:
                train_results = current_train_results
                assert current_train_agg is not None
                agg_train_results = current_train_agg
            else:
                nested_train_results = (
                    self._evaluator.evaluate_batch_from_configs(
                        phase_name=f"round_{round_idx:03d}_baseline",
                        workload_paths=train_paths,
                        configs=[current_config],
                        out_dir=round_dir / "baseline",
                    )
                )
                train_results = nested_train_results[0]
                agg_train_results = SimulationResult.aggregate(
                    train_results, self._agg_metric
                )
                current_train_agg = agg_train_results

            # 2. Find earliest promising checkpoint time.
            next_checkpoint_time = find_next_checkpoint_time(
                train_results,
                slo_resolver=self._slo_resolver,
                slo_objective=self._slo_objective,
                min_delinquent_workloads=min_delinquent_workloads,
                spin_up_delay_s=self._spin_up_delay_s,
            )

            if next_checkpoint_time is None:
                s = "[green]No promising checkpoint time found — stopping."
                console.print(s)
                break

            # 3. Try each RPU size.
            cands: list[
                tuple[CapacityCheckpoint, AggregatedSimulationResults]
            ] = []
            checkpoints = []
            all_configs = []
            initial_rpus = cfgu.getd(
                current_config, "managed_cluster_pool_config.initial_rpus", []
            )
            for rpu in self._allowed_rpu_sizes:
                checkpoint = CapacityCheckpoint(
                    rel_time_s=max(0.0, next_checkpoint_time),
                    min_rpus=tuple(sorted(initial_rpus + [rpu])),
                )
                checkpoints.append(checkpoint)
                trial_config = new_checkpoints_to_config(
                    config=current_config, new_checkpoints=[checkpoint]
                )
                all_configs.append(trial_config)
            all_trial_results = self._evaluator.evaluate_batch_from_configs(
                phase_name=f"round_{round_idx:03d}_candidates",
                workload_paths=train_paths,
                configs=all_configs,
                out_dir=round_dir / "train",
            )
            for i in range(len(self._allowed_rpu_sizes)):
                checkpoint = checkpoints[i]
                trial_results = all_trial_results[i]
                agg = SimulationResult.aggregate(
                    trial_results, self._agg_metric
                )
                cands.append((checkpoint, agg))

            # 4. Pick best on training set (threshold-aware selection).
            sm = self._slo_objective.slo_metric
            st = self._slo_objective.slo_threshold
            best_idx = threshold_aware_select(
                [(agg.primary_violation(sm), agg.cost) for _, agg in cands],
                st,
            )
            best_cp, _ = cands[best_idx]
            self._print_candidate_table(round_idx, cands, best_cp)

            AggregatedSimulationResults.print_comparison(
                ("Current config", agg_train_results),
                ("Best candidate", cands[best_idx][1]),
                agg_metric=self._agg_metric,
                slo_metric=sm,
                console=console,
            )

            # 5. Accept the best checkpoint only if it actually improves the
            # metric we care about.
            previous_best_slo = agg_train_results.primary_violation(sm)
            new_best_slo = cands[best_idx][1].primary_violation(sm)
            relative_improvement = (
                (previous_best_slo - new_best_slo) / abs(previous_best_slo)
                if previous_best_slo != 0
                else 0
            )
            previous_best_cost = agg_train_results.cost
            new_best_cost = cands[best_idx][1].cost
            relative_cost_improvement = (
                (previous_best_cost - new_best_cost) / abs(previous_best_cost)
                if previous_best_cost != 0
                else 0
            )

            min_rel_improvement_for_acceptance = self._cfgd(
                "tuner_config.checkpoint_phase.min_rel_improvement_for_acceptance",
                0.01,
            )
            # To proceed,must either improve the SLO metric by at least epsilon,
            # or improve the SLO by a positive amount and also improve cost by
            # at least min_rel_improvement_for_acceptance.
            if (
                relative_improvement < min_rel_improvement_for_acceptance
            ) and not (
                relative_improvement > 0
                and relative_cost_improvement
                >= min_rel_improvement_for_acceptance
            ):
                console.print(
                    f" [red]Rejecting because SLO metric improvement "
                    f"{relative_improvement:.2%} is below the "
                    f"threshold of {min_rel_improvement_for_acceptance:.2%} "
                    f"and cost impact {relative_cost_improvement:.2%} is not "
                    f"sufficient."
                )
                self._write_round_summary(round_idx, cands, best_cp)
                dump(current_config, round_dir / "final_config.yml")
                break

            current_config = all_configs[best_idx]
            # Carry forward the accepted candidate's results so the
            # next round can skip re-evaluating the same config.
            current_train_results = all_trial_results[best_idx]
            current_train_agg = cands[best_idx][1]
            console.print(
                f"  [green]Accepted with SLO metric improvement "
                f"{relative_improvement:.2%} and cost impact "
                f"{relative_cost_improvement:.2%}"
            )

            # Write round summary.
            self._write_round_summary(round_idx, cands, best_cp)
            dump(current_config, round_dir / "final_config.yml")

        # Write the final config.
        dump(current_config, ckpt_dir / "final_config.yml")
        assert (
            current_train_agg is not None
        ), "No checkpoint rounds were configured (max_checkpoints=0)."
        return current_config, current_train_agg

    # ------------------------------------------------------------------
    # Rich output helpers
    # ------------------------------------------------------------------

    def _print_candidate_table(
        self,
        round_idx: int,
        candidates: list[
            tuple[
                CapacityCheckpoint,
                AggregatedSimulationResults,
            ]
        ],
        best_cp: CapacityCheckpoint,
    ) -> None:
        table = Table(
            title=f"Round {round_idx} — Candidate RPU Sizes",
            show_lines=True,
        )
        table.add_column("RPU", justify="right")
        table.add_column("rel_time_s", justify="right")
        table.add_column("Viol. Rate", justify="right")
        table.add_column("Viol. Amount (s)", justify="right")
        table.add_column("Viol. Relative", justify="right")
        table.add_column("Train Cost ($)", justify="right")
        table.add_column("Selected", justify="center")

        for cp, agg in candidates:
            is_best = cp == best_cp
            style = "bold green" if is_best else ""
            table.add_row(
                ", ".join(str(rpu) for rpu in cp.min_rpus),
                f"{cp.rel_time_s:.0f}",
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
                AggregatedSimulationResults,
            ]
        ],
        best_cp: CapacityCheckpoint,
    ) -> None:
        round_dir = self._run_dir / "checkpoints" / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "round_idx": round_idx,
            "selected_checkpoint": {
                "rel_time_s": best_cp.rel_time_s,
                "min_rpus": list(best_cp.min_rpus),
            },
            "candidates": [
                {
                    "rpu": ", ".join(str(rpu) for rpu in cp.min_rpus),
                    "rel_time_s": cp.rel_time_s,
                    "train_violation": agg.primary_violation(
                        self._slo_objective.slo_metric
                    ),
                    "train_cost": agg.cost,
                    "train_violation_rate": agg.violation_rate,
                    "train_violation_amount_s": agg.violation_amount_s,
                    "train_violation_relative_mean": agg.violation_relative_mean,
                }
                for cp, agg in candidates
            ],
        }
        dump(summary, round_dir / "round_summary.yml")
