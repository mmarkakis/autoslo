"""Greedy scheduled spin-up optimiser — design step 4.

Iteratively places :class:`ScheduledSpinUp` at the earliest
sliding-window with a violation rate above the configured threshold,
tries every allowed RPU size, picks the best on training data, and
validates on held-out scenarios.  Stops when the budget is exhausted,
no violating window remains, or validation improvement is below
epsilon.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rich.table import Table

import autoslo.config.utils as cfgu
from autoslo.clusters.scheduled_spinup import ScheduledSpinUp
from autoslo.config.component_configs import (
    SloObjectiveConfig,
    SloResolverConfig,
    SpinupOptimizerConfig,
    WorkloadConfig,
)
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.filesystem.yaml_helpers import dump_yaml
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.tuner_console import console
from autoslo.workload_execution.aggregated_execution_results import (
    AggregatedExecutionResults,
)
from autoslo.workload_execution.execution_result import ExecutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_spinup_to_config(
    config: dict[str, Any],
    spinup: ScheduledSpinUp,
) -> dict[str, Any]:
    """Return a copy of *config* with *spinup* appended.

    If ``spinup.rel_time_s == 0`` the RPU is folded into
    ``managed_cluster_pool_config.initial_rpus`` instead.
    """
    initial_rpus_key = "managed_cluster_pool_config.initial_rpus"
    if spinup.rel_time_s == 0:
        initial_rpus = copy.deepcopy(cfgu.getd(config, initial_rpus_key, []))
        initial_rpus.append(spinup.rpu)
        return cfgu.copy_and_apply_overrides(
            config, {initial_rpus_key: sorted(initial_rpus)}
        )

    dot_delimited_key = "scheduled_spinups"
    existing = cfgu.getd(config, dot_delimited_key, [])
    merged = sorted(
        existing + [{"rel_time_s": spinup.rel_time_s, "rpu": spinup.rpu}],
        key=lambda s: s["rel_time_s"],
    )
    return cfgu.copy_and_apply_overrides(config, {dot_delimited_key: merged})


def find_next_spinup_time(
    results: list[ExecutionResult],
    slo_resolver: SloResolver,
    slo_objective: SloObjective,
    min_delinquent_workloads: int,
    lead_time_s: float,
    min_candidate_spacing_s: Optional[float] = None,
    verbose: bool = True,
) -> list[float]:
    """Return a list of candidate placement times.

    Each element represents a candidate scheduled-spin-up placement time.
    An empty list means no viable placement time exists.

    ``min_candidate_spacing_s`` is forwarded to
    :func:`find_next_spinup_time_df`; see that function for details.

    See :func:`find_next_spinup_time_df` for the full algorithm.
    """
    completion_structured_logs = []
    for result in results:
        log_path = result.execution_dir / "structured_log.parquet"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")

        completions = StructuredLog.load(log_path).query_latencies(
            drop_incomplete=True
        )
        if completions.empty:
            raise ValueError(
                f"No successful completion events in log: {log_path}"
            )
        completion_structured_logs.append(completions)

    return find_next_spinup_time_df(
        completion_structured_logs=completion_structured_logs,
        slo_resolver=slo_resolver,
        slo_objective=slo_objective,
        min_delinquent_workloads=min_delinquent_workloads,
        lead_time_s=lead_time_s,
        min_candidate_spacing_s=min_candidate_spacing_s,
        verbose=verbose,
    )


def find_next_spinup_time_df(
    completion_structured_logs: list[pd.DataFrame],
    slo_resolver: SloResolver,
    slo_objective: SloObjective,
    min_delinquent_workloads: int,
    lead_time_s: float,
    min_candidate_spacing_s: Optional[float] = None,
    verbose: bool = True,
) -> list[float]:
    """Return a list of candidate placement times.

    Each element represents a candidate scheduled-spin-up placement time.
    An empty list means no viable placement time exists.

    Algorithm
    ---------
    1. Build a unified event timeline (query-start / query-end) across all
       scenarios, maintaining per-scenario delinquency state as before.
    2. Accumulate all non-zero-length intervals with their delinquency count.
    3. Identify congestion *epochs*: maximal contiguous runs of intervals
       where ``count >= min_delinquent_workloads``.  Each epoch contributes
       one candidate placement time ``max(0, epoch_start - lead_time_s)``.
       Epochs that collapse to the same placement time (e.g. multiple early
       epochs all below ``lead_time_s``) are deduplicated.
     4. Greedily drop candidates that are within ``min_candidate_spacing_s``
         of an already-retained candidate. Candidates are considered in
         chronological detection order.
       ``None`` uses ``lead_time_s`` as the spacing threshold.
    """
    # Create events per scenario and form a single timeline.
    n_scenarios = len(completion_structured_logs)
    events = []
    for scenario_id, completions in enumerate(completion_structured_logs):

        completions["slo_s"] = (
            completions["query_text_id"].map(slo_resolver.resolve).fillna(0.0)
        )

        for _, row in completions.iterrows():
            events.append(
                {
                    "time": row["arrival_s"],
                    "event_type": "start",
                    "latency_s": row["latency_s"],
                    "slo_s": row["slo_s"],
                    "scenario_id": scenario_id,
                    "query_id": row["query_id"],
                }
            )
            events.append(
                {
                    "time": row["completion_s"],
                    "event_type": "end",
                    "latency_s": row["latency_s"],
                    "slo_s": row["slo_s"],
                    "scenario_id": scenario_id,
                    "query_id": row["query_id"],
                }
            )
    events.sort(key=lambda x: x["time"])

    # Walk the timeline, accumulating (a, b, delinquency_count) for every
    # non-zero-length interval.
    active_queries: dict[int, dict[str, LatencySlo]] = defaultdict(dict)
    delinquency_per_workload: dict[int, bool] = {
        scenario_id: False for scenario_id in range(n_scenarios)
    }
    intervals: list[tuple[float, float, int]] = []  # (a, b, count)

    for i in range(len(events) - 1):
        event = events[i]
        if event["event_type"] == "start":
            active_queries[event["scenario_id"]][event["query_id"]] = (
                LatencySlo(event["latency_s"], event["slo_s"])
            )
        else:
            active_queries[event["scenario_id"]].pop(event["query_id"], None)

        scenario_active_queries = list(
            active_queries[event["scenario_id"]].values()
        )
        if scenario_active_queries:
            delinquency_per_workload[event["scenario_id"]] = (
                not slo_objective.is_met(
                    per_query_latency_slo=scenario_active_queries
                )
            )
        else:
            delinquency_per_workload[event["scenario_id"]] = False

        a = event["time"]
        b = events[i + 1]["time"]
        if b == a:
            continue  # skip zero-length intervals

        num_delinquent = sum(delinquency_per_workload.values())
        intervals.append((a, b, num_delinquent))

    # Identify congestion epochs and derive one placement time per epoch.
    epoch_placement_times: list[float] = []
    idx = 0
    while idx < len(intervals):
        a, b, count = intervals[idx]
        if count >= min_delinquent_workloads:
            epoch_start = a
            while (
                idx < len(intervals)
                and intervals[idx][2] >= min_delinquent_workloads
            ):
                idx += 1
            epoch_placement_times.append(max(0.0, epoch_start - lead_time_s))
        else:
            idx += 1

    # Deduplicate while preserving first-occurrence order.
    unique_placement_times: list[float] = list(
        dict.fromkeys(epoch_placement_times)
    )

    if not unique_placement_times:
        console.print(
            f"[green]No violating interval found with at least "
            f"{min_delinquent_workloads} delinquent workloads."
        )
        return []

    candidates = unique_placement_times

    # Greedy spacing deduplication: discard candidates within
    # min_candidate_spacing_s of an already-retained candidate.
    spacing = (
        lead_time_s
        if min_candidate_spacing_s is None
        else min_candidate_spacing_s
    )

    if spacing > 0.0:
        retained: list[float] = []
        for t_p in candidates:
            if all(abs(t_p - r_t) >= spacing for r_t in retained):
                retained.append(t_p)
        candidates = retained

    if not candidates:
        if verbose:
            console.print(
                "[green]No viable candidate times remain after spacing "
                f"deduplication (min_candidate_spacing_s={spacing:.1f}s)."
            )
        return []

    if verbose:
        console.print(
            f"[green]Found {len(candidates)} candidate placement time(s). "
            f"First candidate: t_p={candidates[0]:.1f}s."
        )
    return candidates


# ---------------------------------------------------------------------------
# SpinupOptimizer
# ---------------------------------------------------------------------------


class SpinupOptimizer:
    """Greedy scheduled spin-up placement.

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
        spinup_optimizer_config: SpinupOptimizerConfig,
        config: dict[str, Any],
        run_dir: Path,
        agg_method: str,
        verbose_progress: bool = True,
    ) -> None:
        self._evaluator = evaluator
        self._verbose_progress = verbose_progress
        self._config = config
        self._run_dir = run_dir
        self._agg_method = agg_method
        self._spinup_optimizer_config = spinup_optimizer_config
        self._lead_time_s = spinup_optimizer_config.lead_time_s

        # SLO Resolver
        slo_resolver_config = SloResolverConfig.from_config(config)
        self._slo_resolver = SloResolver(slo_resolver_config)

        # SLO objective for threshold-aware candidate selection.
        slo_objective_config = SloObjectiveConfig.from_config(config)
        self._slo_objective = SloObjective(slo_objective_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        train_workload_configs: list[WorkloadConfig],
    ) -> tuple[dict[str, Any], AggregatedExecutionResults]:
        """Run the greedy spin-up placement loop.

        Spin-ups are placed greedily using training data only.
        Validation is deferred to the caller.

        Parameters
        ----------
        train_workload_configs :
            List of WorkloadConfig objects for training scenarios.

        Returns
        -------
        A tuple of ``(config, train_agg)`` where *config* is the full
        config dict with the best spin-up schedule applied and
        *train_agg* is the aggregated training results for that config.
        """
        max_spinups = self._spinup_optimizer_config.max_spinups
        min_delinquent_workload_fraction = (
            self._spinup_optimizer_config.min_delinquent_workload_fraction
        )
        min_delinquent_workloads = math.ceil(
            min_delinquent_workload_fraction * len(train_workload_configs)
        )
        max_attempts_per_round = (
            self._spinup_optimizer_config.max_attempts_per_round
        )

        spinup_dir = self._run_dir / "spinups"

        current_config = copy.deepcopy(self._config)
        dump_yaml(current_config, spinup_dir / "initial_config.yml")

        # Track evaluation results for the current config to avoid
        # re-evaluation across rounds.  When a candidate is accepted,
        # its results carry forward as the next round's baseline.
        current_train_results: list[ExecutionResult] | None = None
        current_train_agg: AggregatedExecutionResults | None = None

        for round_idx in range(max_spinups):
            console.rule(f"[dim]Spin-up round {round_idx}[/]", characters="-")
            round_dir = spinup_dir / f"round_{round_idx:03d}"
            dump_yaml(current_config, round_dir / "initial_config.yml")

            # 1. Get baseline results (reuse from previous round if available).
            if current_train_results is not None:
                train_results = current_train_results
                assert current_train_agg is not None
                agg_train_results = current_train_agg
            else:
                nested_train_results = (
                    self._evaluator.evaluate_batch_from_configs(
                        progress_bar_label=f"round_{round_idx:03d}_baseline",
                        workload_configs=train_workload_configs,
                        configs=[current_config],
                        out_dir=round_dir / "baseline",
                        workload_first=False,
                        verbose_progress=self._verbose_progress,
                    )
                )
                train_results = nested_train_results[0]
                agg_train_results = AggregatedExecutionResults.aggregate_from(
                    train_results, self._agg_method
                )
                current_train_agg = agg_train_results

            # 2. Rank all viable candidate placement times.
            candidate_times = find_next_spinup_time(
                train_results,
                slo_resolver=self._slo_resolver,
                slo_objective=self._slo_objective,
                min_delinquent_workloads=min_delinquent_workloads,
                lead_time_s=self._lead_time_s,
                min_candidate_spacing_s=self._spinup_optimizer_config.min_candidate_spacing_s,
                verbose=self._verbose_progress,
            )

            if not candidate_times:
                console.print(
                    "[green]No promising spin-up time found — stopping."
                )
                break

            # 3. Check cluster budget before trying any candidate.
            initial_rpus = cfgu.getd(
                current_config, "managed_cluster_pool_config.initial_rpus", []
            )
            existing_spinups = cfgu.getd(
                current_config, "scheduled_spinups", []
            )
            max_clusters = int(
                cfgu.getd(current_config, "autoscaling_config.max_clusters", 10)
            )
            total_clusters_needed = (
                len(initial_rpus) + len(existing_spinups) + 1
            )
            if total_clusters_needed > max_clusters:
                console.print(
                    f"[yellow]Cannot place new spin-up: initial setup requires "
                    f"{len(initial_rpus)} clusters, existing spin-ups reserve "
                    f"{len(existing_spinups)}, max_clusters={max_clusters}."
                )
                break

            sm = self._slo_objective.slo_metric
            baseline_vc = ViolationCost(
                agg_train_results.primary_violation(sm),
                agg_train_results.cost,
            )

            accepted_in_round = False
            attempt_records: list[dict] = []
            candidates_to_try = candidate_times[:max_attempts_per_round]

            for attempt_idx, placement_time in enumerate(candidates_to_try):
                attempt_dir = round_dir / f"attempt_{attempt_idx:03d}"

                # 4. Try each RPU size at this placement time.
                spinups = []
                all_configs = []
                for rpu in self._spinup_optimizer_config.allowed_rpu_sizes:
                    spinup = ScheduledSpinUp(
                        rel_time_s=max(0.0, placement_time),
                        rpu=rpu,
                    )
                    spinups.append(spinup)
                    all_configs.append(
                        add_spinup_to_config(
                            config=current_config, spinup=spinup
                        )
                    )

                all_trial_results = self._evaluator.evaluate_batch_from_configs(
                    progress_bar_label=(
                        f"round_{round_idx:03d}_attempt_{attempt_idx:03d}"
                    ),
                    workload_configs=train_workload_configs,
                    configs=all_configs,
                    out_dir=attempt_dir / "train",
                    workload_first=False,
                    verbose_progress=self._verbose_progress,
                )
                cands: list[
                    tuple[ScheduledSpinUp, AggregatedExecutionResults]
                ] = []
                for spinup, trial_results in zip(spinups, all_trial_results):
                    agg = AggregatedExecutionResults.aggregate_from(
                        trial_results, self._agg_method
                    )
                    cands.append((spinup, agg))

                # 5. Pick best on training set.
                best_idx = self._slo_objective.idx_of_best(
                    [
                        ViolationCost(agg.primary_violation(sm), agg.cost)
                        for _, agg in cands
                    ]
                )
                best_su, best_agg = cands[best_idx]
                best_vc = ViolationCost(
                    best_agg.primary_violation(sm), best_agg.cost
                )

                # 6. Accept if the best candidate improves on the baseline.
                accepted = self._slo_objective.cmp(best_vc, baseline_vc) < 0

                attempt_records.append(
                    {
                        "attempt_idx": attempt_idx,
                        "placement_time": placement_time,
                        "outcome": "accepted" if accepted else "rejected",
                        "best_rpu": best_su.rpu,
                        "best_violation": best_vc.violation,
                        "best_cost": best_vc.cost,
                    }
                )

                if accepted:
                    self._print_candidate_table(
                        round_idx,
                        attempt_idx,
                        placement_time,
                        cands,
                        best_su,
                    )
                    AggregatedExecutionResults.print_comparison(
                        ("Current config", agg_train_results),
                        ("Best candidate", best_agg),
                        agg_method=self._agg_method,
                        slo_metric=sm,
                        console=console,
                    )
                    console.print(
                        f"  [green][round {round_idx} / attempt {attempt_idx}] "
                        f"Accepted: violation "
                        f"{baseline_vc.violation:.6f} → {best_vc.violation:.6f}, "
                        f"cost {baseline_vc.cost:.4f} → {best_vc.cost:.4f}."
                    )
                    current_config = all_configs[best_idx]
                    current_train_results = all_trial_results[best_idx]
                    current_train_agg = best_agg
                    accepted_in_round = True
                    break
                else:
                    console.print(
                        f"  [red][round {round_idx} / attempt {attempt_idx}] "
                        f"Rejected t_p={placement_time:.1f}s: "
                        f"best RPU={best_su.rpu}, "
                        f"violation {baseline_vc.violation:.6f} → {best_vc.violation:.6f}, "
                        f"cost {baseline_vc.cost:.4f} → {best_vc.cost:.4f}."
                    )

            self._write_round_summary(
                round_idx, attempt_records, accepted_in_round
            )
            dump_yaml(current_config, round_dir / "final_config.yml")

            if not accepted_in_round:
                console.print(
                    f"[red]Round {round_idx}: all {len(candidates_to_try)} "
                    f"candidate(s) exhausted without acceptance — stopping."
                )
                break

        # Write the final config.
        dump_yaml(current_config, spinup_dir / "final_config.yml")
        assert (
            current_train_agg is not None
        ), "No spin-up rounds were configured (max_spinups=0)."
        return current_config, current_train_agg

    # ------------------------------------------------------------------
    # Rich output helpers
    # ------------------------------------------------------------------

    def _print_candidate_table(
        self,
        round_idx: int,
        attempt_idx: int,
        placement_time: float,
        candidates: list[
            tuple[
                ScheduledSpinUp,
                AggregatedExecutionResults,
            ]
        ],
        best_su: ScheduledSpinUp,
    ) -> None:
        table = Table(
            title=(
                f"Round {round_idx} / Attempt {attempt_idx} — Candidate RPU Sizes "
                f"(t_p={placement_time:.0f}s)"
            ),
            show_lines=True,
        )
        table.add_column("RPU", justify="right")
        table.add_column("rel_time_s", justify="right")
        table.add_column("Viol. Rate", justify="right")
        table.add_column("Viol. Amount (s)", justify="right")
        table.add_column("Viol. Relative", justify="right")
        table.add_column("Train Cost ($)", justify="right")
        table.add_column("Selected", justify="center")

        for su, agg in candidates:
            is_best = su == best_su
            style = "bold green" if is_best else ""
            table.add_row(
                str(su.rpu),
                f"{su.rel_time_s:.0f}",
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
        attempts: list[dict],
        accepted: bool,
    ) -> None:
        round_dir = self._run_dir / "spinups" / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "round_idx": round_idx,
            "outcome": "accepted" if accepted else "rejected",
            "attempts": attempts,
        }
        dump_yaml(summary, round_dir / "round_summary.yml")
