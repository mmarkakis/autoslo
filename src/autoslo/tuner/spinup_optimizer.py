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
from rich.console import Console
from rich.table import Table

import autoslo.config.utils as cfgu
from autoslo.clusters.scheduled_spinup import ScheduledSpinUp
from autoslo.config.component_configs import (
    SpinupOptimizerConfig,
    SloObjectiveConfig,
    SloResolverConfig,
    WorkloadConfig,
)
from autoslo.filesystem.yaml_helpers import dump_yaml
from autoslo.simulator.aggregated_simulation_results import (
    AggregatedSimulationResults,
)
from autoslo.simulator.simulation_result import SimulationResult
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator

logger = logging.getLogger(__name__)
console = Console()


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
    merged = existing + [{"rel_time_s": spinup.rel_time_s, "rpu": spinup.rpu}]
    return cfgu.copy_and_apply_overrides(config, {dot_delimited_key: merged})


def find_next_spinup_time(
    results: list[SimulationResult],
    slo_resolver: SloResolver,
    slo_objective: SloObjective,
    min_delinquent_workloads: int,
    lead_time_s: float,
) -> Optional[float]:
    """
    Find the next time at which to insert a scheduled spin-up, or None
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
    4. Return the start time of that interval minus *lead_time_s*.

    """
    completion_structured_logs = []
    for result in results:
        log_path = result.simulation_dir / "structured_log.parquet"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")

        # Read arrivals and completions, then pivot to compute latency
        # (latency_s lives inside the JSON ``details`` column, not as a
        # top-level field).
        log = pd.read_parquet(
            log_path,
            columns=[
                "rel_time_s",
                "event_type",
                "query_id",
                "query_text_id",
                "details",
            ],
        )
        log = log[log["event_type"].isin({"arrival", "completion"})]

        # Pivot to get arrival and completion times per query.
        pivoted = log.pivot(
            index=["query_id", "query_text_id"],
            columns="event_type",
            values="rel_time_s",
        )
        if (
            "completion" not in pivoted.columns
            or pivoted["completion"].dropna().empty
        ):
            raise ValueError(
                f"No successful completion events in log: {log_path}"
            )

        completions = pd.DataFrame(
            {
                "query_id": pivoted.index.get_level_values("query_id"),
                "query_text_id": pivoted.index.get_level_values(
                    "query_text_id"
                ),
                "rel_time_s": pivoted["completion"].values,
                "latency_s": (
                    pivoted["completion"] - pivoted["arrival"]
                ).values,
            }
        ).dropna(subset=["rel_time_s", "latency_s"])
        completion_structured_logs.append(completions)

    return find_next_spinup_time_df(
        completion_structured_logs=completion_structured_logs,
        slo_resolver=slo_resolver,
        slo_objective=slo_objective,
        min_delinquent_workloads=min_delinquent_workloads,
        lead_time_s=lead_time_s,
    )


def find_next_spinup_time_df(
    completion_structured_logs: list[pd.DataFrame],
    slo_resolver: SloResolver,
    slo_objective: SloObjective,
    min_delinquent_workloads: int,
    lead_time_s: float,
) -> Optional[float]:
    """
    Internal helper for find_next_spinup_time that takes pre-loaded
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
            spinup_time = this_event_time - lead_time_s
            console.print(
                f"[green]Found violating interval: "
                f"{this_event_time:.2f}s to {next_event_time:.2f}s  "
                f"with {num_delinquent_workloads} delinquent workloads. "
                f"Suggesting spin-up at {spinup_time:.2f}s."
            )
            return max(0, spinup_time)

    console.print(
        f" [green]No violating interval found with at least "
        f"{min_delinquent_workloads} delinquent workloads."
    )

    return None


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
    ) -> None:
        self._evaluator = evaluator
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
    ) -> tuple[dict[str, Any], AggregatedSimulationResults]:
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

        spinup_dir = self._run_dir / "spinups"

        current_config = copy.deepcopy(self._config)
        dump_yaml(current_config, spinup_dir / "initial_config.yml")

        # Track evaluation results for the current config to avoid
        # re-evaluation across rounds.  When a candidate is accepted,
        # its results carry forward as the next round's baseline.
        current_train_results: list[SimulationResult] | None = None
        current_train_agg: AggregatedSimulationResults | None = None

        for round_idx in range(max_spinups):
            console.rule(f"[dim]Spin-up round {round_idx}[/]", characters="-")
            round_dir = spinup_dir / f"round_{round_idx:03d}"
            dump_yaml(current_config, round_dir / "initial_config.yml")

            # 1. Get baseline results (reuse from previous round
            #    if available).
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
                    )
                )
                train_results = nested_train_results[0]
                agg_train_results = AggregatedSimulationResults.aggregate_from(
                    train_results, self._agg_method
                )
                current_train_agg = agg_train_results

            # 2. Find earliest promising spin-up time.
            next_spinup_time = find_next_spinup_time(
                train_results,
                slo_resolver=self._slo_resolver,
                slo_objective=self._slo_objective,
                min_delinquent_workloads=min_delinquent_workloads,
                lead_time_s=self._lead_time_s,
            )

            if next_spinup_time is None:
                s = "[green]No promising spin-up time found — stopping."
                console.print(s)
                break

            # 3. Try each RPU size.
            cands: list[
                tuple[ScheduledSpinUp, AggregatedSimulationResults]
            ] = []
            spinups = []
            all_configs = []
            initial_rpus = cfgu.getd(
                current_config, "managed_cluster_pool_config.initial_rpus", []
            )
            existing_spinups = cfgu.getd(
                current_config, "scheduled_spinups", []
            )
            max_clusters = int(
                cfgu.getd(
                    current_config,
                    "autoscaling_config.max_clusters",
                    10,
                )
            )
            total_clusters_needed = (
                len(initial_rpus) + len(existing_spinups) + 1
            )  # +1 for the new spin-up
            if total_clusters_needed > max_clusters:
                console.print(
                    f"[yellow]Cannot place new spin-up because the initial "
                    f"setup requires {len(initial_rpus)} clusters and existing "
                    f"spin-ups reserve {len(existing_spinups)} clusters, "
                    f"while the max_clusters budget is {max_clusters}."
                )
                break

            for rpu in self._spinup_optimizer_config.allowed_rpu_sizes:
                spinup = ScheduledSpinUp(
                    rel_time_s=max(0.0, next_spinup_time),
                    rpu=rpu,
                )
                spinups.append(spinup)
                trial_config = add_spinup_to_config(
                    config=current_config, spinup=spinup
                )
                all_configs.append(trial_config)
            all_trial_results = self._evaluator.evaluate_batch_from_configs(
                progress_bar_label=f"round_{round_idx:03d}_candidates",
                workload_configs=train_workload_configs,
                configs=all_configs,
                out_dir=round_dir / "train",
                workload_first=False,
            )
            for i in range(len(spinups)):
                spinup = spinups[i]
                trial_results = all_trial_results[i]
                agg = AggregatedSimulationResults.aggregate_from(
                    trial_results, self._agg_method
                )
                cands.append((spinup, agg))

            # 4. Pick best on training set.
            sm = self._slo_objective.slo_metric
            best_idx = self._slo_objective.idx_of_best(
                [
                    ViolationCost(agg.primary_violation(sm), agg.cost)
                    for _, agg in cands
                ]
            )
            best_su, _ = cands[best_idx]
            self._print_candidate_table(round_idx, cands, best_su)

            AggregatedSimulationResults.print_comparison(
                ("Current config", agg_train_results),
                ("Best candidate", cands[best_idx][1]),
                agg_method=self._agg_method,
                slo_metric=sm,
                console=console,
            )

            # 5. Accept the best spin-up only if it improves on the baseline
            # under the canonical SloObjective ordering AND clears the minimum
            # relative improvement threshold in the decisive dimension.
            # When the baseline is infeasible the decisive dimension is
            # violation; when it is feasible the decisive dimension is cost
            # (mirroring the SloObjective.cmp contract exactly — see
            # _is_improvement_large_enough for the full case analysis).
            baseline_vc = ViolationCost(
                agg_train_results.primary_violation(sm),
                agg_train_results.cost,
            )
            best_vc = ViolationCost(
                cands[best_idx][1].primary_violation(sm),
                cands[best_idx][1].cost,
            )
            if not self._slo_objective.is_sufficient_improvement(
                baseline_vc,
                best_vc,
                self._spinup_optimizer_config.min_rel_improvement_for_acceptance,
            ):
                console.print(
                    f" [red]Rejecting: best candidate does not improve on the "
                    f"baseline sufficiently "
                    f"(violation {best_vc.violation:.6f} vs "
                    f"{baseline_vc.violation:.6f}, "
                    f"cost {best_vc.cost:.4f} vs {baseline_vc.cost:.4f})."
                )
                self._write_round_summary(round_idx, cands, best_su)
                dump_yaml(current_config, round_dir / "final_config.yml")
                break

            current_config = all_configs[best_idx]
            # Carry forward the accepted candidate's results so the
            # next round can skip re-evaluating the same config.
            current_train_results = all_trial_results[best_idx]
            current_train_agg = cands[best_idx][1]
            console.print(
                f"  [green]Accepted: violation "
                f"{baseline_vc.violation:.6f} → {best_vc.violation:.6f}, "
                f"cost {baseline_vc.cost:.4f} → {best_vc.cost:.4f}."
            )

            # Write round summary.
            self._write_round_summary(round_idx, cands, best_su)
            dump_yaml(current_config, round_dir / "final_config.yml")

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
        candidates: list[
            tuple[
                ScheduledSpinUp,
                AggregatedSimulationResults,
            ]
        ],
        best_su: ScheduledSpinUp,
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
        candidates: list[
            tuple[
                ScheduledSpinUp,
                AggregatedSimulationResults,
            ]
        ],
        best_su: ScheduledSpinUp,
    ) -> None:
        round_dir = self._run_dir / "spinups" / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "round_idx": round_idx,
            "selected_spinup": {
                "rel_time_s": best_su.rel_time_s,
                "rpu": best_su.rpu,
            },
            "candidates": [
                {
                    "rpu": su.rpu,
                    "rel_time_s": su.rel_time_s,
                    "train_violation": agg.primary_violation(
                        self._slo_objective.slo_metric
                    ),
                    "train_cost": agg.cost,
                    "train_violation_rate": agg.violation_rate,
                    "train_violation_amount_s": agg.violation_amount_s,
                    "train_violation_relative_mean": agg.violation_relative_mean,
                }
                for su, agg in candidates
            ],
        }
        dump_yaml(summary, round_dir / "round_summary.yml")
