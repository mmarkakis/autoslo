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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from rich.table import Table

import autoslo.config.utils as cfgu
from autoslo.clusters.scheduled_spinup import ScheduledSpinUp
from autoslo.config.component_configs import (
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
    all_starts: list[pd.DataFrame] = []
    all_ends: list[pd.DataFrame] = []
    for scenario_id, completions in enumerate(completion_structured_logs):
        completions["slo_s"] = (
            completions["query_text_id"].map(slo_resolver.resolve).fillna(0.0)
        )
        # Precompute per-query violation now so latency_s/slo_s need not be
        # carried through the sort.
        violation_col = np.array(
            slo_objective.slo_metric.calculate_batch(
                [
                    LatencySlo(lat, slo)
                    for lat, slo in zip(
                        completions["latency_s"].to_numpy(),
                        completions["slo_s"].to_numpy(),
                    )
                ]
            ),
            dtype=float,
        )
        base = pd.DataFrame(
            {"scenario_id": scenario_id, "violation": violation_col}
        )
        all_starts.append(
            base.assign(time=completions["arrival_s"].to_numpy(), is_start=True)
        )
        all_ends.append(
            base.assign(
                time=completions["completion_s"].to_numpy(), is_start=False
            )
        )
    events_df = pd.concat(all_starts + all_ends)
    times_raw = events_df["time"].to_numpy()
    order = np.argsort(times_raw, kind="stable")
    times = times_raw[order]
    is_start_arr = events_df["is_start"].to_numpy()[order]
    scenario_ids = events_df["scenario_id"].to_numpy()[order]
    violation_arr = events_df["violation"].to_numpy()[order]

    # Walk the timeline, accumulating (a, b, delinquency_count) for every
    # non-zero-length interval.
    violation_sum: dict[int, float] = defaultdict(float)
    active_count: dict[int, int] = defaultdict(int)
    delinquency_per_workload: dict[int, bool] = {
        scenario_id: False for scenario_id in range(n_scenarios)
    }
    num_delinquent = 0
    spacing = (
        lead_time_s
        if min_candidate_spacing_s is None
        else min_candidate_spacing_s
    )
    candidates: list[float] = []
    in_epoch = False
    epoch_start = 0.0

    def _try_add(t_p: float) -> None:
        if spacing == 0.0 or not candidates or t_p - candidates[-1] >= spacing:
            candidates.append(t_p)

    for i in range(len(times) - 1):
        s_id = int(scenario_ids[i])
        v = float(violation_arr[i])
        if is_start_arr[i]:
            violation_sum[s_id] += v
            active_count[s_id] += 1
        else:
            violation_sum[s_id] -= v
            active_count[s_id] -= 1

        was_delinquent = delinquency_per_workload[s_id]
        is_now_delinquent = not slo_objective.is_met_from_aggregated(
            slo_objective.slo_metric.aggregate_from_running_sum(
                violation_sum[s_id], active_count[s_id]
            )
        )
        if was_delinquent != is_now_delinquent:
            num_delinquent += 1 if is_now_delinquent else -1
            delinquency_per_workload[s_id] = is_now_delinquent

        a = times[i]
        b = times[i + 1]
        if b == a:
            continue  # skip zero-length intervals

        if num_delinquent >= min_delinquent_workloads:
            if not in_epoch:
                in_epoch = True
                epoch_start = a
        elif in_epoch:
            _try_add(max(0.0, epoch_start - lead_time_s))
            in_epoch = False

    if in_epoch:
        _try_add(max(0.0, epoch_start - lead_time_s))

    if verbose:
        if not candidates:
            console.print(
                f"[green]No violating interval found with at least "
                f"{min_delinquent_workloads} delinquent workloads."
            )
        else:
            console.print(
                f"[green]Found {len(candidates)} candidate placement time(s). "
                f"First candidate: t_p={candidates[0]:.1f}s."
            )
    return candidates


# ---------------------------------------------------------------------------
# _CandidateState — mutable per-candidate state for synchronized optimization
# ---------------------------------------------------------------------------


@dataclass
class _CandidateState:
    """Mutable state for one initial-RPU candidate during synchronized
    multi-candidate spin-up optimization."""

    idx: int
    tag: str
    run_dir: Path
    current_config: dict[str, Any]
    current_train_results: list[ExecutionResult] | None = None
    current_train_agg: AggregatedExecutionResults | None = None
    done: bool = False
    spinup_count: int = 0


# ---------------------------------------------------------------------------
# _AttemptProgress — per-candidate progress through placement-time attempts
# ---------------------------------------------------------------------------


@dataclass
class _AttemptProgress:
    """Per-candidate progress through attempts within one round."""

    times: list[float]
    attempt_idx: int = 0


# ---------------------------------------------------------------------------
# SpinupOptimizer
# ---------------------------------------------------------------------------


class SpinupOptimizer:
    """
    Synchronized greedy spin-up placement across multiple initial-RPU
    candidates.

    Instead of running each candidate to completion before starting the next
    (as a serial loop over :class:`SpinupOptimizer` does), this class advances
    all candidates in lock-step, batching their evaluations into single
    :class:`ScenarioEvaluator` calls at every sub-step.  The process pool
    therefore stays fully utilised rather than cycling between single-config
    pool invocations.

    The baseline evaluation, RPU-size attempt evaluations, and (via the
    caller) validation rollouts are all eligible for cross-candidate batching.

    Parameters
    ----------
    evaluator :
        The shared scenario evaluator.
    initial_configs :
        One full config dict per candidate.  Candidates typically differ only
        in ``managed_cluster_pool_config.initial_rpus``.
    run_root :
        Root directory for output. Each candidate writes under
        ``run_root / candidate_{i}``.
    spinup_optimizer_config :
        Shared hyper-parameters for spin-up placement.
    agg_method :
        Aggregation method forwarded to
        :class:`~autoslo.workload_execution.aggregated_execution_results.AggregatedExecutionResults`.
    tuning_slo_objective :
        SLO objective used for candidate ranking.
    verbose_progress :
        Whether to emit per-config rich progress bars inside the evaluator.
    """

    def __init__(
        self,
        evaluator: ScenarioEvaluator,
        spinup_optimizer_config: SpinupOptimizerConfig,
        initial_configs: list[dict[str, Any]],
        run_root: Path,
        agg_method: str,
        tuning_slo_objective: SloObjective,
        verbose_progress: bool = True,
    ) -> None:
        if not initial_configs:
            raise ValueError("initial_configs must be non-empty.")
        self._evaluator = evaluator
        self._verbose_progress = verbose_progress
        self._initial_configs = initial_configs
        self._run_root = run_root
        self._agg_method = agg_method
        self._spinup_optimizer_config = spinup_optimizer_config
        self._lead_time_s = spinup_optimizer_config.lead_time_s

        # SLO Resolver
        slo_resolver_config = SloResolverConfig.from_config(initial_configs[0])
        self._slo_resolver = SloResolver(slo_resolver_config)

        # SLO objective for threshold-aware candidate selection.
        self._tuning_slo_objective = tuning_slo_objective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        train_workload_configs: list[WorkloadConfig],
    ) -> list[tuple[dict[str, Any], AggregatedExecutionResults]]:
        """Run synchronized greedy spin-up placement for all candidates.

        Advances every candidate one round at a time, batching their
        evaluations at each sub-step so that the process pool runs at
        maximum utilisation.

        Parameters
        ----------
        train_workload_configs :
            List of WorkloadConfig objects for training scenarios.

        Returns
        -------
        A list of ``(final_config, train_agg)`` tuples, one per candidate,
        in the same order as *initial_configs*.
        """
        cfg = self._spinup_optimizer_config
        max_spinups = cfg.max_spinups
        min_delinquent_workloads = math.ceil(
            cfg.min_delinquent_workload_fraction * len(train_workload_configs)
        )
        max_attempts_per_round = cfg.max_attempts_per_round
        n_rpus = len(cfg.allowed_rpu_sizes)
        sm = self._tuning_slo_objective.slo_metric

        # Initialise per-candidate state.
        states: list[_CandidateState] = [
            _CandidateState(
                idx=i,
                tag=f"candidate_{i}",
                run_dir=self._run_root / f"candidate_{i}",
                current_config=copy.deepcopy(initial_cfg),
            )
            for i, initial_cfg in enumerate(self._initial_configs)
        ]

        # Persist initial configs.
        for s in states:
            spinup_dir = s.run_dir / "spinups"
            spinup_dir.mkdir(parents=True, exist_ok=True)
            dump_yaml(s.current_config, spinup_dir / "initial_config.yml")

        for round_idx in range(max_spinups):
            active = [s for s in states if not s.done]
            console.rule(
                f"[dim]Spin-up round {round_idx} "
                f"(active candidates: {len(active)})[/]",
                characters="-",
            )
            if not active:
                break

            # ── Step 1: Batch baseline for any state that needs it ────────
            needs_baseline = [
                s for s in active if s.current_train_results is None
            ]
            if needs_baseline:
                batch_baseline = self._evaluator.evaluate_batch_from_configs(
                    progress_bar_label=f"round_{round_idx:03d}_baselines",
                    workload_configs=train_workload_configs,
                    configs=[s.current_config for s in needs_baseline],
                    config_labels=[s.tag for s in needs_baseline],
                    out_dir=(
                        self._run_root / f"round_{round_idx:03d}" / "baselines"
                    ),
                    workload_first=False,
                    verbose_progress=self._verbose_progress,
                )
                for s, results in zip(needs_baseline, batch_baseline):
                    s.current_train_results = results
                    s.current_train_agg = (
                        AggregatedExecutionResults.aggregate_from(
                            results, self._agg_method
                        )
                    )

            # Write per-round initial configs for active candidates.
            for s in active:
                round_dir = s.run_dir / "spinups" / f"round_{round_idx:03d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                dump_yaml(s.current_config, round_dir / "initial_config.yml")

            # ── Step 2: Find candidate spin-up times (CPU-only) ──────────
            candidate_times_per_state: dict[int, list[float]] = {}
            for s in active:
                assert s.current_train_results is not None
                candidate_times_per_state[s.idx] = find_next_spinup_time(
                    s.current_train_results,
                    slo_resolver=self._slo_resolver,
                    slo_objective=self._tuning_slo_objective,
                    min_delinquent_workloads=min_delinquent_workloads,
                    lead_time_s=self._lead_time_s,
                    min_candidate_spacing_s=cfg.min_candidate_spacing_s,
                    verbose=self._verbose_progress,
                )

            # ── Step 3: Mark done — no viable times or budget exceeded ────
            for s in active:
                if not candidate_times_per_state[s.idx]:
                    console.print(
                        f"  [green][candidate {s.idx}] No promising spin-up "
                        f"time found — stopping."
                    )
                    s.done = True
                    continue
                initial_rpus = cfgu.getd(
                    s.current_config,
                    "managed_cluster_pool_config.initial_rpus",
                    [],
                )
                existing_spinups = cfgu.getd(
                    s.current_config, "scheduled_spinups", []
                )
                max_clusters_val = cfgu.getd(
                    s.current_config,
                    "managed_cluster_pool_config.max_clusters",
                    None,
                )
                if max_clusters_val is not None:
                    max_clusters_val = int(max_clusters_val)
                    total_needed = len(initial_rpus) + len(existing_spinups) + 1
                    if total_needed > max_clusters_val:
                        console.print(
                            f"  [yellow][candidate {s.idx}] Cannot place new "
                            f"spin-up: initial_rpus={len(initial_rpus)}, "
                            f"existing_spinups={len(existing_spinups)}, "
                            f"max_clusters={max_clusters_val}."
                        )
                        s.done = True

            active = [s for s in states if not s.done]
            if not active:
                break

            # ── Step 4: Attempt-wave loop ─────────────────────────────────
            attempt_pending: dict[int, _AttemptProgress] = {
                s.idx: _AttemptProgress(times=candidate_times_per_state[s.idx])
                for s in active
            }
            baseline_vcs: dict[int, ViolationCost] = {}
            for s in active:
                assert s.current_train_agg is not None
                baseline_vcs[s.idx] = ViolationCost(
                    s.current_train_agg.primary_violation(sm),
                    s.current_train_agg.cost,
                )
            accepted_in_round: dict[int, bool] = {s.idx: False for s in active}
            # Per-state attempt records accumulated across waves.
            round_attempt_records: dict[int, list[dict]] = {
                s.idx: [] for s in active
            }
            # Snapshot before the wave loop — some states may get marked
            # done inside the loop; we still need to write their summaries.
            active_for_summary = list(active)
            attempt_wave_idx = 0

            while attempt_pending:
                sorted_pending_ids = sorted(attempt_pending.keys())

                # Build combined batch: for each pending state, A configs
                # (one per allowed RPU size at that state's current attempt).
                batch_items: list[
                    tuple[int, ScheduledSpinUp, dict[str, Any]]
                ] = []
                for s_idx in sorted_pending_ids:
                    ap = attempt_pending[s_idx]
                    t_p = ap.times[ap.attempt_idx]
                    state = states[s_idx]
                    for rpu in cfg.allowed_rpu_sizes:
                        spinup = ScheduledSpinUp(
                            rel_time_s=max(0.0, t_p), rpu=rpu
                        )
                        batch_items.append(
                            (
                                s_idx,
                                spinup,
                                add_spinup_to_config(
                                    config=state.current_config,
                                    spinup=spinup,
                                ),
                            )
                        )

                wave_configs = [item[2] for item in batch_items]
                wave_labels = [
                    f"c{s_idx}_a{attempt_pending[s_idx].attempt_idx:03d}_rpu{su.rpu}"
                    for (s_idx, su, _) in batch_items
                ]
                wave_out_dir = (
                    self._run_root
                    / f"round_{round_idx:03d}"
                    / f"attempt_wave_{attempt_wave_idx:03d}"
                )
                wave_results = self._evaluator.evaluate_batch_from_configs(
                    progress_bar_label=(
                        f"round_{round_idx:03d}_wave_{attempt_wave_idx:03d}"
                    ),
                    workload_configs=train_workload_configs,
                    configs=wave_configs,
                    config_labels=wave_labels,
                    out_dir=wave_out_dir,
                    workload_first=False,
                    verbose_progress=self._verbose_progress,
                )

                # Process results per state, in the same sorted order used
                # when building batch_items so that slice offsets are correct.
                offset = 0
                next_pending: dict[int, _AttemptProgress] = {}
                for s_idx in sorted_pending_ids:
                    ap = attempt_pending[s_idx]
                    times, attempt_idx = ap.times, ap.attempt_idx
                    state = states[s_idx]
                    t_p = times[attempt_idx]
                    baseline_vc = baseline_vcs[s_idx]

                    # Slice the n_rpus results for this state.
                    state_results = wave_results[offset : offset + n_rpus]
                    state_items = batch_items[offset : offset + n_rpus]
                    offset += n_rpus

                    cands: list[
                        tuple[ScheduledSpinUp, AggregatedExecutionResults]
                    ] = []
                    for (_, su, _), trial_results in zip(
                        state_items, state_results
                    ):
                        agg = AggregatedExecutionResults.aggregate_from(
                            trial_results, self._agg_method
                        )
                        cands.append((su, agg))

                    best_local_idx = self._tuning_slo_objective.idx_of_best(
                        [
                            ViolationCost(agg.primary_violation(sm), agg.cost)
                            for _, agg in cands
                        ]
                    )
                    best_su, best_agg = cands[best_local_idx]
                    best_vc = ViolationCost(
                        best_agg.primary_violation(sm), best_agg.cost
                    )
                    accepted = (
                        self._tuning_slo_objective.cmp(best_vc, baseline_vc) < 0
                    )

                    record: dict = {
                        "attempt_idx": attempt_idx,
                        "placement_time": t_p,
                        "outcome": "accepted" if accepted else "rejected",
                        "best_rpu": best_su.rpu,
                        "best_violation": best_vc.violation,
                        "best_cost": best_vc.cost,
                    }
                    round_attempt_records[s_idx].append(record)

                    if accepted:
                        self._print_candidate_table(
                            s_idx,
                            round_idx,
                            attempt_idx,
                            t_p,
                            cands,
                            best_su,
                        )
                        assert state.current_train_agg is not None
                        AggregatedExecutionResults.print_comparison(
                            (
                                f"[candidate {s_idx}] Current",
                                state.current_train_agg,
                            ),
                            (
                                f"[candidate {s_idx}] Best candidate",
                                best_agg,
                            ),
                            agg_method=self._agg_method,
                            slo_metric=sm,
                            console=console,
                        )
                        console.print(
                            f"  [green][candidate {s_idx} / "
                            f"round {round_idx} / attempt {attempt_idx}] "
                            f"Accepted: violation "
                            f"{baseline_vc.violation:.6f} \u2192 "
                            f"{best_vc.violation:.6f}, "
                            f"cost {baseline_vc.cost:.4f} \u2192 "
                            f"{best_vc.cost:.4f}."
                        )
                        # Carry the winning results forward to the next round.
                        state.current_config = wave_configs[
                            offset - n_rpus + best_local_idx
                        ]
                        state.current_train_results = wave_results[
                            offset - n_rpus + best_local_idx
                        ]
                        state.current_train_agg = best_agg
                        state.spinup_count += 1
                        accepted_in_round[s_idx] = True
                        # Accepted — do not add to next_pending.
                    else:
                        console.print(
                            f"  [red][candidate {s_idx} / "
                            f"round {round_idx} / attempt {attempt_idx}] "
                            f"Rejected t_p={t_p:.1f}s: "
                            f"best RPU={best_su.rpu}, "
                            f"violation "
                            f"{baseline_vc.violation:.6f} \u2192 "
                            f"{best_vc.violation:.6f}, "
                            f"cost {baseline_vc.cost:.4f} \u2192 "
                            f"{best_vc.cost:.4f}."
                        )
                        next_attempt_idx = attempt_idx + 1
                        if next_attempt_idx < min(
                            len(times), max_attempts_per_round
                        ):
                            next_pending[s_idx] = _AttemptProgress(
                                times=times, attempt_idx=next_attempt_idx
                            )
                        else:
                            # All attempts for this candidate exhausted.
                            console.print(
                                f"  [red][candidate {s_idx}] "
                                f"Round {round_idx}: all "
                                f"{attempt_idx + 1} attempt(s) exhausted "
                                f"\u2014 stopping."
                            )
                            state.done = True

                attempt_pending = next_pending
                attempt_wave_idx += 1

            # ── Step 5: Persist per-round summaries and final configs ─────
            for s in active_for_summary:
                round_dir = s.run_dir / "spinups" / f"round_{round_idx:03d}"
                summary = {
                    "round_idx": round_idx,
                    "outcome": (
                        "accepted"
                        if accepted_in_round.get(s.idx, False)
                        else "rejected"
                    ),
                    "attempts": round_attempt_records.get(s.idx, []),
                }
                dump_yaml(summary, round_dir / "round_summary.yml")
                dump_yaml(s.current_config, round_dir / "final_config.yml")

        # Write the final config for every candidate and build the return list.
        out: list[tuple[dict[str, Any], AggregatedExecutionResults]] = []
        for s in states:
            spinup_dir = s.run_dir / "spinups"
            dump_yaml(s.current_config, spinup_dir / "final_config.yml")
            assert s.current_train_agg is not None, (
                f"Candidate {s.idx}: current_train_agg is None after the "
                f"optimization loop.  Ensure max_spinups > 0."
            )
            out.append((s.current_config, s.current_train_agg))
        return out

    # ------------------------------------------------------------------
    # Rich output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_candidate_table(
        candidate_idx: int,
        round_idx: int,
        attempt_idx: int,
        placement_time: float,
        candidates: list[tuple[ScheduledSpinUp, AggregatedExecutionResults]],
        best_su: ScheduledSpinUp,
    ) -> None:
        table = Table(
            title=(
                f"Candidate {candidate_idx} / Round {round_idx} / "
                f"Attempt {attempt_idx} (t_p={placement_time:.0f}s)"
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
