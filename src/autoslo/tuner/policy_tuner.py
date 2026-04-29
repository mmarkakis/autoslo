"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotext as plt
import yaml
from rich.console import Console

import autoslo.utils.config as cfgu
import autoslo.utils.paths as pu
from autoslo.config.component_configs import (
    AutoscalerConfig,
    ForecasterConfig,
    ManagedClusterPoolConfig,
    ProvisionerConfig,
    QueryRouterConfig,
    ReservoirConfig,
    SamplingConfig,
    SloObjectiveConfig,
    SloResolverConfig,
    WorkloadConfig,
    WorkloadRunnerConfig,
)
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.checkpoint_optimizer import CheckpointOptimizer
from autoslo.tuner.forecaster import Forecaster
from autoslo.tuner.param_sweep import ParamSweep
from autoslo.tuner.policy_tuner_timer import PolicyTunerTimer
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.tuner_utils import (
    AggregatedSimulationResults,
    SimulationResult,
)
from autoslo.utils.config import copy_and_apply_overrides
from autoslo.utils.structured_events import wall_clock_utc
from autoslo.utils.yaml_helpers import dump_yaml
from autoslo.workload_definition.workload import Workload

logger = logging.getLogger(__name__)
console = Console()


class PolicyTuner:
    """Orchestrates the end-to-end policy tuning pipeline."""

    def __init__(
        self,
        initial_config: dict,
        run_id: Optional[str] = None,
        force: bool = False,
    ) -> None:
        self._initial_config = initial_config
        self._force = force

        # ── Determine run_id and set up logging ──────────────────────────────
        run_id = run_id or str(int(wall_clock_utc() * 1000))
        self._out_dir = Path(
            os.path.join(pu.get_data_path(), "tuner_runs", run_id)
        )
        os.makedirs(self._out_dir, exist_ok=True)

        # The run dir must not already contain results from a previous run.
        # ``--force`` opts in to wiping it and starting from scratch.
        if self._out_dir.exists() and any(self._out_dir.iterdir()):
            if not self._force:
                raise SystemExit(
                    f"Tuner run dir {self._out_dir} already exists and is "
                    f"non-empty. Pass --force to overwrite it."
                )
            shutil.rmtree(self._out_dir)

        # Persist config for reproducibility.
        self._out_dir.mkdir(parents=True, exist_ok=True)
        initial_config_path = self._out_dir / "initial_config.yml"
        dump_yaml(self._initial_config, initial_config_path)

        # Scenario evaluator — shared by all tuning phases.
        self._evaluator = ScenarioEvaluator()

        # SLO objective — drives metric routing and threshold-aware selection.
        slo_objective_config = SloObjectiveConfig.from_config(
            self._initial_config
        )
        self._slo_objective = SloObjective(slo_objective_config)

        # SLO Resolver - shared by all phases for consistent SLO evaluation.
        slo_resolver_config = SloResolverConfig.from_config(
            self._initial_config
        )
        self._slo_resolver = SloResolver(slo_resolver_config)

        # Aggregation metric — shared by all phases.
        self._sampling_config = SamplingConfig.from_config(self._initial_config)
        self._agg_method = self._sampling_config.aggregation_method

        # Timing instrumentation.
        self._timer = PolicyTunerTimer()

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def build_reservoir(self) -> QueryReservoir:
        """
        Phase 1: Build or load the query reservoir.
        """
        save_dir = self._out_dir / "01_reservoir"
        reservoir_config = ReservoirConfig.from_config(self._initial_config)
        reservoir = QueryReservoir(reservoir_config=reservoir_config)
        reservoir.save(save_dir)
        console.print(f"  Saved reservoir to {save_dir}.")

        return reservoir

    def sample_workloads(
        self, reservoir: QueryReservoir
    ) -> tuple[list[Path], list[Path]]:
        """
        Phase 2: Sample train/val workloads from the reservoir.

        Returns the lists of train and val workload paths.
        """

        # Set up forecast policy for sampling.
        forecaster_config = ForecasterConfig.from_config(self._initial_config)
        forecaster = Forecaster(reservoir, forecaster_config)

        train_dir = self._out_dir / "02_workloads" / "train"
        val_dir = self._out_dir / "02_workloads" / "val"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Ground-truth mode: use the real target-day workload as both
        # train and val rather than sampling from the reservoir.
        workload_config = WorkloadConfig.from_config(self._initial_config)
        if forecaster_config.forecast_policy_name == "ground_truth":
            workload = Workload(workload_config=workload_config)
            workload.save(out_dir=train_dir, out_workload_name="t_0")
            workload.save(out_dir=val_dir, out_workload_name="v_0")

            console.print(
                f"  Ground truth mode: copied target workload into "
                f"1 train + 1 val scenario "
                f"under {self._out_dir / '02_workloads'}."
            )
            return [train_dir / "t_0.parquet"], [val_dir / "v_0.parquet"]

        # Sample.
        num_scenarios = self._sampling_config.num_scenarios
        train_fraction = self._sampling_config.train_fraction
        n_train = int(num_scenarios * train_fraction)
        n_val = num_scenarios - n_train
        target_date = workload_config.start_date_inclusive
        rescale_factor = workload_config.rescale_factor
        assert target_date is not None

        _, train_paths = forecaster.forecast_n_scenarios(
            target_date=target_date,
            n_scenarios=n_train,
            initial_seed=self._sampling_config.seed,
            workload_name_prefix="t",
            out_dir=train_dir,
            rescale_factor=rescale_factor,
        )
        _, val_paths = forecaster.forecast_n_scenarios(
            target_date=target_date,
            n_scenarios=n_val,
            initial_seed=self._sampling_config.seed + n_train,
            workload_name_prefix="v",
            out_dir=val_dir,
            rescale_factor=rescale_factor,
        )
        console.print(
            f"  Sampled {n_train} train + {n_val} val workloads "
            f"to {self._out_dir / '02_workloads'}"
        )

        if not train_paths:
            raise ValueError(
                f"train_paths is empty: num_scenarios={num_scenarios}, "
                f"train_fraction={train_fraction} produced n_train=0."
            )
        if not val_paths:
            raise ValueError(
                f"val_paths is empty: num_scenarios={num_scenarios}, "
                f"train_fraction={train_fraction} produced n_val=0."
            )
        return train_paths, val_paths

    def evaluate_baseline(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
    ) -> tuple[AggregatedSimulationResults, AggregatedSimulationResults]:
        """Phase 3: Evaluate the initial config as a baseline.

        Runs all training and validation scenarios with the unmodified
        initial config, aggregates metrics, writes a summary, and
        prints a rich table.

        Returns ``(train_agg, val_agg)``."""
        summary_dir = self._out_dir / "03_baseline"
        train_summary_path = summary_dir / "train_summary.yml"
        val_summary_path = summary_dir / "val_summary.yml"

        # Run train + val workloads in a single parallel batch so the
        # process pool stays fully utilised instead of idling between
        # the two sequential calls.
        n_train = len(train_paths)
        all_paths = train_paths + val_paths
        console.print(
            f"Evaluating baseline on {n_train} train + "
            f"{len(val_paths)} val scenarios..."
        )
        all_results_nested = self._evaluator.evaluate_batch_from_configs(
            phase_name="baseline",
            workload_paths=all_paths,
            configs=[self._initial_config],
            out_dir=summary_dir / "results",
        )
        all_results = all_results_nested[0]
        train_results = all_results[:n_train]
        val_results = all_results[n_train:]

        train_agg = SimulationResult.aggregate(train_results, self._agg_method)
        val_agg = SimulationResult.aggregate(val_results, self._agg_method)

        # Persist summaries.
        summary_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(train_agg, train_summary_path)
        dump_yaml(val_agg, val_summary_path)

        return train_agg, val_agg

    def find_checkpoints(
        self, train_paths: list[Path], val_paths: list[Path]
    ) -> tuple[
        dict[str, Any], AggregatedSimulationResults, AggregatedSimulationResults
    ]:
        """Phase 4: Find promising checkpoints via greedy optimization.

        When ``tuner_config.initial_rpu_candidates`` is provided, runs
        checkpoint optimization independently for each candidate initial
        RPU set, then selects the best (initial_rpus, checkpoint_schedule)
        pair on the validation set using threshold-aware selection.

        When the key is absent, behaves as before (single candidate
        from ``managed_cluster_pool_config.initial_rpus``).
        """
        ckpt_root = self._out_dir / "04_checkpoints"

        # Determine candidates.
        default_rpus = cfgu.getd(
            self._initial_config,
            "managed_cluster_pool_config.initial_rpus",
            [8],
        )
        candidates: list[list[int]] = cfgu.getd(
            self._initial_config,
            "tuner_config.checkpoint_phase.initial_rpu_candidates",
            [default_rpus],
        )

        # Run checkpoint optimization for each candidate.
        candidate_configs: list[dict[str, Any]] = []
        candidate_val_aggs: list[AggregatedSimulationResults] = []
        candidate_train_aggs: list[AggregatedSimulationResults] = []

        for i, rpus in enumerate(candidates):
            tag = f"candidate_{i}"
            console.print(f"  [bold]Candidate {i}[/]: initial_rpus={rpus}")

            # Stamp this candidate's initial_rpus into the config.
            candidate_config = copy_and_apply_overrides(
                self._initial_config,
                {"managed_cluster_pool_config.initial_rpus": rpus},
            )

            # Each candidate gets its own subdirectory.
            candidate_dir = ckpt_root / tag
            train_summary_path = (
                candidate_dir / "checkpoints" / "train_summary.yml"
            )

            optimizer = CheckpointOptimizer(
                evaluator=self._evaluator,
                config=candidate_config,
                run_dir=candidate_dir,
                agg_method=self._agg_method,
            )
            post_ckpt_config, train_agg = optimizer.optimize(
                train_paths=train_paths,
            )
            dump_yaml(train_agg, train_summary_path)

            # Evaluate on validation data.
            val_out = candidate_dir / "final" / "val"
            nested = self._evaluator.evaluate_batch_from_configs(
                phase_name=f"{tag}_ckpt_val",
                workload_paths=val_paths,
                configs=[post_ckpt_config],
                out_dir=val_out,
            )
            val_results = nested[0]
            val_agg = SimulationResult.aggregate(val_results, self._agg_method)

            candidate_configs.append(post_ckpt_config)
            candidate_train_aggs.append(train_agg)
            candidate_val_aggs.append(val_agg)

        # Select the best candidate on validation data.
        val_scores = [
            ViolationCost(
                agg.primary_violation(self._slo_objective.slo_metric), agg.cost
            )
            for agg in candidate_val_aggs
        ]
        best_idx = self._slo_objective.idx_of_best(val_scores)

        best_config = candidate_configs[best_idx]
        best_train_agg = candidate_train_aggs[best_idx]
        best_val_agg = candidate_val_aggs[best_idx]
        best_rpus = candidates[best_idx]

        console.print(
            f"  [green]Selected candidate {best_idx} "
            f"(initial_rpus={best_rpus})[/]"
        )

        # Print comparison across all candidates.
        if len(candidates) > 1:
            comparison_entries = [
                (f"Candidate {i} (rpus={candidates[i]})", agg)
                for i, agg in enumerate(candidate_val_aggs)
            ]
            AggregatedSimulationResults.print_comparison(
                *comparison_entries,
                console=console,
                agg_method=self._agg_method,
                slo_metric=self._slo_objective.slo_metric,
                highlight_best=True,
            )

        # Persist best results.
        best_config_path = ckpt_root / "best_config.yml"
        dump_yaml(best_config, best_config_path)
        dump_yaml(best_train_agg, ckpt_root / "best_train_summary.yml")
        dump_yaml(best_val_agg, ckpt_root / "best_val_summary.yml")

        return best_config, best_train_agg, best_val_agg

    def param_sweep(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        initial_config: dict[str, Any],
        phase_name: str,
        config_key: str | None = None,
    ) -> tuple[
        dict[str, Any], AggregatedSimulationResults, AggregatedSimulationResults
    ]:
        """
        Phases 5 & 6: Autoscaler and routing parameter sweeps.
        """
        if config_key is None:
            config_key = phase_name
        phase_dir = self._out_dir / phase_name
        train_summary_path = phase_dir / "best_train_summary.yml"
        val_summary_path = phase_dir / "best_val_summary.yml"

        sweeper = ParamSweep(
            evaluator=self._evaluator,
            initial_config=initial_config,
            run_dir=self._out_dir,
            phase_name=phase_name,
            slo_objective=self._slo_objective,
            agg_method=self._agg_method,
        )
        post_sweep_config, train_agg, val_agg = sweeper.sweep(
            train_paths=train_paths,
            val_paths=val_paths,
            sweep_config=cfgu.getd(
                self._initial_config, f"tuner_config.{config_key}", {}
            ),
        )
        dump_yaml(train_agg, train_summary_path)
        dump_yaml(val_agg, val_summary_path)
        return post_sweep_config, train_agg, val_agg

    def tune(self) -> Path:
        """Execute the full tuning pipeline end-to-end.

        Returns the path to the final optimised config file.
        """
        final_path = self._out_dir / "final_config.yml"
        try:
            ### Phase 1: Build reservoir
            with self._timer.timed_phase(
                "01_reservoir", "Phase 1: Building reservoir"
            ):
                self._print_banner("Phase 1: Building reservoir")
                reservoir = self.build_reservoir()

            ### Phase 2: Preparing workloads
            with self._timer.timed_phase(
                "02_workloads", "Phase 2: Preparing workloads"
            ):
                self._print_banner("Phase 2: Preparing workloads")
                train_paths, val_paths = self.sample_workloads(reservoir)

            ### Phase 3: Baseline evaluation
            with self._timer.timed_phase(
                "03_baseline", "Phase 3: Baseline evaluation"
            ):
                self._print_banner("Phase 3: Baseline evaluation")
                baseline_train, baseline_val = self.evaluate_baseline(
                    train_paths, val_paths
                )
                AggregatedSimulationResults.print_comparison(
                    ("Baseline (train)", baseline_train),
                    ("Baseline (val)", baseline_val),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                    highlight_best=False,
                )

            ### Phase 4: Checkpoint optimization
            with self._timer.timed_phase(
                "04_checkpoints", "Phase 4: Checkpoint optimization"
            ):
                self._print_banner("Phase 4: Checkpoint optimization")
                (
                    post_checkpoints_config,
                    post_checkpoints_train,
                    post_checkpoints_val,
                ) = self.find_checkpoints(train_paths, val_paths)
                AggregatedSimulationResults.print_comparison(
                    ("Baseline (train)", baseline_train),
                    ("Post-checkpoints (train)", post_checkpoints_train),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                )
                AggregatedSimulationResults.print_comparison(
                    ("Baseline (val)", baseline_val),
                    ("Post-checkpoints (val)", post_checkpoints_val),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                )

            ### Phase 5: Autoscaler parameter sweep
            with self._timer.timed_phase(
                "05_autoscaling_param_sweep", "Phase 5: Autoscaler param sweep"
            ):
                self._print_banner("Phase 5: Autoscaler parameter sweep")
                post_first_sweep_config, _, _ = self.param_sweep(
                    train_paths=train_paths,
                    val_paths=val_paths,
                    initial_config=post_checkpoints_config,
                    phase_name="05_autoscaling_param_sweep",
                    config_key="autoscaling_param_sweep",
                )

            ### Phase 6: Routing parameter sweep
            with self._timer.timed_phase(
                "06_routing_param_sweep", "Phase 6: Routing param sweep"
            ):
                self._print_banner("Phase 6: Routing parameter sweep")
                post_second_sweep_config, tuned_train, tuned_val = (
                    self.param_sweep(
                        train_paths=train_paths,
                        val_paths=val_paths,
                        initial_config=post_first_sweep_config,
                        phase_name="06_routing_param_sweep",
                        config_key="routing_param_sweep",
                    )
                )

            ### Phase 7: Persist final config and final comparison
            with self._timer.timed_phase(
                "07_final", "Phase 7: Persist & compare final"
            ):
                self._print_banner(
                    "Phase 7: Final comparison with tuned config"
                )
                final_config = post_second_sweep_config
                dump_yaml(final_config, final_path)
                console.print(
                    f"  Final config written to [bold]{final_path}[/]"
                )
                summary_dir = self._out_dir / "07_final"
                summary_dir.mkdir(parents=True, exist_ok=True)
                dump_yaml(tuned_train, summary_dir / "train_summary.yml")
                dump_yaml(tuned_val, summary_dir / "val_summary.yml")
                AggregatedSimulationResults.print_comparison(
                    ("Initial (train)", baseline_train),
                    ("Final (train)", tuned_train),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                )
                AggregatedSimulationResults.print_comparison(
                    ("Initial (val)", baseline_val),
                    ("Final (val)", tuned_val),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                )

            ### Phase 8: Evaluation on target-period data
            with self._timed_phase("08_target", "Phase 8: Target evaluation"):
                self._print_banner("Phase 8: Evaluation on target-period data")
                self._evaluate_target(
                    initial_config=self._initial_config,
                    final_config=final_config,
                )

            return final_path
        finally:
            self._timer.finalize(self._out_dir)

    @staticmethod
    def _print_banner(message: str) -> None:
        """Print a rich section banner."""
        console.print()
        console.rule(f"[bold cyan]{message}")
        console.print()

    def _extract_and_save_target_workload(self) -> Path:
        """Extract the real target-day workload from the raw trace and save it.

        Slices the workload defined by ``workload_config`` by the configured
        absolute time range, zero-aligns relative start times, applies the
        rescale factor, and persists it to
        ``<run_dir>/02_workloads/target.parquet``.

        Returns the path to the saved parquet file.
        """
        target_workload_path = self._run_dir / "02_workloads" / "target.parquet"

        schema_name = self._cfgd("basic_config.schema_name", None)
        full_workload_name = self._cfgd("workload_config.workload_name", None)
        if (schema_name is None) or (full_workload_name is None):
            raise ValueError(
                "workload_config.schema_name and workload_config.workload_name "
                "must be specified."
            )
        workload = Workload(full_workload_name, schema_name=schema_name)
        date_str = self._cfgd("workload_config.date")
        rescale_factor = self._cfgd("workload_config.rescale_factor", 1.0)
        workload = workload.prepare(
            abs_start=date_str,
            abs_end=date_str,
            rescale_factor=rescale_factor,
        )
        workload = workload.rename_workload("target")
        workload.save(out_dir=self._run_dir / "02_workloads", overwrite=True)
        console.print(
            f"Extracted target workload from {full_workload_name} for "
            f"date {date_str}, "
            f"rescaled by factor {rescale_factor}, "
            f"and saved to {target_workload_path}."
        )
        return target_workload_path

    def _sample_ground_truth_workloads(self) -> tuple[list[Path], list[Path]]:
        """Phase 2 (ground truth mode): use the real target-day workload.

        Rather than sampling synthetic scenarios, both the train and val sets
        consist of the actual target-day workload.  This lets the tuner
        optimise directly against the ground truth, giving an oracle upper
        bound for holdout performance.
        """
        target_path = self._extract_and_save_target_workload()

        # Copy the single target workload into train/ and val/ directories so
        # downstream phases find paths in the expected layout.
        train_dir = self._run_dir / "02_workloads" / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        dst = train_dir / f"t_0.parquet"
        shutil.copy2(target_path, dst)
        train_paths = [dst]

        val_dir = self._run_dir / "02_workloads" / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        dst = val_dir / f"v_0.parquet"
        shutil.copy2(target_path, dst)
        val_paths = [dst]

        console.print(
            f"  Ground truth mode: copied target workload into "
            f"1 train + 1 val scenario "
            f"under {self._run_dir / '02_workloads'}."
        )
        return train_paths, val_paths

    def _evaluate_target(
        self,
        initial_config: dict[str, Any],
        final_config: dict[str, Any],
    ) -> None:
        """Evaluate baseline, tuned, and static-baseline configs on real data.

        All evaluations (baseline, tuned, and static baselines) are submitted
        to a single process pool via :meth:`evaluate_batch` for maximum
        parallelism.
        """

        ### Extract and save target day.
        target_workload_path = self._extract_and_save_target_workload()

        # Run the initial config and the static baselines.
        static_baselines = self._cfgd("tuner_config.static_baselines", [])
        baseline_overrides = [
            baseline.get("overrides", {}) for baseline in static_baselines
        ]
        baseline_configs = [
            copy_and_apply_overrides(initial_config, bo)
            for bo in baseline_overrides
        ]
        all_results = self._evaluator.evaluate_batch_from_configs(
            phase_name="target",
            out_dir=self._run_dir / "08_target",
            workload_paths=[target_workload_path],
            configs=[initial_config, final_config] + baseline_configs,
        )

        # Extract initial + final results.
        initial_results = all_results[0]
        final_results = all_results[1]
        base_agg = SimulationResult.aggregate(initial_results, self._agg_metric)
        tuned_agg = SimulationResult.aggregate(final_results, self._agg_metric)

        comparison_entries: list[tuple[str, AggregatedSimulationResults]] = [
            ("Initial", base_agg),
            ("Final", tuned_agg),
        ]

        # Extract static baseline summaries.
        static_summaries: list[dict[str, Any]] = []

        for i, sb in enumerate(static_baselines):
            sb_results = all_results[2 + i]
            sb_agg = SimulationResult.aggregate(sb_results, self._agg_metric)
            static_summaries.append(
                {
                    "label": sb["label"],
                    "violation": sb_agg.primary_violation(self._slo_metric),
                    "cost": sb_agg.cost,
                    "violation_rate": sb_agg.violation_rate,
                    "violation_amount_s": sb_agg.violation_amount_s,
                    "violation_relative_mean": sb_agg.violation_relative_mean,
                }
            )
            comparison_entries.append((sb["label"], sb_agg))

        AggregatedSimulationResults.print_comparison(
            *comparison_entries,
            agg_metric=self._agg_metric,
            slo_metric=self._slo_metric,
            console=console,
            highlight_best=True,
        )
        self._print_scatter(comparison_entries, self._slo_objective)

        # --- Write holdout summary --------------------------------------
        summary_dir = self._run_dir / "09_holdout"
        summary_dir.mkdir(parents=True, exist_ok=True)
        holdout_summary: dict[str, Any] = {
            "initial_violation": base_agg.primary_violation(self._slo_metric),
            "initial_cost": base_agg.cost,
            "initial_violation_rate": base_agg.violation_rate,
            "initial_violation_amount_s": base_agg.violation_amount_s,
            "initial_violation_relative_mean": base_agg.violation_relative_mean,
            "final_violation": tuned_agg.primary_violation(self._slo_metric),
            "final_cost": tuned_agg.cost,
            "final_violation_rate": tuned_agg.violation_rate,
            "final_violation_amount_s": tuned_agg.violation_amount_s,
            "final_violation_relative_mean": tuned_agg.violation_relative_mean,
            "slo_metric": self._slo_metric.value,
        }
        if static_summaries:
            holdout_summary["static_baselines"] = static_summaries

        summary_path = summary_dir / "summary.yml"
        dump_yaml(holdout_summary, summary_path)

    _SCATTER_MARKERS = ["●", "■", "▲", "◆", "★", "✦", "◉", "▶"]

    @staticmethod
    def _print_scatter(
        entries: list[tuple[str, AggregatedSimulationResults]],
        slo_objective: SloObjective,
    ) -> None:
        """Print a terminal scatter plot of violation vs cost."""
        x_label = slo_objective.slo_metric.to_plot_axis_label()

        labels: list[str] = []
        xs: list[float] = []
        ys: list[float] = []
        for label, agg in entries:
            labels.append(label)
            xs.append(agg.primary_violation(slo_objective.slo_metric))
            ys.append(agg.cost)

        if len(xs) < 2:
            return

        markers = PolicyTuner._SCATTER_MARKERS

        plt.clear_figure()
        plt.plot_size(60, 20)
        for i in range(len(xs)):
            mk = markers[i % len(markers)]
            plt.scatter([xs[i]], [ys[i]], marker=mk)
        plt.xlabel(x_label)
        plt.ylabel("Cost ($)")

        # Add a vertical line at the SLO threshold and make sure it is included
        # in the plot bounds with some padding.
        threshold = slo_objective.slo_threshold
        plt.vline(
            threshold,
            color="gray",
        )

        x_lo, x_hi = min(min(xs), threshold), max(max(xs), threshold)
        y_lo, y_hi = min(ys), max(ys)
        x_pad = max((x_hi - x_lo) * 0.15, x_hi * 0.05) or 0.01
        y_pad = max((y_hi - y_lo) * 0.15, y_hi * 0.05) or 0.01
        plt.xlim(x_lo - x_pad, x_hi + x_pad)
        plt.ylim(y_lo - y_pad, y_hi + y_pad)

        plt.title("Violation vs. Cost")
        plt.theme("clear")

        # Build the plot as a string and append a legend to the right.
        plot_str = plt.build()
        plot_lines = plot_str.split("\n")

        legend_lines: list[str] = [""]  # blank line at top
        for i, lbl in enumerate(labels):
            mk = markers[i % len(markers)]
            legend_lines.append(f"  {mk} {lbl}")
        legend_lines.append("")

        # Vertically centre the legend against the plot.
        total_plot = len(plot_lines)
        total_legend = len(legend_lines)
        offset = max(0, (total_plot - total_legend) // 2)

        out_lines: list[str] = []
        for row, pline in enumerate(plot_lines):
            li = row - offset
            suffix = legend_lines[li] if 0 <= li < total_legend else ""
            out_lines.append(pline + suffix)

        print("\n".join(out_lines))


if __name__ == "__main__":

    cfg, args = cfgu.load_config_from_cli(
        "Run the policy tuner from a YAML config file.",
    )

    if args.publish_as is not None:
        publication_path = os.path.join(
            pu.get_data_path(), "configs", "tuned", args.publish_as
        )
        if os.path.exists(publication_path) and not args.force:
            console.print(
                f"[red]Error: A tuned config with the name '{args.publish_as}' "
                f"already exists at {publication_path}. Use --force to "
                f"overwrite.[/]"
            )
            exit(1)
    pt = PolicyTuner(cfg, force=args.force)
    final_config_path = pt.tune()
    if args.publish_as is not None:
        publication_path = os.path.join(
            pu.get_data_path(), "configs", "tuned", args.publish_as
        )
        shutil.copy2(final_config_path, publication_path)
        console.print(f"Published tuned config to [bold]{publication_path}[/]")
