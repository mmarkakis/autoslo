"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import (
    ParamSweepConfig,
    SamplingConfig,
    SloObjectiveConfig,
    SloResolverConfig,
    SpinupOptimizerConfig,
    WorkloadConfig,
)
from autoslo.config.utils import copy_and_apply_overrides, make_run_id
from autoslo.filesystem.path_utils import is_up_to_date
from autoslo.filesystem.yaml_helpers import dump_yaml, load_yaml_with_params
from autoslo.forecasting.forecaster import Forecaster
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.param_sweep import ParamSweep
from autoslo.tuner.policy_tuner_timer import PolicyTunerTimer
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.spinup_optimizer import SpinupOptimizer
from autoslo.workload_definition.workload import Workload
from autoslo.workload_execution.aggregated_execution_results import (
    AggregatedExecutionResults,
)

logger = logging.getLogger(__name__)
console = Console()


class AlreadyCompleteError(Exception):
    """Raised when a tuning run's outputs are already up to date.

    Caught by callers (the script's ``__main__`` block and
    ``run_tuning.py``) to emit a skip message and exit 0.
    """


class PolicyTuner:
    """Orchestrates the end-to-end policy tuning pipeline."""

    def __init__(
        self,
        initial_execution_config_path: str,
        tuner_config_path: str,
        force: bool = False,
        params: dict[str, str] | None = None,
        run_id: str | None = None,
    ) -> None:

        if params is None:
            params = {}

        self._initial_execution_config = load_yaml_with_params(
            initial_execution_config_path, params
        )
        self._tuner_config = load_yaml_with_params(tuner_config_path, params)

        # Construct a run_id from the execution config and tuner config stems
        # plus any injected params.
        self._run_id = run_id
        if self._run_id is None:
            self._run_id = make_run_id(
                [
                    Path(initial_execution_config_path).stem,
                    Path(tuner_config_path).stem,
                ],
                params,
            )

        self._out_dir = Path(
            os.path.join(pu.get_data_path(), "tuner_runs", self._run_id)
        )
        self._publication_path = os.path.join(
            pu.get_data_path(),
            "execution_configs",
            "tuned",
            self._run_id + ".yml",
        )

        # Without --force, skip if the published tuned config already exists
        # and is not older than both input configs (i.e. it is up to date).
        if not force and is_up_to_date(
            Path(self._publication_path),
            Path(initial_execution_config_path),
            Path(tuner_config_path),
        ):
            raise AlreadyCompleteError(
                f"Tuned config '{self._publication_path}' is up to date; "
                f"skipping. Pass --force to re-run regardless."
            )

        # Wipe any existing outputs before re-running (stale or --force).
        if self._out_dir.exists():
            shutil.rmtree(self._out_dir)
        if os.path.exists(self._publication_path):
            os.remove(self._publication_path)
        os.makedirs(self._out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._publication_path), exist_ok=True)

        # Persist substituted configs for reproducibility.
        initial_execution_config_out_path = (
            self._out_dir / "initial_execution_config.yml"
        )
        dump_yaml(
            self._initial_execution_config, initial_execution_config_out_path
        )
        tuner_config_out_path = self._out_dir / "tuner_config.yml"
        dump_yaml(self._tuner_config, tuner_config_out_path)

        # Scenario evaluator — shared by all tuning phases.
        self._evaluator = ScenarioEvaluator()

        # SLO objective — drives metric routing and threshold-aware selection.
        slo_objective_config = SloObjectiveConfig.from_config(
            self._initial_execution_config
        )
        self._slo_objective = SloObjective(slo_objective_config)

        # SLO Resolver - shared by all phases for consistent SLO evaluation.
        slo_resolver_config = SloResolverConfig.from_config(
            self._initial_execution_config
        )
        self._slo_resolver = SloResolver(slo_resolver_config)

        # Aggregation metric — shared by all phases.
        self._sampling_config = SamplingConfig.from_config(self._tuner_config)
        self._agg_method = self._sampling_config.aggregation_method

        # Timing instrumentation.
        self._timer = PolicyTunerTimer()

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def sample_workloads(
        self,
    ) -> tuple[list[WorkloadConfig], list[WorkloadConfig]]:
        """
        Phase 2: Sample train/val workloads from the reservoir.

        Returns the lists of train and val workload configurations.
        """
        train_dir = self._out_dir / "02_workloads" / "train"
        val_dir = self._out_dir / "02_workloads" / "val"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Ground-truth mode: use the real target-day workload as both
        # train and val rather than sampling from the reservoir.
        workload_config = WorkloadConfig.from_config(
            self._initial_execution_config
        )
        if self._sampling_config.forecaster_config is None:
            workload = Workload(workload_config=workload_config)
            workload.save(out_dir=train_dir, out_workload_name="t_0")
            workload.save(out_dir=val_dir, out_workload_name="v_0")

            train_workload_config = WorkloadConfig(
                workload_name="t_0",
                workload_dir=train_dir,
                target_date=workload_config.target_date,
                rescale_factor=workload_config.rescale_factor,
            )
            val_workload_config = WorkloadConfig(
                workload_name="v_0",
                workload_dir=val_dir,
                target_date=workload_config.target_date,
                rescale_factor=workload_config.rescale_factor,
            )

            console.print(
                f"  Ground truth mode: copied target workload into "
                f"1 train + 1 val scenario "
                f"under {self._out_dir / '02_workloads'}."
            )
            return [train_workload_config], [val_workload_config]

        # Sample.
        forecaster = Forecaster(self._sampling_config.forecaster_config)
        num_scenarios = self._sampling_config.num_scenarios
        train_fraction = self._sampling_config.train_fraction
        n_train = int(num_scenarios * train_fraction)
        n_val = num_scenarios - n_train
        target_date = workload_config.target_date
        rescale_factor = workload_config.rescale_factor
        assert target_date is not None

        train_workload_configs = forecaster.forecast_n_scenarios(
            target_date=target_date,
            n_scenarios=n_train,
            initial_seed=self._sampling_config.seed,
            workload_name_prefix="t",
            out_dir=train_dir,
            rescale_factor=rescale_factor,
        )
        val_workload_configs = forecaster.forecast_n_scenarios(
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

        if not train_workload_configs:
            raise ValueError(
                f"train_workload_configs is empty: num_scenarios={num_scenarios}, "
                f"train_fraction={train_fraction} produced n_train=0."
            )
        if not val_workload_configs:
            raise ValueError(
                f"val_workload_configs is empty: num_scenarios={num_scenarios}, "
                f"train_fraction={train_fraction} produced n_val=0."
            )
        return train_workload_configs, val_workload_configs

    def evaluate_baseline(
        self,
        train_workload_configs: list[WorkloadConfig],
        val_workload_configs: list[WorkloadConfig],
    ) -> tuple[AggregatedExecutionResults, AggregatedExecutionResults]:
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
        n_train = len(train_workload_configs)
        all_workload_configs = train_workload_configs + val_workload_configs
        console.print(
            f"Evaluating baseline on {n_train} train + "
            f"{len(val_workload_configs)} val scenarios..."
        )
        all_results_nested = self._evaluator.evaluate_batch_from_configs(
            progress_bar_label="baseline",
            workload_configs=all_workload_configs,
            configs=[self._initial_execution_config],
            out_dir=summary_dir / "results",
            workload_first=False,
        )
        all_results = all_results_nested[0]
        train_results = all_results[:n_train]
        val_results = all_results[n_train:]

        train_agg = AggregatedExecutionResults.aggregate_from(
            train_results, self._agg_method
        )
        val_agg = AggregatedExecutionResults.aggregate_from(
            val_results, self._agg_method
        )

        # Persist summaries.
        summary_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(train_agg, train_summary_path)
        dump_yaml(val_agg, val_summary_path)

        return train_agg, val_agg

    def find_spinups(
        self,
        train_workload_configs: list[WorkloadConfig],
        val_workload_configs: list[WorkloadConfig],
    ) -> tuple[
        dict[str, Any], AggregatedExecutionResults, AggregatedExecutionResults
    ]:
        """Phase 4: Find promising spin-ups via greedy optimization.

        When ``tuner_config.initial_rpu_candidates`` is provided, runs
        spin-up optimization independently for each candidate initial
        RPU set, then selects the best (initial_rpus, spinup_schedule)
        pair on the validation set using threshold-aware selection.

        When the key is absent, behaves as before (single candidate
        from ``managed_cluster_pool_config.initial_rpus``).
        """
        spinup_root = self._out_dir / "04_spinups"

        # Determine candidates.
        spinup_optimizer_config = SpinupOptimizerConfig.from_config(
            self._tuner_config
        )
        candidates = spinup_optimizer_config.initial_rpu_candidates

        # Run spin-up optimization for each candidate.
        candidate_configs: list[dict[str, Any]] = []
        candidate_val_aggs: list[AggregatedExecutionResults] = []
        candidate_train_aggs: list[AggregatedExecutionResults] = []

        for i, rpus in enumerate(candidates):
            tag = f"candidate_{i}"
            console.rule(f"[bold yellow]Candidate {i}: initial_rpus={rpus}[/]")

            # Stamp this candidate's initial_rpus into the config.
            candidate_config = copy_and_apply_overrides(
                self._initial_execution_config,
                {"managed_cluster_pool_config.initial_rpus": rpus},
            )

            # Each candidate gets its own subdirectory.
            candidate_dir = spinup_root / tag
            train_summary_path = candidate_dir / "spinups" / "train_summary.yml"

            optimizer = SpinupOptimizer(
                evaluator=self._evaluator,
                config=candidate_config,
                spinup_optimizer_config=spinup_optimizer_config,
                run_dir=candidate_dir,
                agg_method=self._agg_method,
            )
            post_spinups_config, train_agg = optimizer.optimize(
                train_workload_configs=train_workload_configs,
            )
            dump_yaml(train_agg, train_summary_path)

            # Evaluate on validation data.
            val_out = candidate_dir / "final" / "val"
            nested = self._evaluator.evaluate_batch_from_configs(
                progress_bar_label=f"{tag}_spinup_val",
                workload_configs=val_workload_configs,
                configs=[post_spinups_config],
                out_dir=val_out,
                workload_first=False,
            )
            val_results = nested[0]
            val_agg = AggregatedExecutionResults.aggregate_from(
                val_results, self._agg_method
            )

            candidate_configs.append(post_spinups_config)
            candidate_train_aggs.append(train_agg)
            candidate_val_aggs.append(val_agg)

        # Print candidate comparison on training data.
        console.rule(f"[bold yellow]Candidate Comparison on Training Data[/]")
        comparison_entries = [
            (f"Candidate {i} (initial_rpus={candidates[i]})", agg)
            for i, agg in enumerate(candidate_train_aggs)
        ]
        AggregatedExecutionResults.print_comparison(
            *comparison_entries,
            console=console,
            agg_method=self._agg_method,
            slo_metric=self._slo_objective.slo_metric,
            highlight_best=True,
        )

        # Print candidate comparison on validation data.
        console.rule(f"[bold yellow]Candidate Comparison on Validation Data[/]")
        comparison_entries = [
            (f"Candidate {i} (initial_rpus={candidates[i]})", agg)
            for i, agg in enumerate(candidate_val_aggs)
        ]
        AggregatedExecutionResults.print_comparison(
            *comparison_entries,
            console=console,
            agg_method=self._agg_method,
            slo_metric=self._slo_objective.slo_metric,
            highlight_best=True,
        )

        # Select the best candidate on validation data and print.
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

        # Persist best results.
        best_config_path = spinup_root / "best_config.yml"
        dump_yaml(best_config, best_config_path)
        dump_yaml(best_train_agg, spinup_root / "best_train_summary.yml")
        dump_yaml(best_val_agg, spinup_root / "best_val_summary.yml")

        return best_config, best_train_agg, best_val_agg

    def param_sweep(
        self,
        train_workload_configs: list[WorkloadConfig],
        val_workload_configs: list[WorkloadConfig],
        initial_config: dict[str, Any],
        phase_name: str,
        config_key: str,
    ) -> tuple[
        dict[str, Any], AggregatedExecutionResults, AggregatedExecutionResults
    ]:
        """
        Phases 5 & 6: Autoscaler and routing parameter sweeps.
        """
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
            train_workload_configs=train_workload_configs,
            val_workload_configs=val_workload_configs,
            param_sweep_config=ParamSweepConfig.from_config(
                self._tuner_config[config_key]
            ),
        )
        dump_yaml(train_agg, train_summary_path)
        dump_yaml(val_agg, val_summary_path)
        return post_sweep_config, train_agg, val_agg

    def tune(self) -> None:
        """Execute the full tuning pipeline end-to-end.

        Returns the path to the final optimised config file.
        """
        final_path = self._out_dir / "final_execution_config.yml"
        try:
            ### Phase 2: Preparing workloads
            with self._timer.timed_phase(
                "02_workloads", "Phase 2: Preparing workloads"
            ):
                self._print_banner("Phase 2: Preparing workloads")
                train_workload_configs, val_workload_configs = (
                    self.sample_workloads()
                )

            ### Phase 3: Baseline evaluation
            with self._timer.timed_phase(
                "03_baseline", "Phase 3: Baseline evaluation"
            ):
                self._print_banner("Phase 3: Baseline evaluation")
                baseline_train, baseline_val = self.evaluate_baseline(
                    train_workload_configs, val_workload_configs
                )
                AggregatedExecutionResults.print_comparison(
                    ("Baseline (train)", baseline_train),
                    ("Baseline (val)", baseline_val),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                    highlight_best=False,
                )

            ### Phase 4: Spin-up optimization
            with self._timer.timed_phase(
                "04_spinups", "Phase 4: Spin-up optimization"
            ):
                self._print_banner("Phase 4: Spin-up optimization")
                (
                    post_spinups_config,
                    post_spinups_train,
                    post_spinups_val,
                ) = self.find_spinups(
                    train_workload_configs, val_workload_configs
                )
                AggregatedExecutionResults.print_comparison(
                    ("Baseline (train)", baseline_train),
                    ("Post-spinups (train)", post_spinups_train),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                )
                AggregatedExecutionResults.print_comparison(
                    ("Baseline (val)", baseline_val),
                    ("Post-spinups (val)", post_spinups_val),
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
                    train_workload_configs=train_workload_configs,
                    val_workload_configs=val_workload_configs,
                    initial_config=post_spinups_config,
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
                        train_workload_configs=train_workload_configs,
                        val_workload_configs=val_workload_configs,
                        initial_config=post_first_sweep_config,
                        phase_name="06_routing_param_sweep",
                        config_key="query_routing_param_sweep",
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
                AggregatedExecutionResults.print_comparison(
                    ("Initial (train)", baseline_train),
                    ("Final (train)", tuned_train),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                )
                AggregatedExecutionResults.print_comparison(
                    ("Initial (val)", baseline_val),
                    ("Final (val)", tuned_val),
                    console=console,
                    agg_method=self._agg_method,
                    slo_metric=self._slo_objective.slo_metric,
                )

            shutil.copy2(final_path, self._publication_path)
            console.print(
                f"Published tuned config to [bold]{self._publication_path}[/]"
            )
        finally:
            self._timer.finalize(self._out_dir)

    @staticmethod
    def _print_banner(message: str) -> None:
        """Print a rich section banner."""
        console.print()
        console.rule(f"[bold cyan]{message}")
        console.print()
