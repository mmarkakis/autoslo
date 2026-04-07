"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotext as plt
import yaml
from rich.console import Console

import autoslo.utils.config as cfgu
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.checkpoint_optimizer import CheckpointOptimizer
from autoslo.tuner.forecast_policy import ForecastPolicy
from autoslo.tuner.param_sweep import ParamSweep
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.tuner_utils import (
    AggregatedSimulationResults,
    SimulationResult,
)
from autoslo.utils.config import copy_and_apply_overrides
from autoslo.utils.structured_log import setup_structured_logging
from autoslo.utils.yaml_helpers import dump
from autoslo.workload_definition.workload import Workload

logger = logging.getLogger(__name__)
console = Console()


class PolicyTuner:
    """Orchestrates the end-to-end policy tuning pipeline."""

    def __init__(
        self,
        initial_config: dict,
    ) -> None:
        self._initial_config = initial_config

        # Find or generate a unique run ID and set up output directory.
        ts = int(datetime.now().timestamp() * 1000)
        self._run_id = self._cfgd("basic_config.run_id", f"tuner_{ts}")
        self._run_dir = Path("data/tuner_runs") / self._run_id
        if self._cfgd("basic_config.experiment_name") is not None:
            experiment_name = self._cfgd("basic_config.experiment_name")
            self._run_dir = (
                Path("data/tuner_runs") / experiment_name / self._run_id
            )

        # Check overwrite setting and dump config.
        if self._run_dir.exists() and not self._cfgd(
            "basic_config.overwrite", False
        ):
            raise FileExistsError(
                f"Output directory {self._run_dir} already exists. "
                "Set basic_config.overwrite: true to overwrite."
            )
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Persist config for reproducibility.
        initial_config_path = self._run_dir / "initial_config.yml"
        dump(self._initial_config, initial_config_path)

        # Set up structured log for the evolution ledger.
        self._evolution_handler = setup_structured_logging(
            out_dir=str(self._run_dir),
            filename="evolution.parquet",
        )

        # Scenario evaluator — shared by all tuning phases.
        self._evaluator = ScenarioEvaluator(
            tuner_run_id=self._run_id,
            evolution_logger=self._evolution_handler,
        )

        # SLO objective — drives metric routing and threshold-aware selection.
        self._slo_metric = SloMetric(
            cfgu.getd(self._initial_config, "slo_config.slo_metric", "binary")
        )
        self._slo_threshold = float(
            cfgu.getd(self._initial_config, "slo_config.slo_threshold", 1.0)
        )
        self._slo_objective = SloObjective(
            slo_metric=self._slo_metric,
            slo_threshold=self._slo_threshold,
        )

        # SLO Resolver - shared by all phases for consistent SLO evaluation.
        slo_s = float(self._cfgd("slo_config.slo_s", 30.0))
        slo_dict_filename: str | None = self._cfgd(
            "slo_config.slo_dict_filename", None
        )
        self._slo_resolver = SloResolver(
            default_slo_s=slo_s, slo_dict_filename=slo_dict_filename
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def evaluator(self) -> ScenarioEvaluator:
        return self._evaluator

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def build_reservoir(self) -> None:
        """
        Phase 1: Build or load the query reservoir.
        """

        # Load from disk if it already exists, to save time on re-runs.
        save_dir = self._run_dir / "reservoir"
        if save_dir.exists() and not self._cfgd("tuner_config.force", False):
            console.print(
                f"  Reservoir already exists at {save_dir}; loading from disk."
            )
            self._reservoir = QueryReservoir.load(save_dir)
            return

        # Build reservoir from workload.
        schema_name = self._cfgd("basic_config.schema_name", "def_schema")
        workload_name = self._cfgd(
            "workload_config.workload_name", "def_workload"
        )
        workload = Workload(workload_name, schema_name)
        start = self._cfgd(
            "tuner_config.forecast_config.history_abs_start_time_start"
        )
        end = self._cfgd(
            "tuner_config.forecast_config.history_abs_start_time_end"
        )
        if start or end:
            workload.slice_by_abs_time(start=start, end=end)
        self._reservoir = QueryReservoir(workload=workload)
        self._reservoir.save(save_dir)

        # TODO: Have the reservoir itself generate a nice `rich` summary.
        console.print(
            f"  Built reservoir based on workload {workload_name} over the "
            f"period {start} to {end}, and saved to {save_dir}."
        )

    def sample_workloads(
        self,
    ) -> tuple[list[Path], list[Path]]:
        """
        Phase 2: Sample train/val workloads from the reservoir.

        Returns the lists of train and val workload paths.
        """

        # Set up forecast policy for sampling.
        self._forecast_policy_name = self._cfgd(
            "tuner_config.forecast_config.forecast_policy",
            "OneDayForecastPolicy",
        )
        self._forecast_policy_params = self._cfgd(
            "tuner_config.forecast_config", {}
        )
        self._forecast_policy = ForecastPolicy.from_name(
            name=self._forecast_policy_name,
            reservoir=self._reservoir,
            **self._forecast_policy_params,
        )

        # Sample.
        num_scenarios = self._cfgd(
            "tuner_config.forecast_config.num_scenarios", 20
        )
        train_fraction = self._cfgd(
            "tuner_config.forecast_config.train_fraction", 0.6
        )
        n_train = int(num_scenarios * train_fraction)
        n_val = num_scenarios - n_train

        train_dir = self._run_dir / "workloads" / "train"
        val_dir = self._run_dir / "workloads" / "val"

        target_date = pd.Timestamp(
            self._cfgd(
                "workload_config.abs_start_time_start",
                "2024-01-01T00:00:00",
            )
        ).date()

        train_paths: list[Path]
        val_paths: list[Path]
        if (
            train_dir.exists()
            and val_dir.exists()
            and not self._cfgd("tuner_config.force", False)
        ):
            train_paths = sorted(train_dir.glob("*.parquet"))
            val_paths = sorted(val_dir.glob("*.parquet"))
            console.print(
                f"  Sampled workloads already exist; found {len(train_paths)} "
                f"train and {len(val_paths)} val workloads. Loading from disk."
            )
            return train_paths, val_paths

        initial_seed = self._cfgd(
            "tuner_config.forecast_config.initial_seed", 42
        )

        _, train_paths = self._forecast_policy.forecast_n_scenarios(
            target_date=target_date,
            n_scenarios=n_train,
            initial_seed=initial_seed,
            workload_name_prefix="t",
            out_dir=train_dir,
        )
        _, val_paths = self._forecast_policy.forecast_n_scenarios(
            target_date=target_date,
            n_scenarios=n_val,
            initial_seed=initial_seed + n_train,
            workload_name_prefix="v",
            out_dir=val_dir,
        )
        console.print(
            f"  Sampled {n_train} train + {n_val} val workloads "
            f"to {self._run_dir / 'workloads'}"
        )

        assert train_paths and val_paths, "No workload paths returned."
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
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "p90"
        )

        # Training set.
        console.print(
            f"Evaluating baseline on {len(train_paths)} training scenarios..."
        )
        train_out_dir = self._run_dir / "baseline" / "train"
        train_results: list[SimulationResult]
        if train_out_dir.exists() and not self._cfgd(
            "tuner_config.force", False
        ):
            console.print(
                f"  Baseline train results already exist at {train_out_dir}; "
                "loading from disk."
            )
            train_results = SimulationResult.load_batch(
                train_out_dir / "config_0"
            )
        else:
            train_results_nested = self._evaluator.evaluate_batch_from_configs(
                phase_name="baseline_train",
                workload_paths=train_paths,
                configs=[self._initial_config],
                out_dir=train_out_dir,
            )
            train_results = train_results_nested[0]

        train_agg = SimulationResult.aggregate(train_results, metric)

        # Validation set.
        console.print(
            f"Evaluating baseline on {len(val_paths)} validation scenarios..."
        )
        val_out_dir = self._run_dir / "baseline" / "val"
        val_results: list[SimulationResult]
        if val_out_dir.exists() and not self._cfgd("tuner_config.force", False):
            console.print(
                f"  Baseline val results already exist at {val_out_dir}; "
                "loading from disk."
            )
            val_results = SimulationResult.load_batch(val_out_dir / "config_0")
        else:
            val_results_nested = self._evaluator.evaluate_batch_from_configs(
                phase_name="baseline_val",
                workload_paths=val_paths,
                configs=[self._initial_config],
                out_dir=val_out_dir,
            )
            val_results = val_results_nested[0]
        val_agg = SimulationResult.aggregate(val_results, metric)

        # Persist summary.
        summary_dir = self._run_dir / "baseline"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self._write_phase_summary(summary_dir / "train_summary.yml", train_agg)
        self._write_phase_summary(summary_dir / "val_summary.yml", val_agg)

        return train_agg, val_agg

    def find_checkpoints(
        self, train_paths: list[Path], val_paths: list[Path], agg_metric: str
    ) -> tuple[
        dict[str, Any], AggregatedSimulationResults, AggregatedSimulationResults
    ]:
        """
        Phase 4: Find promising checkpoints via greedy training-only optimization.
        """
        # Compute or retrieve the post-checkpoint-optimization config.
        optimizer = CheckpointOptimizer(
            evaluator=self._evaluator,
            config=self._initial_config,
            run_dir=self._run_dir,
        )
        final_config_path = self._run_dir / "checkpoints" / "final_config.yml"
        if final_config_path.exists() and not self._cfgd(
            "tuner_config.force", False
        ):
            console.print(
                f"  Checkpoint optimization results already exist at "
                f"{final_config_path}; loading from disk."
            )
            with open(final_config_path) as f:
                post_checkpoints_config = yaml.safe_load(f) or {}
        else:
            post_checkpoints_config = optimizer.optimize(
                train_paths=train_paths,
            )

        # Validate checkpoint schedule on training and validation data.
        train_out_dir = self._run_dir / "checkpoints" / "final" / "train"
        train_results: list[SimulationResult]
        if train_out_dir.exists() and not self._cfgd(
            "tuner_config.force", False
        ):
            console.print(
                f"  Checkpoint-optimized train results already exist at "
                f"{train_out_dir}; loading from disk."
            )
            train_results = SimulationResult.load_batch(
                train_out_dir / "config_0"
            )
        else:
            train_results_nested = self._evaluator.evaluate_batch_from_configs(
                phase_name="checkpoints_train",
                workload_paths=train_paths,
                configs=[post_checkpoints_config],
                out_dir=train_out_dir,
            )
            train_results = train_results_nested[0]
        train_agg = SimulationResult.aggregate(train_results, agg_metric)

        val_out_dir = self._run_dir / "checkpoints" / "final" / "val"
        val_results: list[SimulationResult]
        if val_out_dir.exists() and not self._cfgd("tuner_config.force", False):
            console.print(
                f"  Checkpoint-optimized val results already exist at "
                f"{val_out_dir}; loading from disk."
            )
            val_results = SimulationResult.load_batch(val_out_dir / "config_0")
        else:
            val_results_nested = self._evaluator.evaluate_batch_from_configs(
                phase_name="checkpoints_val",
                workload_paths=val_paths,
                configs=[post_checkpoints_config],
                out_dir=val_out_dir,
            )
            val_results = val_results_nested[0]
        val_agg = SimulationResult.aggregate(val_results, agg_metric)

        return post_checkpoints_config, train_agg, val_agg

    def param_sweep(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        initial_config: dict[str, Any],
        phase_name: str,
    ) -> dict[str, Any]:
        """
        Phases 5 & 6: Autoscaler and routing parameter sweeps.
        """
        sweeper = ParamSweep(
            evaluator=self._evaluator,
            initial_config=initial_config,
            run_dir=self._run_dir,
            phase_name=phase_name,
            slo_objective=self._slo_objective,
        )
        final_config_path = self._run_dir / phase_name / "final_config.yml"
        if final_config_path.exists() and not self._cfgd(
            "tuner_config.force", False
        ):
            console.print(
                f"  Parameter sweep results for phase '{phase_name}' already "
                f"exist at {final_config_path}; loading from disk."
            )
            with open(final_config_path) as f:
                post_sweep_config = yaml.safe_load(f) or {}
        else:
            post_sweep_config = sweeper.sweep(
                train_paths=train_paths,
                val_paths=val_paths,
                param_ranges=self._cfgd(f"tuner_config.{phase_name}", {}),
            )
        return post_sweep_config

    def tune(self) -> Path:
        """Execute the full tuning pipeline end-to-end.

        Returns the path to the final optimised config file.
        """

        agg_metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "p90"
        )

        ### Phase 1: Build reservoir
        self._print_banner("Phase 1: Building reservoir")
        self.build_reservoir()

        ### Phase 2: Preparing workloads
        self._print_banner("Phase 2: Preparing workloads")
        train_paths, val_paths = self.sample_workloads()

        ### Phase 3: Baseline evaluation
        self._print_banner("Phase 3: Baseline evaluation")
        baseline_train, baseline_val = self.evaluate_baseline(
            train_paths, val_paths
        )
        AggregatedSimulationResults.print_comparison(
            ("Baseline (train)", baseline_train),
            ("Baseline (val)", baseline_val),
            console=console,
            agg_metric=agg_metric,
            slo_metric=self._slo_metric,
            highlight_best=False,
        )

        ### Phase 4: Checkpoint optimization
        self._print_banner("Phase 4: Checkpoint optimization")
        (
            post_checkpoints_config,
            post_checkpoints_train,
            post_checkpoints_val,
        ) = self.find_checkpoints(train_paths, val_paths, agg_metric=agg_metric)
        AggregatedSimulationResults.print_comparison(
            ("Baseline (train)", baseline_train),
            ("Post-checkpoints (train)", post_checkpoints_train),
            console=console,
            agg_metric=agg_metric,
            slo_metric=self._slo_metric,
        )
        AggregatedSimulationResults.print_comparison(
            ("Baseline (val)", baseline_val),
            ("Post-checkpoints (val)", post_checkpoints_val),
            console=console,
            agg_metric=agg_metric,
            slo_metric=self._slo_metric,
        )

        ### Phase 5: Autoscaler parameter sweep
        self._print_banner("Phase 5: Autoscaler parameter sweep")
        post_first_sweep_config = self.param_sweep(
            train_paths=train_paths,
            val_paths=val_paths,
            initial_config=post_checkpoints_config,
            phase_name="autoscaling_param_sweep",
        )

        ### Phase 6: Routing parameter sweep
        self._print_banner("Phase 6: Routing parameter sweep")
        post_second_sweep_config = self.param_sweep(
            train_paths=train_paths,
            val_paths=val_paths,
            initial_config=post_first_sweep_config,
            phase_name="routing_param_sweep",
        )

        ### Phase 6.5: Persist final config
        final_config = post_second_sweep_config
        final_path = self._run_dir / "final_config.yml"
        dump(final_config, final_path)
        self._evolution_handler.finalize()
        console.print(f"  Final config written to [bold]{final_path}[/]")

        ### Phase 7: Final evaluation with tuned config
        self._print_banner("Phase 7: Final evaluation with tuned config")
        tuned_train, tuned_val = self._evaluate_final(
            train_paths, val_paths, final_config
        )
        AggregatedSimulationResults.print_comparison(
            ("Initial (train)", baseline_train),
            ("Final (train)", tuned_train),
            console=console,
            agg_metric=agg_metric,
            slo_metric=self._slo_metric,
        )
        AggregatedSimulationResults.print_comparison(
            ("Initial (val)", baseline_val),
            ("Final (val)", tuned_val),
            console=console,
            agg_metric=agg_metric,
            slo_metric=self._slo_metric,
        )

        ### Phase 8: Evaluation on target-period data
        self._print_banner("Phase 8: Evaluation on target-period data")
        self._evaluate_target(
            initial_config=self._initial_config, final_config=final_config
        )

        return final_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cfgd(self, dot_delimited_key: str, default: Any = None) -> Any:
        """Helper to get config values from the initial config."""
        return cfgu.getd(self._initial_config, dot_delimited_key, default)

    @staticmethod
    def _print_banner(message: str) -> None:
        """Print a rich section banner."""
        console.print()
        console.rule(f"[bold cyan]{message}")
        console.print()

    def _evaluate_final(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        final_config: dict[str, Any],
    ) -> tuple[AggregatedSimulationResults, AggregatedSimulationResults]:
        """Re-run evaluation with the fully-tuned config.

        Returns ``(train_agg, val_agg)``."""
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "mean"
        )
        summary_dir = self._run_dir / "final"
        summary_dir.mkdir(parents=True, exist_ok=True)
        train_summary_path = summary_dir / "train_summary.yml"
        val_summary_path = summary_dir / "val_summary.yml"
        if (
            train_summary_path.exists()
            and val_summary_path.exists()
            and not self._cfgd("tuner_config.force", False)
        ):
            console.print(
                f"  Final evaluation results already exist at {summary_dir}; "
                "loading from disk."
            )
            return (
                self._parse_phase_summary(train_summary_path),
                self._parse_phase_summary(val_summary_path),
            )

        all_train_results = self._evaluator.evaluate_batch_from_configs(
            phase_name="final",
            out_dir=self._run_dir / "final" / "train",
            workload_paths=train_paths,
            configs=[final_config],
        )
        train_results = all_train_results[0]
        train_agg = SimulationResult.aggregate(train_results, metric)

        all_val_results = self._evaluator.evaluate_batch_from_configs(
            phase_name="final",
            out_dir=self._run_dir / "final" / "val",
            workload_paths=val_paths,
            configs=[final_config],
        )
        val_results = all_val_results[0]
        val_agg = SimulationResult.aggregate(val_results, metric)

        self._write_phase_summary(train_summary_path, train_agg)
        self._write_phase_summary(val_summary_path, val_agg)
        return train_agg, val_agg

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
        schema_name = self._cfgd("basic_config.schema_name", None)
        target_workload_path = self._run_dir / "workloads" / "target.parquet"
        if not target_workload_path.exists() or self._cfgd(
            "tuner_config.force", False
        ):
            full_workload_name = self._cfgd(
                "workload_config.workload_name", None
            )
            if (schema_name is None) or (full_workload_name is None):
                raise ValueError(
                    "workload_config.schema_name and workload_config.workload_name "
                    "must be specified."
                )
            workload = Workload(full_workload_name, schema_name=schema_name)
            start = self._cfgd("workload_config.abs_start_time_start", None)
            end = self._cfgd("workload_config.abs_start_time_end", None)
            workload = workload.slice_by_abs_time(start=start, end=end)
            workload = workload.set_rel_start_times_from_zero()
            rescale_factor = self._cfgd("workload_config.rescale_factor", 1.0)
            workload = workload.rescale_rel_start_times(factor=rescale_factor)
            workload = workload.rename_workload("target")
            workload.save(out_dir=self._run_dir / "workloads", overwrite=True)
            console.print(
                f"Extracted target workload from {full_workload_name} with "
                f"time range {start} to {end}, "
                f"rescaled by factor {rescale_factor}, "
                f"and saved to {target_workload_path}."
            )
        else:
            console.print(
                f"  Target workload already exists at {target_workload_path}; "
                "loading from disk."
            )

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
            out_dir=self._run_dir / "target",
            workload_paths=[target_workload_path],
            configs=[initial_config, final_config] + baseline_configs,
        )

        # Extract initial + final results.
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "mean"
        )
        initial_results = all_results[0]
        final_results = all_results[1]
        base_agg = SimulationResult.aggregate(initial_results, metric)
        tuned_agg = SimulationResult.aggregate(final_results, metric)

        comparison_entries: list[tuple[str, AggregatedSimulationResults]] = [
            ("Initial", base_agg),
            ("Final", tuned_agg),
        ]

        # Extract static baseline summaries.
        static_summaries: list[dict[str, Any]] = []

        for i, sb in enumerate(static_baselines):
            sb_results = all_results[2 + i]
            sb_agg = SimulationResult.aggregate(sb_results, metric)
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
            agg_metric=metric,
            slo_metric=self._slo_metric,
            console=console,
            highlight_best=True,
        )
        self._print_scatter(comparison_entries, self._slo_metric)

        # --- Write holdout summary --------------------------------------
        summary_dir = self._run_dir / "holdout"
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
        dump(holdout_summary, summary_path)

    _SCATTER_MARKERS = ["●", "■", "▲", "◆", "★", "✦", "◉", "▶"]

    @staticmethod
    def _print_scatter(
        entries: list[tuple[str, AggregatedSimulationResults]],
        slo_metric: str | SloMetric,
    ) -> None:
        """Print a terminal scatter plot of violation vs cost."""
        slo_metric_obj = (
            SloMetric(slo_metric) if isinstance(slo_metric, str) else slo_metric
        )
        x_label = slo_metric_obj.aggregate_string_description

        labels: list[str] = []
        xs: list[float] = []
        ys: list[float] = []
        for label, agg in entries:
            labels.append(label)
            xs.append(agg.primary_violation(slo_metric))
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

        x_lo, x_hi = min(xs), max(xs)
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

    @staticmethod
    def _write_phase_summary(
        path: Path, agg: AggregatedSimulationResults
    ) -> None:
        """Persist an AggregatedSimulationResults as YAML."""
        summary: dict[str, Any] = {
            "violation_rate": agg.violation_rate,
            "violation_amount_s": agg.violation_amount_s,
            "violation_relative_mean": agg.violation_relative_mean,
            "cost": agg.cost,
            "scenarios": [
                {
                    "simulation_dir": str(r.simulation_dir),
                    "violation_rate": r.violation_rate,
                    "violation_amount_s": r.violation_amount_s,
                    "violation_relative_mean": r.violation_relative_mean,
                    "total_cost": r.total_cost,
                    "num_queries": r.num_queries,
                }
                for r in agg.scenario_results
            ],
        }
        dump(summary, path)

    @staticmethod
    def _parse_phase_summary(path: Path) -> AggregatedSimulationResults:
        """Parse an AggregatedSimulationResults from YAML."""
        with open(path) as f:
            summary = yaml.safe_load(f) or {}
        scenario_results = tuple(
            SimulationResult(
                simulation_dir=Path(r["simulation_dir"]),
                violation_rate=r["violation_rate"],
                violation_amount_s=r["violation_amount_s"],
                violation_relative_mean=r["violation_relative_mean"],
                total_cost=r["total_cost"],
                num_queries=r["num_queries"],
            )
            for r in summary.get("scenarios", [])
        )
        return AggregatedSimulationResults(
            violation_rate=summary.get("violation_rate", 0.0),
            violation_amount_s=summary.get("violation_amount_s", 0.0),
            violation_relative_mean=summary.get("violation_relative_mean", 0.0),
            cost=summary.get("cost", 0.0),
            scenario_results=scenario_results,
        )


if __name__ == "__main__":
    cfg, config_path = cfgu.load_config_from_cli(
        "Run queries from a workload using a YAML config file.",
    )
    pt = PolicyTuner(cfg)
    pt.tune()
