"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import copy
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotext as plt
import yaml
from rich.console import Console
from rich.table import Table

from autoslo.utils.yaml_helpers import dump_config

from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.checkpoint_optimizer import CheckpointOptimizer
from autoslo.tuner.forecast_policy import ForecastPolicy
from autoslo.tuner.param_sweep import ParamSweep
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.tuner_utils import (
    AggregatedSimulationResults,
    PhaseResult,
    SimulationResult,
    SloObjective,
    primary_violation,
)
import autoslo.utils.config as cfgu

from autoslo.utils.config import copy_and_apply_overrides
from autoslo.utils.structured_log import (
    StructuredLogHandler,
    setup_structured_logging,
)
from autoslo.workload_definition.workload import Workload
from autoslo.blueprint_selection.slo_resolver import SloResolver

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
        with open(self._run_dir / "initial_config.yml", "w") as f:
            dump_config(self._initial_config, f)

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
        self._slo_metric = str(
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
    ) -> PhaseResult:
        """Phase 3: Evaluate the initial config as a baseline.

        Runs all training and validation scenarios with the unmodified
        initial config, aggregates metrics, writes a summary, and
        prints a rich table.
        """
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
            train_results_nested_dict = (
                self._evaluator.evaluate_batch_from_configs(
                    phase_name="baseline_train",
                    workload_paths=train_paths,
                    configs=[self._initial_config],
                    out_dir=train_out_dir,
                )
            )
            train_results = list(train_results_nested_dict[0].values())

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
            val_results_nested_dict = (
                self._evaluator.evaluate_batch_from_configs(
                    phase_name="baseline_val",
                    workload_paths=val_paths,
                    configs=[self._initial_config],
                    out_dir=val_out_dir,
                )
            )
            val_results = list(val_results_nested_dict[0].values())
        val_agg = SimulationResult.aggregate(val_results, metric)

        result = PhaseResult(
            params={},
            train_results=train_results,
            val_results=val_results,
            train_violation_agg=primary_violation(train_agg, self._slo_metric),
            train_cost_agg=train_agg.cost,
            val_violation_agg=primary_violation(val_agg, self._slo_metric),
            val_cost_agg=val_agg.cost,
            train_metrics=train_agg,
            val_metrics=val_agg,
        )

        # Persist summary.
        summary_dir = self._run_dir / "baseline"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self._write_phase_summary(summary_dir / "summary.yml", result)

        return result

    def find_checkpoints(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        baseline_val_violation: Optional[float],
    ) -> dict[str, Any]:
        """
        Phase 4: Find promising checkpoints via optimization.
        """
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
            assert (
                baseline_val_violation is not None
            ), "Baseline violation agg is None"
            post_checkpoints_config = optimizer.optimize(
                train_paths=train_paths,
                val_paths=val_paths,
                baseline_val_violation=baseline_val_violation,
            )
        return post_checkpoints_config

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

        ### Phase 1: Build reservoir
        self._print_banner("Phase 1: Building reservoir")
        self.build_reservoir()

        ### Phase 2: Preparing workloads
        self._print_banner("Phase 2: Preparing workloads")
        train_paths, val_paths = self.sample_workloads()

        ### Phase 3: Baseline evaluation
        self._print_banner("Phase 3: Baseline evaluation")
        baseline = self.evaluate_baseline(train_paths, val_paths)

        ### Phase 4: Checkpoint optimization
        self._print_banner("Phase 4: Checkpoint optimization")
        post_checkpoints_config = self.find_checkpoints(
            train_paths, val_paths, baseline.val_violation_agg
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
        with open(final_path, "w") as f:
            dump_config(final_config, f)
        self._evolution_handler.finalize()
        console.print(f"  Final config written to [bold]{final_path}[/]")

        ### Phase 7: Final evaluation with tuned config
        self._print_banner("Phase 7: Final evaluation with tuned config")
        tuned = self._evaluate_final(train_paths, val_paths, final_config)
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "p90"
        )
        self._print_comparison(
            ("Initial", baseline),
            ("Final", tuned),
            agg_method=metric,
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
    ) -> PhaseResult:
        """Re-run evaluation with the fully-tuned config."""
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "mean"
        )
        summary_dir = self._run_dir / "final"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_yml_path = summary_dir / "summary.yml"
        if summary_yml_path.exists() and not self._cfgd(
            "tuner_config.force", False
        ):
            console.print(
                f"  Final evaluation results already exist at {summary_yml_path}; "
                "loading from disk."
            )
            return self._parse_phase_summary(summary_yml_path)

        all_train_results = self._evaluator.evaluate_batch_from_configs(
            phase_name="final",
            out_dir=self._run_dir / "final" / "train",
            workload_paths=train_paths,
            configs=[final_config],
        )
        train_results = list(all_train_results[0].values())
        train_agg = SimulationResult.aggregate(train_results, metric)

        all_val_results = self._evaluator.evaluate(
            workload_paths=val_paths,
            configs=[final_config],
        )
        val_results = list(all_val_results[0].values())
        val_agg = SimulationResult.aggregate(val_results, metric)

        result = PhaseResult(
            params=final_config,
            train_results=train_results,
            val_results=val_results,
            train_violation_agg=primary_violation(train_agg, self._slo_metric),
            train_cost_agg=train_agg.cost,
            val_violation_agg=primary_violation(val_agg, self._slo_metric),
            val_cost_agg=val_agg.cost,
            train_metrics=train_agg,
            val_metrics=val_agg,
        )
        self._write_phase_summary(summary_dir / "summary.yml", result)
        return result

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
        initial_results = list(all_results[0].values())
        final_results = list(all_results[1].values())
        base_agg = SimulationResult.aggregate(initial_results, metric)
        tuned_agg = SimulationResult.aggregate(final_results, metric)

        initial_phase = PhaseResult(
            params={},
            train_results=initial_results,
            val_results=None,
            train_violation_agg=primary_violation(base_agg, self._slo_metric),
            train_cost_agg=base_agg.cost,
            val_violation_agg=None,
            val_cost_agg=None,
            train_metrics=base_agg,
            val_metrics=None,
        )
        final_phase = PhaseResult(
            params={}
            train_results=final_results,
            val_results=None,
            train_violation_agg=primary_violation(tuned_agg, self._slo_metric),
            train_cost_agg=tuned_agg.cost,
            val_violation_agg=None,
            val_cost_agg=None,
            train_metrics=tuned_agg,
            val_metrics=None,
        )
        comparison_entries: list[tuple[str, PhaseResult]] = [
            ("Initial", initial_phase),
            ("Final", final_phase),
        ]

        # Extract static baseline summaries.
        static_summaries: list[dict[str, Any]] = []

        for i, sb in enumerate(static_baselines):
            sb_results = list(all_results[2 + i].values())
            sb_agg = SimulationResult.aggregate(sb_results, metric)
            static_summaries.append(
                {
                    "label": sb["label"],
                    "violation": primary_violation(sb_agg, self._slo_metric),
                    "cost": sb_agg.cost,
                    "violation_rate": sb_agg.violation_rate,
                    "violation_amount_s": sb_agg.violation_amount_s,
                    "violation_relative_mean": sb_agg.violation_relative_mean,
                }
            )
            comparison_entries.append(
                (
                    sb["label"],
                    PhaseResult(
                        params=sb.get("overrides", {}),
                        train_results=sb_results,
                        val_results=None,
                        train_violation_agg=primary_violation(
                            sb_agg, self._slo_metric
                        ),
                        train_cost_agg=sb_agg.cost,
                        val_violation_agg=None,
                        val_cost_agg=None,
                        train_metrics=sb_agg,
                        val_metrics=None,
                    ),
                )
            )

        self._print_comparison(*comparison_entries, agg_method=metric)
        self._print_scatter(comparison_entries, self._slo_metric)

        # --- Write holdout summary --------------------------------------
        summary_dir = self._run_dir / "holdout"
        summary_dir.mkdir(parents=True, exist_ok=True)
        holdout_summary: dict[str, Any] = {
            "initial_violation": primary_violation(base_agg, self._slo_metric),
            "initial_cost": base_agg.cost,
            "initial_violation_rate": base_agg.violation_rate,
            "initial_violation_amount_s": base_agg.violation_amount_s,
            "initial_violation_relative_mean": base_agg.violation_relative_mean,
            "final_violation": primary_violation(tuned_agg, self._slo_metric),
            "final_cost": tuned_agg.cost,
            "final_violation_rate": tuned_agg.violation_rate,
            "final_violation_amount_s": tuned_agg.violation_amount_s,
            "final_violation_relative_mean": tuned_agg.violation_relative_mean,
            "slo_metric": self._slo_metric,
        }
        if static_summaries:
            holdout_summary["static_baselines"] = static_summaries
        with open(summary_dir / "summary.yml", "w") as f:
            dump_config(holdout_summary, f)

    @staticmethod
    def _print_comparison(
        *entries: tuple[str, PhaseResult],
        agg_method: str = "p90",
    ) -> None:
        """Print a table comparing multiple PhaseResults side-by-side.

        Parameters
        ----------
        *entries :
            ``(label, phase_result)`` pairs.  Each gets one row.
        agg_method :
            Aggregation method shown in the title.
        """
        # Decide which splits to show based on the entries.
        has_val = any(pr.val_metrics is not None for _, pr in entries)

        table = Table(
            title=f"Comparison  [dim](agg: {agg_method})[/dim]",
            show_lines=True,
        )
        table.add_column("Config", justify="left")

        metric_labels = [
            "Viol. Rate",
            "Viol. Amt (s)",
            "Viol. Rel.",
            "Cost ($)",
        ]
        for ml in metric_labels:
            header = f"Train {ml}" if has_val else ml
            table.add_column(header, justify="right")
        if has_val:
            for ml in metric_labels:
                table.add_column(f"Val {ml}", justify="right")

        fmt = PolicyTuner._fmt_cell

        def _extract(
            m: AggregatedSimulationResults | None,
            rs: list[SimulationResult] | None,
        ) -> tuple[list[float], list[list[float]]]:
            if m is not None and rs:
                return (
                    [
                        m.violation_rate,
                        m.violation_amount_s,
                        m.violation_relative_mean,
                        m.cost,
                    ],
                    [
                        [r.violation_rate for r in rs],
                        [r.violation_amount_s for r in rs],
                        [r.violation_relative_mean for r in rs],
                        [r.total_cost for r in rs],
                    ],
                )
            return [0.0, 0.0, 0.0, 0.0], [[], [], [], []]

        # Build row data: each entry → (label, train_aggs, train_per, val_aggs, val_per)
        row_data: list[
            tuple[
                str,
                list[float],
                list[list[float]],
                list[float],
                list[list[float]],
            ]
        ] = []
        for label, pr in entries:
            t_agg, t_per = _extract(pr.train_metrics, pr.train_results)
            v_agg, v_per = _extract(pr.val_metrics, pr.val_results)
            row_data.append((label, t_agg, t_per, v_agg, v_per))

        # Find the best (lowest) aggregated value per column.
        n_metric = 4
        n_cols = n_metric * (2 if has_val else 1)
        best_per_col: list[int] = []
        if row_data:
            for c in range(n_cols):
                if c < n_metric:
                    best_per_col.append(
                        min(
                            range(len(row_data)),
                            key=lambda i, _c=c: row_data[i][1][_c],
                        )
                    )
                else:
                    best_per_col.append(
                        min(
                            range(len(row_data)),
                            key=lambda i, _c=c - n_metric: row_data[i][3][_c],
                        )
                    )

        for row_idx, (label, t_agg, t_per, v_agg, v_per) in enumerate(row_data):
            cells: list[str] = []
            for c in range(n_metric):
                cell = fmt(t_agg[c], t_per[c])
                if len(row_data) > 1 and row_idx == best_per_col[c]:
                    cell = f"[green]{cell}[/green]"
                cells.append(cell)
            if has_val:
                for c in range(n_metric):
                    cell = fmt(v_agg[c], v_per[c])
                    if (
                        len(row_data) > 1
                        and row_idx == best_per_col[n_metric + c]
                    ):
                        cell = f"[green]{cell}[/green]"
                    cells.append(cell)
            table.add_row(label, *cells)

        console.print(table)

    _SCATTER_MARKERS = ["●", "■", "▲", "◆", "★", "✦", "◉", "▶"]

    @staticmethod
    def _print_scatter(
        entries: list[tuple[str, PhaseResult]],
        slo_metric: str,
    ) -> None:
        """Print a terminal scatter plot of violation vs cost."""
        _METRIC_LABELS = {
            "binary": "Violation Rate",
            "absolute_s": "Violation Amount (s)",
            "relative": "Violation Relative Mean",
        }
        x_label = _METRIC_LABELS.get(slo_metric, slo_metric)

        labels: list[str] = []
        xs: list[float] = []
        ys: list[float] = []
        for label, pr in entries:
            m = pr.train_metrics
            if m is None:
                continue
            labels.append(label)
            xs.append(primary_violation(m, slo_metric))
            ys.append(m.cost)

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
    def _write_phase_summary(path: Path, result: PhaseResult) -> None:
        """Persist a PhaseResult as YAML."""
        summary: dict[str, Any] = {
            "params": result.params,
            "train_violation_agg": result.train_violation_agg,
            "train_cost_agg": result.train_cost_agg,
            "val_violation_agg": result.val_violation_agg,
            "val_cost_agg": result.val_cost_agg,
        }
        if result.train_metrics is not None:
            tm = result.train_metrics
            summary["train_violation_rate"] = tm.violation_rate
            summary["train_violation_amount_s"] = tm.violation_amount_s
            summary["train_violation_relative_mean"] = (
                tm.violation_relative_mean
            )
        if result.val_metrics is not None:
            vm = result.val_metrics
            summary["val_violation_rate"] = vm.violation_rate
            summary["val_violation_amount_s"] = vm.violation_amount_s
            summary["val_violation_relative_mean"] = vm.violation_relative_mean
        summary["train_scenarios"] = [
            {
                "simulation_dir": str(r.simulation_dir),
                "violation_rate": r.violation_rate,
                "violation_amount_s": r.violation_amount_s,
                "violation_relative_mean": r.violation_relative_mean,
                "total_cost": r.total_cost,
                "num_queries": r.num_queries,
            }
            for r in result.train_results
        ]
        if result.val_results is not None:
            summary["val_scenarios"] = [
                {
                    "simulation_dir": str(r.simulation_dir),
                    "violation_rate": r.violation_rate,
                    "violation_amount_s": r.violation_amount_s,
                    "violation_relative_mean": r.violation_relative_mean,
                    "total_cost": r.total_cost,
                    "num_queries": r.num_queries,
                }
                for r in result.val_results
            ]
        with open(path, "w") as f:
            dump_config(summary, f)

    @staticmethod
    def _parse_phase_summary(path: Path) -> PhaseResult:
        """Parse a PhaseResult from YAML."""
        with open(path) as f:
            summary = yaml.safe_load(f) or {}
        train_results = [
            SimulationResult(
                simulation_dir=Path(r["simulation_dir"]),
                violation_rate=r["violation_rate"],
                violation_amount_s=r["violation_amount_s"],
                violation_relative_mean=r["violation_relative_mean"],
                total_cost=r["total_cost"],
                num_queries=r["num_queries"],
            )
            for r in summary.get("train_scenarios", [])
        ]
        val_results = None
        if "val_scenarios" in summary:
            val_results = [
                SimulationResult(
                    simulation_dir=Path(r["simulation_dir"]),
                    violation_rate=r["violation_rate"],
                    violation_amount_s=r["violation_amount_s"],
                    violation_relative_mean=r["violation_relative_mean"],
                    total_cost=r["total_cost"],
                    num_queries=r["num_queries"],
                )
                for r in summary.get("val_scenarios", [])
            ]
        return PhaseResult(
            params=summary.get("params", {}),
            train_results=train_results,
            val_results=val_results,
            train_violation_agg=summary.get("train_violation_agg"),
            train_cost_agg=summary.get("train_cost_agg"),
            val_violation_agg=summary.get("val_violation_agg"),
            val_cost_agg=summary.get("val_cost_agg"),
            train_metrics=AggregatedSimulationResults(
                violation_rate=summary.get("train_violation_rate", 0.0),
                violation_amount_s=summary.get("train_violation_amount_s", 0.0),
                violation_relative_mean=summary.get(
                    "train_violation_relative_mean", 0.0
                ),
                cost=summary.get("train_cost_agg", 0.0),
            ),
            val_metrics=(
                AggregatedSimulationResults(
                    violation_rate=summary.get("val_violation_rate", 0.0),
                    violation_amount_s=summary.get(
                        "val_violation_amount_s", 0.0
                    ),
                    violation_relative_mean=summary.get(
                        "val_violation_relative_mean", 0.0
                    ),
                    cost=summary.get("val_cost_agg", 0.0),
                )
                if "val_violation_rate" in summary
                else None
            ),
        )

    @staticmethod
    def _fmt_cell(
        agg_val: float,
        scenario_vals: list[float],
    ) -> str:
        """Format a metric cell: aggregated value with dim min–max range."""
        main = f"{agg_val:.4f}"
        if len(scenario_vals) >= 2:
            lo, hi = min(scenario_vals), max(scenario_vals)
            main += f"\n[dim]{lo:.4f} … {hi:.4f}[/dim]"
        return main


if __name__ == "__main__":
    cfg, config_path = cfgu.load_config_from_cli(
        "Run queries from a workload using a YAML config file.",
    )
    pt = PolicyTuner(cfg)
    pt.tune()
