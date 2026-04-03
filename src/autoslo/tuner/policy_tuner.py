"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import copy
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from autoslo.utils.yaml_helpers import dump_config

from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.checkpoint_optimizer import (
    CheckpointOptimizer,
    _checkpoints_to_config,
)
from autoslo.tuner.config import TunerConfig
from autoslo.tuner.forecast_policy import ForecastPolicy
from autoslo.tuner.param_sweep import ParamSweep
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.scenario_evaluator import EvalSpec, ScenarioEvaluator
from autoslo.tuner.tuner_utils import (
    AggregatedMetrics,
    PhaseResult,
    ScenarioResult,
    SloObjective,
    aggregate,
    primary_violation,
)
import autoslo.utils.config as cfgu

from autoslo.utils.config import apply_overrides
from autoslo.utils.structured_log import (
    StructuredLogHandler,
    setup_structured_logging,
)
from autoslo.workload_definition.workload import Workload

logger = logging.getLogger(__name__)
console = Console()


class PolicyTuner:
    """Orchestrates the end-to-end policy tuning pipeline."""

    def __init__(
        self,
        config: dict,
    ) -> None:
        self._config = config

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
        with open(self._run_dir / "config.yml", "w") as f:
            dump_config(config, f)

        # Set up structured log for the evolution ledger.
        self._evolution_handler = setup_structured_logging(
            out_dir=str(self._run_dir),
            filename="evolution.parquet",
        )

        # Scenario evaluator — shared by all tuning phases.
        self._evaluator = ScenarioEvaluator(
            config=config,
            tuner_run_id=self._run_id,
            evolution_logger=self._evolution_handler,
        )

        # SLO objective — drives metric routing and threshold-aware selection.
        self._slo_metric = str(
            cfgu.cfg_getd(self._config, "slo_config.slo_metric", "binary")
        )
        self._slo_threshold = float(
            cfgu.cfg_getd(self._config, "slo_config.slo_threshold", 1.0)
        )
        self._slo_objective = SloObjective(
            slo_metric=self._slo_metric,
            slo_threshold=self._slo_threshold,
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
        if save_dir.exists():
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

        train_dir = self._run_dir / "sampled_workloads" / "train"
        val_dir = self._run_dir / "sampled_workloads" / "val"

        target_date = pd.Timestamp(
            self._cfgd(
                "workload_config.abs_start_time_start",
                "2024-01-01T00:00:00",
            )
        ).date()

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
            f"to {self._run_dir / 'sampled_workloads'}"
        )

        assert train_paths and val_paths, "No workload paths returned."
        return train_paths, val_paths

    def evaluate_baseline(
        self, train_paths: list[Path], val_paths: list[Path]
    ) -> PhaseResult:
        """Phase 3: Evaluate the initial config as a baseline.

        Runs all training and validation scenarios with the unmodified
        initial config, aggregates metrics, writes a summary, and
        prints a rich table.
        """
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", "p90"
        )

        console.rule("[bold cyan]Baseline evaluation")

        # Training set.
        console.print(
            f"Evaluating baseline on {len(train_paths)} training scenarios..."
        )
        train_results = self._evaluator.evaluate(
            workload_paths=train_paths,
            config_overrides={},
            phase="baseline",
            grid_point="base-train",
            out_subdir=self._run_dir / "baseline" / "train",
        )
        train_agg = aggregate(train_results, metric)

        # Validation set.
        console.print(
            f"Evaluating baseline on {len(val_paths)} validation scenarios..."
        )
        val_results = self._evaluator.evaluate(
            workload_paths=val_paths,
            config_overrides={},
            phase="baseline",
            grid_point="base-val",
            out_subdir=self._run_dir / "baseline" / "val",
        )
        val_agg = aggregate(val_results, metric)

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

        # Rich table.
        self._print_phase_summary("Baseline", result, agg_method=metric)

        return result

    def tune(self) -> Path:
        """Execute the full tuning pipeline end-to-end.

        Returns the path to the final optimised config file.
        """
        final_path = self._run_dir / "final_config.yml"
        overrides_path = self._run_dir / "tuned_overrides.yml"

        # Skip-retuning: if final_config already exists and
        # force_retuning is not set, jump straight to holdout.
        # if not self._tuner_config.force_retuning and final_path.exists():
        #     console.print(
        #         "[bold yellow]Skipping tuning phases 1-7 "
        #         "(final_config.yml already exists). "
        #         "Set force_retuning: true to re-run.[/bold yellow]"
        #     )
        #     if overrides_path.exists():
        #         with open(overrides_path) as f:
        #             overrides: dict[str, Any] = yaml.safe_load(f) or {}
        #     else:
        #         # Legacy run: reconstruct overrides by diffing configs.
        #         overrides = self._rebuild_overrides(final_path)
        #         with open(overrides_path, "w") as f:
        #             dump_config(overrides, f)
        #         console.print(
        #             "  [dim]Reconstructed tuned_overrides.yml from "
        #             "config diff.[/dim]"
        #         )

        #     if self._tuner_config.holdout_evaluation:
        #         self._print_banner("Holdout: real-data evaluation")
        #         self._evaluate_holdout(traces, overrides)

        #     return final_path

        ### Phase 1: Build reservoir
        self._print_banner("Phase 1: Building reservoir")
        self.build_reservoir()

        ### Phase 2: Sampling workloads
        self._print_banner("Phase 2: Sampling workloads")
        train_paths, val_paths = self.sample_workloads()

        ### Phase 3: Baseline evaluation
        self._print_banner("Phase 3: Baseline evaluation")
        baseline = self.evaluate_baseline(train_paths, val_paths)

        ### Phase 4: Checkpoint optimization
        self._print_banner("Phase 4: Checkpoint optimization")
        optimizer = CheckpointOptimizer(
            evaluator=self._evaluator,
            config=self._config,
            run_dir=self._run_dir,
        )
        assert (
            baseline.val_violation_agg is not None
        ), "Baseline violation agg is None"
        checkpoints = optimizer.optimize(
            train_paths=train_paths,
            val_paths=val_paths,
            baseline_val_violation=baseline.val_violation_agg,
        )
        base_overrides = _checkpoints_to_config(checkpoints)

        return

        ### Phase 5: Autoscaler parameter
        self._print_banner("Phase 5: Autoscaler parameter sweep")
        sweeper = ParamSweep(
            evaluator=self._evaluator,
            tuner_config=self._tuner_config,
            base_overrides=base_overrides,
            run_dir=self._run_dir,
            phase_name="autoscaler",
            slo_objective=self._slo_objective,
        )
        autoscaler_config = sweeper.sweep(
            train_paths=train_paths,
            val_paths=val_paths,
            param_ranges=self._tuner_config.autoscaler_ranges,
            config_section="autoscaling_config",
        )

        ### Phase 6: Routing parameter sweep
        self._print_banner("Phase 6: Routing parameter sweep")
        for k, v in autoscaler_config.items():
            base_overrides[f"autoscaling_config.{k}"] = v

        sweeper = ParamSweep(
            evaluator=self._evaluator,
            tuner_config=self._tuner_config,
            base_overrides=base_overrides,
            run_dir=self._run_dir,
            phase_name="routing",
            slo_objective=self._slo_objective,
        )

        routing_config = sweeper.sweep(
            train_paths=train_paths,
            val_paths=val_paths,
            param_ranges=self._tuner_config.routing_ranges,
            config_section="routing_config",
        )

        self._print_banner("Final: Writing optimized config")
        final_path = self._write_final_config(
            checkpoints, autoscaler_config, routing_config
        )

        self._print_banner("Final evaluation with tuned config")
        tuned = self._evaluate_final(
            train_paths,
            val_paths,
            checkpoints,
            autoscaler_config,
            routing_config,
        )
        self._print_comparison(baseline, tuned)

        # Build overrides for holdout (same structure as tuned_overrides.yml).
        overrides = _checkpoints_to_config(checkpoints)
        for k, v in autoscaler_config.items():
            overrides[f"autoscaling_config.{k}"] = v
        for k, v in routing_config.items():
            overrides[f"routing_config.{k}"] = v

        if self._cfgd("tuner_config.holdout_evaluation", False):
            self._print_banner("Holdout: real-data evaluation")
            self._evaluate_holdout(traces, overrides)

        return final_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cfgd(self, dot_delimited_key: str, default: Any = None) -> Any:
        """Helper to get config values from the initial config."""
        return cfgu.cfg_getd(self._config, dot_delimited_key, default)

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
        checkpoints: list[CapacityCheckpoint],
        autoscaler_config: dict[str, Any],
        routing_config: dict[str, Any],
    ) -> PhaseResult:
        """Re-run evaluation with the fully-tuned config."""
        metric = self._cfgd(
            "tuner_config.forecast_config.aggregation_metric", metric
        )
        overrides = _checkpoints_to_config(checkpoints)
        for k, v in autoscaler_config.items():
            overrides[f"autoscaling_config.{k}"] = v
        for k, v in routing_config.items():
            overrides[f"routing_config.{k}"] = v

        train_results = self._evaluator.evaluate(
            workload_paths=train_paths,
            config_overrides=overrides,
            phase="final",
            grid_point="tuned",
            out_subdir=self._run_dir / "final" / "train",
        )
        train_agg = aggregate(train_results, metric)

        val_results = self._evaluator.evaluate(
            workload_paths=val_paths,
            config_overrides=overrides,
            phase="final",
            grid_point="tuned",
            out_subdir=self._run_dir / "final" / "val",
        )
        val_agg = aggregate(val_results, metric)

        result = PhaseResult(
            params=overrides,
            train_results=train_results,
            val_results=val_results,
            train_violation_agg=primary_violation(train_agg, self._slo_metric),
            train_cost_agg=train_agg.cost,
            val_violation_agg=primary_violation(val_agg, self._slo_metric),
            val_cost_agg=val_agg.cost,
            train_metrics=train_agg,
            val_metrics=val_agg,
        )
        summary_dir = self._run_dir / "final"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self._write_phase_summary(summary_dir / "summary.yml", result)
        return result

    def _prepare_holdout_workloads(
        self,
        traces: list[Path],
    ) -> list[Path]:
        """Slice real target-period data from traces and write to disk.

        Returns the list of holdout parquet paths, or an empty list if
        no queries fall in the target period.
        """
        schema_name = (self._initial_config.get("basic_config") or {}).get(
            "schema_name", "default"
        )
        holdout_dir = self._run_dir / "holdout" / "workloads"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        holdout_paths: list[Path] = []

        for i, trace_path in enumerate(traces):
            df = pd.read_parquet(trace_path)
            wl = Workload(f"holdout_{i:03d}", schema_name, df=df)
            wl.slice_by_abs_time(
                start=self._tuner_config.target_start.isoformat(),
                end=self._tuner_config.target_end.isoformat(),
            )
            if len(wl.df) == 0:
                continue
            out_path = holdout_dir / f"holdout_{i:03d}.parquet"
            wl.df.to_parquet(out_path)
            holdout_paths.append(out_path)

        return holdout_paths

    @staticmethod
    def _slugify(label: str) -> str:
        """Convert a label to a filesystem-safe slug."""
        slug = label.lower().replace(" ", "_")
        slug = re.sub(r"[^a-z0-9_]", "", slug)
        return f"static_{slug}"

    def _rebuild_overrides(self, final_path: Path) -> dict[str, Any]:
        """Reconstruct tuned overrides by diffing initial and final configs."""
        with open(final_path) as f:
            final_cfg = yaml.safe_load(f) or {}

        initial_flat = self._flatten(self._initial_config)
        final_flat = self._flatten(final_cfg)

        overrides: dict[str, Any] = {}
        for key in sorted(set(initial_flat) | set(final_flat)):
            iv = initial_flat.get(key)
            fv = final_flat.get(key)
            if iv != fv:
                overrides[key] = fv
        return overrides

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Recursively flatten a nested dict to dot-path keys."""
        items: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(PolicyTuner._flatten(v, key))
            else:
                items[key] = v
        return items

    def _evaluate_holdout(
        self,
        traces: list[Path],
        tuned_overrides: dict[str, Any],
    ) -> None:
        """Evaluate baseline, tuned, and static-baseline configs on real data.

        All evaluations (baseline, tuned, and static baselines) are submitted
        to a single process pool via :meth:`evaluate_batch` for maximum
        parallelism.
        """
        metric = self._tuner_config.aggregation_metric

        # Extract target-period slice from traces.
        holdout_paths = self._prepare_holdout_workloads(traces)

        if not holdout_paths:
            console.print(
                "[yellow]  No real data in target period; "
                "skipping holdout evaluation.[/yellow]"
            )
            return

        total_queries = sum(len(pd.read_parquet(p)) for p in holdout_paths)
        console.print(
            f"  Holdout: {total_queries:,} real queries from target period"
        )

        # Build evaluation specs: baseline + tuned + static baselines.
        specs: list[EvalSpec] = [
            EvalSpec(
                label="baseline",
                config_overrides={},
                grid_point="baseline",
                out_subdir=self._run_dir / "holdout" / "baseline",
            ),
            EvalSpec(
                label="tuned",
                config_overrides=tuned_overrides,
                grid_point="tuned",
                out_subdir=self._run_dir / "holdout" / "tuned",
            ),
        ]
        static_baselines = self._tuner_config.static_baselines or []
        for sb in static_baselines:
            label = sb["label"]
            slug = self._slugify(label)
            specs.append(
                EvalSpec(
                    label=label,
                    config_overrides=sb.get("overrides", {}),
                    grid_point=slug,
                    out_subdir=self._run_dir / "holdout" / slug,
                )
            )

        # Run all evaluations in a single pool.
        all_results = self._evaluator.evaluate_batch(
            workload_paths=holdout_paths,
            specs=specs,
            phase="holdout",
        )

        # Extract baseline + tuned results.
        baseline_results, tuned_results = all_results[0], all_results[1]
        base_agg = aggregate(baseline_results, metric)
        tuned_agg = aggregate(tuned_results, metric)

        baseline_phase = PhaseResult(
            params={},
            train_results=baseline_results,
            val_results=None,
            train_violation_agg=primary_violation(base_agg, self._slo_metric),
            train_cost_agg=base_agg.cost,
            val_violation_agg=None,
            val_cost_agg=None,
            train_metrics=base_agg,
            val_metrics=None,
        )
        tuned_phase = PhaseResult(
            params=tuned_overrides,
            train_results=tuned_results,
            val_results=None,
            train_violation_agg=primary_violation(tuned_agg, self._slo_metric),
            train_cost_agg=tuned_agg.cost,
            val_violation_agg=None,
            val_cost_agg=None,
            train_metrics=tuned_agg,
            val_metrics=None,
        )
        self._print_comparison(baseline_phase, tuned_phase)

        # Extract static baseline summaries.
        static_summaries: list[dict[str, Any]] = []
        for i, sb in enumerate(static_baselines):
            sb_results = all_results[2 + i]
            sb_agg = aggregate(sb_results, metric)
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

        if static_summaries:
            self._print_static_summary(static_summaries)

        # --- Write holdout summary --------------------------------------
        summary_dir = self._run_dir / "holdout"
        summary_dir.mkdir(parents=True, exist_ok=True)
        holdout_summary: dict[str, Any] = {
            "num_holdout_queries": total_queries,
            "baseline_violation": primary_violation(base_agg, self._slo_metric),
            "baseline_cost": base_agg.cost,
            "baseline_violation_rate": base_agg.violation_rate,
            "baseline_violation_amount_s": base_agg.violation_amount_s,
            "baseline_violation_relative_mean": base_agg.violation_relative_mean,
            "tuned_violation": primary_violation(tuned_agg, self._slo_metric),
            "tuned_cost": tuned_agg.cost,
            "tuned_violation_rate": tuned_agg.violation_rate,
            "tuned_violation_amount_s": tuned_agg.violation_amount_s,
            "tuned_violation_relative_mean": tuned_agg.violation_relative_mean,
            "slo_metric": self._slo_metric,
        }
        if static_summaries:
            holdout_summary["static_baselines"] = static_summaries
        with open(summary_dir / "summary.yml", "w") as f:
            dump_config(holdout_summary, f)

    @staticmethod
    def _print_comparison(baseline: PhaseResult, tuned: PhaseResult) -> None:
        """Print baseline vs tuned metrics side-by-side."""
        table = Table(title="Baseline vs. Tuned Performance", show_lines=True)
        table.add_column("Metric", justify="left")
        table.add_column("Baseline", justify="right")
        table.add_column("Tuned", justify="right")
        table.add_column("Δ", justify="right")

        rows: list[tuple[str, float, float]] = []

        # All 3 violation metrics from train_metrics if available.
        bm = baseline.train_metrics
        tm = tuned.train_metrics
        if bm is not None and tm is not None:
            rows.append(
                ("Train Viol. Rate", bm.violation_rate, tm.violation_rate)
            )
            rows.append(
                (
                    "Train Viol. Amount (s)",
                    bm.violation_amount_s,
                    tm.violation_amount_s,
                )
            )
            rows.append(
                (
                    "Train Viol. Relative",
                    bm.violation_relative_mean,
                    tm.violation_relative_mean,
                )
            )
            rows.append(("Train Cost ($)", bm.cost, tm.cost))
        else:
            rows.append(
                (
                    "Train Violation",
                    baseline.train_violation_agg,
                    tuned.train_violation_agg,
                )
            )
            rows.append(
                (
                    "Train Cost ($)",
                    baseline.train_cost_agg,
                    tuned.train_cost_agg,
                )
            )

        bvm = baseline.val_metrics
        tvm = tuned.val_metrics
        if bvm is not None and tvm is not None:
            rows.append(
                ("Val Viol. Rate", bvm.violation_rate, tvm.violation_rate)
            )
            rows.append(
                (
                    "Val Viol. Amount (s)",
                    bvm.violation_amount_s,
                    tvm.violation_amount_s,
                )
            )
            rows.append(
                (
                    "Val Viol. Relative",
                    bvm.violation_relative_mean,
                    tvm.violation_relative_mean,
                )
            )
            rows.append(("Val Cost ($)", bvm.cost, tvm.cost))
        elif (
            baseline.val_violation_agg is not None
            and tuned.val_violation_agg is not None
        ):
            rows.append(
                (
                    "Val Violation",
                    baseline.val_violation_agg,
                    tuned.val_violation_agg,
                )
            )
            if (
                baseline.val_cost_agg is not None
                and tuned.val_cost_agg is not None
            ):
                rows.append(
                    ("Val Cost ($)", baseline.val_cost_agg, tuned.val_cost_agg)
                )

        for label, base_val, tuned_val in rows:
            delta = tuned_val - base_val
            sign = "+" if delta >= 0 else ""
            style = "green" if delta <= 0 else "red"
            table.add_row(
                label,
                f"{base_val:.4f}",
                f"{tuned_val:.4f}",
                f"[{style}]{sign}{delta:.4f}[/{style}]",
            )
        console.print(table)

    @staticmethod
    def _print_static_summary(
        static_summaries: list[dict[str, Any]],
    ) -> None:
        """Print a Rich table summarising static baseline holdout results."""
        table = Table(title="Static Baselines — Holdout", show_lines=True)
        table.add_column("Label", justify="left")
        table.add_column("Viol. Rate", justify="right")
        table.add_column("Viol. Amount (s)", justify="right")
        table.add_column("Viol. Relative", justify="right")
        table.add_column("Cost ($)", justify="right")
        for entry in static_summaries:
            table.add_row(
                entry["label"],
                f"{entry.get('violation_rate', entry.get('violation', 0.0)):.4f}",
                f"{entry.get('violation_amount_s', 0.0):.4f}",
                f"{entry.get('violation_relative_mean', 0.0):.4f}",
                f"{entry['cost']:.2f}",
            )
        console.print(table)

    def _write_final_config(
        self,
        checkpoints: list[CapacityCheckpoint],
        autoscaler_config: dict[str, Any],
        routing_config: dict[str, Any],
    ) -> Path:
        """Deep-copy initial config, overlay tuned params, and persist."""
        cfg = copy.deepcopy(self._initial_config)

        # Apply checkpoint overrides.
        overrides = _checkpoints_to_config(checkpoints)
        # Apply autoscaler params.
        for k, v in autoscaler_config.items():
            overrides[f"autoscaling_config.{k}"] = v
        # Apply routing params.
        for k, v in routing_config.items():
            overrides[f"routing_config.{k}"] = v

        apply_overrides(cfg, overrides)

        # Record the rescale factor so that downstream tooling knows
        # the workload was time-compressed.
        if self._tuner_config.rescale_factor is not None:
            cfg.setdefault("workload_config", {})[
                "rescale_factor"
            ] = self._tuner_config.rescale_factor

        # Write final config.
        final_path = self._run_dir / "final_config.yml"
        with open(final_path, "w") as f:
            dump_config(cfg, f)

        # Persist raw overrides so skip-retuning runs can reload them.
        overrides_path = self._run_dir / "tuned_overrides.yml"
        with open(overrides_path, "w") as f:
            dump_config(overrides, f)

        # Finalize evolution log.
        self._evolution_handler.finalize()

        console.print(f"  Final config written to [bold]{final_path}[/]")

        return final_path

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
                "scenario_idx": r.scenario_idx,
                "violation_rate": r.violation_rate,
                "violation_amount_s": r.violation_amount_s,
                "violation_relative_mean": r.violation_relative_mean,
                "total_cost": r.total_cost,
            }
            for r in result.train_results
        ]
        if result.val_results is not None:
            summary["val_scenarios"] = [
                {
                    "scenario_idx": r.scenario_idx,
                    "violation_rate": r.violation_rate,
                    "violation_amount_s": r.violation_amount_s,
                    "violation_relative_mean": r.violation_relative_mean,
                    "total_cost": r.total_cost,
                }
                for r in result.val_results
            ]
        with open(path, "w") as f:
            dump_config(summary, f)

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

    @staticmethod
    def _print_phase_summary(
        label: str, result: PhaseResult, agg_method: str = "p90"
    ) -> None:
        """Print a rich table summarising a phase result."""
        table = Table(title=f"{label} Performance", show_lines=True)
        table.add_column("Split", justify="left")
        table.add_column("Viol. Rate", justify="right")
        table.add_column("Viol. Amount (s)", justify="right")
        table.add_column("Viol. Relative", justify="right")
        table.add_column("Cost ($)", justify="right")
        table.add_column("# Scenarios", justify="right")
        table.add_column("Agg.", justify="center")

        fmt = PolicyTuner._fmt_cell

        tm = result.train_metrics
        tr = result.train_results
        if tm is not None:
            table.add_row(
                "Train",
                fmt(tm.violation_rate, [r.violation_rate for r in tr]),
                fmt(tm.violation_amount_s, [r.violation_amount_s for r in tr]),
                fmt(
                    tm.violation_relative_mean,
                    [r.violation_relative_mean for r in tr],
                ),
                fmt(tm.cost, [r.total_cost for r in tr]),
                str(len(tr)),
                agg_method,
            )
        else:
            table.add_row(
                "Train",
                f"{result.train_violation_agg:.4f}",
                "—",
                "—",
                f"{result.train_cost_agg:.4f}",
                str(len(tr)),
                agg_method,
            )

        vm = result.val_metrics
        vr = result.val_results or []
        if vm is not None:
            table.add_row(
                "Val",
                fmt(vm.violation_rate, [r.violation_rate for r in vr]),
                fmt(vm.violation_amount_s, [r.violation_amount_s for r in vr]),
                fmt(
                    vm.violation_relative_mean,
                    [r.violation_relative_mean for r in vr],
                ),
                fmt(vm.cost, [r.total_cost for r in vr]),
                str(len(vr)),
                agg_method,
            )
        elif result.val_results is not None:
            table.add_row(
                "Val",
                f"{result.val_violation_agg:.4f}",
                "—",
                "—",
                f"{result.val_cost_agg:.4f}",
                str(len(vr)),
                agg_method,
            )
        console.print(table)


if __name__ == "__main__":
    cfg, config_path = cfgu.load_config_from_cli(
        "Run queries from a workload using a YAML config file.",
    )
    pt = PolicyTuner(cfg)
    pt.tune()
