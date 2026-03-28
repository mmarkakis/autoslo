"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.checkpoint_optimizer import (
    CheckpointOptimizer,
    _checkpoints_to_config,
)
from autoslo.tuner.config import TunerConfig
from autoslo.tuner.forecast_policy import (
    RecencyWeightedForecastPolicy,
    UniformForecastPolicy,
)
from autoslo.tuner.param_sweep import ParamSweep
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.types import PhaseResult, ScenarioResult, aggregate
from autoslo.tuner.workload_sampler import WorkloadSampler
from autoslo.utils.config import apply_overrides
from autoslo.utils.structured_log import StructuredLogHandler, setup_structured_logging
from autoslo.workload_definition.workload import Workload

logger = logging.getLogger(__name__)
console = Console()


class PolicyTuner:
    """Orchestrates the end-to-end policy tuning pipeline.

    Parameters
    ----------
    initial_config :
        The base simulator configuration dict (as produced by reading
        a ``conn.yml`` / ``blueprints.yml`` style YAML).
    tuner_config :
        Hyper-parameters for the tuning process.
    run_dir :
        Optional explicit root directory for this tuner run.  If *None*,
        a timestamped directory under ``data/tuner_runs/`` is created.
    """

    def __init__(
        self,
        initial_config: dict[str, Any],
        tuner_config: TunerConfig,
        run_dir: Path | None = None,
    ) -> None:
        self._initial_config = initial_config
        self._tuner_config = tuner_config

        # Generate a unique run id.
        ts = int(datetime.now().timestamp() * 1000)
        self._run_id = f"tuner_{ts}"

        # Set up run directory.
        if run_dir is not None:
            self._run_dir = Path(run_dir)
        else:
            self._run_dir = Path("data/tuner_runs") / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Persist configs for reproducibility.
        with open(self._run_dir / "initial_config.yml", "w") as f:
            yaml.dump(initial_config, f, default_flow_style=False)
        with open(self._run_dir / "tuner_config.yml", "w") as f:
            yaml.dump(
                {
                    k: (v.isoformat() if isinstance(v, datetime) else v)
                    for k, v in tuner_config.__dict__.items()
                },
                f,
                default_flow_style=False,
            )

        # Set up structured log for the evolution ledger.
        self._evolution_handler = setup_structured_logging(
            out_dir=str(self._run_dir),
            filename="evolution.parquet",
        )

        # Scenario evaluator — shared by all tuning phases.
        self._evaluator = ScenarioEvaluator(
            initial_config=initial_config,
            tuner_config=tuner_config,
            tuner_run_id=self._run_id,
            evolution_logger=self._evolution_handler,
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
    # Pipeline steps (stubs — implemented in later phases)
    # ------------------------------------------------------------------

    def build_reservoir(self, traces: list[Path]) -> Path:
        """Phase 1: Ingest raw traces and build the query reservoir.

        Parameters
        ----------
        traces :
            Parquet files containing historical workload data.

        Returns
        -------
        Path to the directory containing the saved reservoir.
        """
        schema_name = (
            (self._initial_config.get("basic_config") or {}).get(
                "schema_name", "default"
            )
        )

        workloads = []
        for i, trace_path in enumerate(traces):
            df = pd.read_parquet(trace_path)
            wl = Workload(
                workload_name=f"trace_{i:03d}",
                schema_name=schema_name,
                df=df,
            )

            # Restrict to history window when configured.
            if self._tuner_config.history_start is not None:
                wl.slice_by_abs_time(
                    start=self._tuner_config.history_start.isoformat()
                )
            if self._tuner_config.history_end is not None:
                wl.slice_by_abs_time(
                    end=self._tuner_config.history_end.isoformat()
                )
            if len(wl.df) == 0:
                logger.warning(
                    "Trace %s has 0 rows after history-window slicing",
                    trace_path,
                )
                continue

            workloads.append(wl)

        if self._tuner_config.history_start or self._tuner_config.history_end:
            h_start = self._tuner_config.history_start or "start"
            h_end = self._tuner_config.history_end or "end"
            total = sum(len(w.df) for w in workloads)
            console.print(
                f"  History window: {h_start} → {h_end} "
                f"({total:,} arrivals from {len(workloads)} trace(s))"
            )

        reservoir = QueryReservoir.build(
            workloads, schema_name=schema_name
        )

        if self._tuner_config.classify_arrivals:
            classifications = reservoir.classify_arrivals()
            n_windowed = sum(
                1 for c in classifications.values()
                if c.get("classification") == "windowed"
            )
            console.print(
                f"  Classified {len(classifications)} groups: "
                f"{n_windowed} windowed"
            )

        reservoir_dir = self._run_dir / "reservoir"
        reservoir.save(reservoir_dir)
        console.print(
            f"  Reservoir built from {len(traces)} trace(s), "
            f"{len(reservoir.df)} rows saved to {reservoir_dir}"
        )
        return reservoir_dir

    def sample_workloads(
        self, reservoir_path: Path
    ) -> tuple[list[Path], list[Path]]:
        """Phase 2: Sample train/val workloads from the reservoir.

        Returns
        -------
        ``(train_paths, val_paths)`` — lists of Parquet file paths.
        """
        reservoir = QueryReservoir.load(reservoir_path)
        schema_name = reservoir.meta.get("schema_name", "default")

        policy_name = self._tuner_config.forecast_policy
        if policy_name == "recency_weighted":
            forecast_policy = RecencyWeightedForecastPolicy()
        elif policy_name == "uniform":
            forecast_policy = UniformForecastPolicy()
        else:
            raise ValueError(f"Unknown forecast policy: {policy_name!r}")

        sampler = WorkloadSampler(
            reservoir=reservoir,
            forecast_policy=forecast_policy,
            schema_name=schema_name,
        )

        n_train = self._tuner_config.n_train
        n_val = self._tuner_config.n_val

        train_dir = self._run_dir / "sampled_workloads" / "train"
        val_dir = self._run_dir / "sampled_workloads" / "val"

        train_paths = sampler.sample_to_disk(
            target_start=self._tuner_config.target_start,
            target_end=self._tuner_config.target_end,
            n_scenarios=n_train,
            out_dir=train_dir,
            prefix="t",
            seed=self._tuner_config.random_seed,
        )

        val_paths = sampler.sample_to_disk(
            target_start=self._tuner_config.target_start,
            target_end=self._tuner_config.target_end,
            n_scenarios=n_val,
            out_dir=val_dir,
            prefix="v",
            seed=self._tuner_config.random_seed + n_train,
        )

        console.print(
            f"  Sampled {n_train} train + {n_val} val workloads "
            f"to {self._run_dir / 'sampled_workloads'}"
        )
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
        metric = self._tuner_config.aggregation_metric

        console.rule("[bold cyan]Baseline evaluation")

        # Training set.
        train_results = self._evaluator.evaluate(
            workload_paths=train_paths,
            config_overrides={},
            phase="baseline",
            grid_point="base",
            out_subdir=self._run_dir / "baseline" / "train",
        )
        train_viol, train_cost = aggregate(train_results, metric)

        # Validation set.
        val_results = self._evaluator.evaluate(
            workload_paths=val_paths,
            config_overrides={},
            phase="baseline",
            grid_point="base",
            out_subdir=self._run_dir / "baseline" / "val",
        )
        val_viol, val_cost = aggregate(val_results, metric)

        result = PhaseResult(
            params={},
            train_results=train_results,
            val_results=val_results,
            train_violation_agg=train_viol,
            train_cost_agg=train_cost,
            val_violation_agg=val_viol,
            val_cost_agg=val_cost,
        )

        # Persist summary.
        summary_dir = self._run_dir / "baseline"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self._write_phase_summary(summary_dir / "summary.yml", result)

        # Rich table.
        self._print_phase_summary("Baseline", result)

        return result

    def optimize_checkpoints(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        baseline_val_violation: float,
    ) -> list[CapacityCheckpoint]:
        """Phase 4: Greedy capacity-checkpoint optimisation."""
        optimizer = CheckpointOptimizer(
            evaluator=self._evaluator,
            tuner_config=self._tuner_config,
            initial_config=self._initial_config,
            run_dir=self._run_dir,
        )
        return optimizer.optimize(
            train_paths=train_paths,
            val_paths=val_paths,
            baseline_val_violation=baseline_val_violation,
        )

    def sweep_autoscaler(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        checkpoints: list[CapacityCheckpoint],
    ) -> dict[str, Any]:
        """Phase 5: Grid-search autoscaler hyper-parameters."""
        base_overrides = _checkpoints_to_config(checkpoints)

        sweeper = ParamSweep(
            evaluator=self._evaluator,
            tuner_config=self._tuner_config,
            base_overrides=base_overrides,
            run_dir=self._run_dir,
            phase_name="autoscaler",
        )

        return sweeper.sweep(
            train_paths=train_paths,
            val_paths=val_paths,
            param_ranges=self._tuner_config.autoscaler_ranges,
            config_section="autoscaling_config",
        )

    def sweep_routing(
        self,
        train_paths: list[Path],
        val_paths: list[Path],
        checkpoints: list[CapacityCheckpoint],
        autoscaler_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Phase 6: Grid-search routing hyper-parameters."""
        base_overrides = _checkpoints_to_config(checkpoints)
        for k, v in autoscaler_config.items():
            base_overrides[f"autoscaling_config.{k}"] = v

        sweeper = ParamSweep(
            evaluator=self._evaluator,
            tuner_config=self._tuner_config,
            base_overrides=base_overrides,
            run_dir=self._run_dir,
            phase_name="routing",
        )

        return sweeper.sweep(
            train_paths=train_paths,
            val_paths=val_paths,
            param_ranges=self._tuner_config.routing_ranges,
            config_section="routing_config",
        )

    def tune(self, traces: list[Path]) -> Path:
        """Execute the full tuning pipeline end-to-end.

        Returns the path to the final optimised config file.
        """
        self._print_banner("Phase 1: Building reservoir")
        reservoir_path = self.build_reservoir(traces)

        self._print_banner("Phase 2: Sampling workloads")
        train_paths, val_paths = self.sample_workloads(reservoir_path)

        self._print_banner("Phase 3: Baseline evaluation")
        baseline = self.evaluate_baseline(train_paths, val_paths)

        self._print_banner("Phase 4: Checkpoint optimization")
        checkpoints = self.optimize_checkpoints(
            train_paths, val_paths, baseline.val_violation_agg
        )

        self._print_banner("Phase 5: Autoscaler parameter sweep")
        autoscaler_config = self.sweep_autoscaler(
            train_paths, val_paths, checkpoints
        )

        self._print_banner("Phase 6: Routing parameter sweep")
        routing_config = self.sweep_routing(
            train_paths, val_paths, checkpoints, autoscaler_config
        )

        self._print_banner("Final: Writing optimized config")
        final_path = self._write_final_config(
            checkpoints, autoscaler_config, routing_config
        )

        self._print_banner("Final evaluation with tuned config")
        tuned = self._evaluate_final(
            train_paths, val_paths,
            checkpoints, autoscaler_config, routing_config,
        )
        self._print_comparison(baseline, tuned)

        if self._tuner_config.holdout_evaluation:
            self._print_banner("Holdout: real-data evaluation")
            self._evaluate_holdout(
                traces, checkpoints, autoscaler_config, routing_config,
            )

        return final_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        metric = self._tuner_config.aggregation_metric
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
        train_viol, train_cost = aggregate(train_results, metric)

        val_results = self._evaluator.evaluate(
            workload_paths=val_paths,
            config_overrides=overrides,
            phase="final",
            grid_point="tuned",
            out_subdir=self._run_dir / "final" / "val",
        )
        val_viol, val_cost = aggregate(val_results, metric)

        result = PhaseResult(
            params=overrides,
            train_results=train_results,
            val_results=val_results,
            train_violation_agg=train_viol,
            train_cost_agg=train_cost,
            val_violation_agg=val_viol,
            val_cost_agg=val_cost,
        )
        summary_dir = self._run_dir / "final"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self._write_phase_summary(summary_dir / "summary.yml", result)
        return result

    def _evaluate_holdout(
        self,
        traces: list[Path],
        checkpoints: list[CapacityCheckpoint],
        autoscaler_config: dict[str, Any],
        routing_config: dict[str, Any],
    ) -> None:
        """Evaluate baseline and tuned configs on real held-out data."""
        schema_name = (
            (self._initial_config.get("basic_config") or {}).get(
                "schema_name", "default"
            )
        )
        metric = self._tuner_config.aggregation_metric

        # Extract target-period slice from traces.
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

        if not holdout_paths:
            console.print(
                "[yellow]  No real data in target period; "
                "skipping holdout evaluation.[/yellow]"
            )
            return

        total_queries = sum(
            len(pd.read_parquet(p)) for p in holdout_paths
        )
        console.print(
            f"  Holdout: {total_queries:,} real queries from target period"
        )

        # Evaluate baseline on real data.
        baseline_results = self._evaluator.evaluate(
            workload_paths=holdout_paths,
            config_overrides={},
            phase="holdout",
            grid_point="baseline",
            out_subdir=self._run_dir / "holdout" / "baseline",
        )
        base_viol, base_cost = aggregate(baseline_results, metric)

        # Build tuned overrides.
        overrides = _checkpoints_to_config(checkpoints)
        for k, v in autoscaler_config.items():
            overrides[f"autoscaling_config.{k}"] = v
        for k, v in routing_config.items():
            overrides[f"routing_config.{k}"] = v

        # Evaluate tuned config on real data.
        tuned_results = self._evaluator.evaluate(
            workload_paths=holdout_paths,
            config_overrides=overrides,
            phase="holdout",
            grid_point="tuned",
            out_subdir=self._run_dir / "holdout" / "tuned",
        )
        tuned_viol, tuned_cost = aggregate(tuned_results, metric)

        # Build PhaseResults for comparison and persistence.
        baseline_phase = PhaseResult(
            params={},
            train_results=baseline_results,
            val_results=None,
            train_violation_agg=base_viol,
            train_cost_agg=base_cost,
            val_violation_agg=None,
            val_cost_agg=None,
        )
        tuned_phase = PhaseResult(
            params=overrides,
            train_results=tuned_results,
            val_results=None,
            train_violation_agg=tuned_viol,
            train_cost_agg=tuned_cost,
            val_violation_agg=None,
            val_cost_agg=None,
        )

        summary_dir = self._run_dir / "holdout"
        summary_dir.mkdir(parents=True, exist_ok=True)
        holdout_summary: dict[str, Any] = {
            "num_holdout_queries": total_queries,
            "baseline_violation": base_viol,
            "baseline_cost": base_cost,
            "tuned_violation": tuned_viol,
            "tuned_cost": tuned_cost,
        }
        with open(summary_dir / "summary.yml", "w") as f:
            yaml.dump(holdout_summary, f, default_flow_style=False)

        self._print_comparison(baseline_phase, tuned_phase)

    @staticmethod
    def _print_comparison(baseline: PhaseResult, tuned: PhaseResult) -> None:
        """Print baseline vs tuned metrics side-by-side."""
        table = Table(
            title="Baseline vs. Tuned Performance", show_lines=True
        )
        table.add_column("Metric", justify="left")
        table.add_column("Baseline", justify="right")
        table.add_column("Tuned", justify="right")
        table.add_column("Δ", justify="right")

        rows = [
            (
                "Train Violation",
                baseline.train_violation_agg,
                tuned.train_violation_agg,
            ),
            (
                "Train Cost ($)",
                baseline.train_cost_agg,
                tuned.train_cost_agg,
            ),
        ]
        if baseline.val_violation_agg is not None and tuned.val_violation_agg is not None:
            rows.append((
                "Val Violation",
                baseline.val_violation_agg,
                tuned.val_violation_agg,
            ))
        if baseline.val_cost_agg is not None and tuned.val_cost_agg is not None:
            rows.append((
                "Val Cost ($)",
                baseline.val_cost_agg,
                tuned.val_cost_agg,
            ))

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

        # Write final config.
        final_path = self._run_dir / "final_config.yml"
        with open(final_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

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
            "train_scenarios": [
                {
                    "scenario_idx": r.scenario_idx,
                    "violation_rate": r.violation_rate,
                    "total_cost": r.total_cost,
                }
                for r in result.train_results
            ],
        }
        if result.val_results is not None:
            summary["val_scenarios"] = [
                {
                    "scenario_idx": r.scenario_idx,
                    "violation_rate": r.violation_rate,
                    "total_cost": r.total_cost,
                }
                for r in result.val_results
            ]
        with open(path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

    @staticmethod
    def _print_phase_summary(label: str, result: PhaseResult) -> None:
        """Print a rich table summarising a phase result."""
        table = Table(title=f"{label} Performance", show_lines=True)
        table.add_column("Split", justify="left")
        table.add_column("Violation", justify="right")
        table.add_column("Cost ($)", justify="right")
        table.add_column("# Scenarios", justify="right")

        table.add_row(
            "Train",
            f"{result.train_violation_agg:.4f}",
            f"{result.train_cost_agg:.4f}",
            str(len(result.train_results)),
        )
        if result.val_results is not None:
            table.add_row(
                "Val",
                f"{result.val_violation_agg:.4f}",
                f"{result.val_cost_agg:.4f}",
                str(len(result.val_results)),
            )
        console.print(table)
