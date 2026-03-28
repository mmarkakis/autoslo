"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.checkpoint_optimizer import (
    CheckpointOptimizer,
    _checkpoints_to_config,
)
from autoslo.tuner.config import TunerConfig
from autoslo.tuner.param_sweep import ParamSweep
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.types import PhaseResult, ScenarioResult, aggregate
from autoslo.utils.structured_log import StructuredLogHandler, setup_structured_logging

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
        """Phase 1: Ingest raw traces and build the query reservoir."""
        raise NotImplementedError("build_reservoir")

    def sample_workloads(
        self, reservoir_path: Path
    ) -> tuple[list, list]:
        """Phase 2: Sample train/val workloads from the reservoir."""
        raise NotImplementedError("sample_workloads")

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

        Returns the path to the run directory.
        """
        raise NotImplementedError("tune")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
