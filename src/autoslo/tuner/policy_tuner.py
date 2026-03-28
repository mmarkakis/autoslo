"""PolicyTuner — orchestrator for automated policy tuning."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from autoslo.tuner.config import TunerConfig
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.tuner.types import PhaseResult
from autoslo.utils.structured_log import StructuredLogHandler, setup_structured_logging

logger = logging.getLogger(__name__)


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
        self, train: list, val: list
    ) -> PhaseResult:
        """Phase 3: Evaluate the initial config as a baseline."""
        raise NotImplementedError("evaluate_baseline")

    def optimize_checkpoints(
        self, train: list, val: list
    ) -> list:
        """Phase 4: Find optimal capacity checkpoints."""
        raise NotImplementedError("optimize_checkpoints")

    def sweep_autoscaler(
        self, train: list, val: list, checkpoints: list
    ) -> dict:
        """Phase 5: Grid-search autoscaler hyper-parameters."""
        raise NotImplementedError("sweep_autoscaler")

    def sweep_routing(
        self, train: list, val: list, checkpoints: list, autoscaler_config: dict
    ) -> dict:
        """Phase 6: Grid-search routing hyper-parameters."""
        raise NotImplementedError("sweep_routing")

    def tune(self, traces: list[Path]) -> Path:
        """Execute the full tuning pipeline end-to-end.

        Returns the path to the run directory.
        """
        raise NotImplementedError("tune")
