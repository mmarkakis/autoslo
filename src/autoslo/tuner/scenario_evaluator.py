"""ScenarioEvaluator — runs N simulations in parallel and collects results."""

from __future__ import annotations

import copy
import logging
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import Manager, get_context
from pathlib import Path
from typing import Any

import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from autoslo.tuner.config import TunerConfig
from autoslo.tuner.types import ScenarioResult, extract_scenario_result
from autoslo.utils.paralellism import (
    _init_worker,
    deg_of_paralellism,
    inner_level_num_cpus,
)
from autoslo.utils.structured_log import StructuredLogHandler, emit_structured

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level worker (must be picklable — no closures / lambdas)
# ---------------------------------------------------------------------------


def _run_scenario(
    config_dict: dict[str, Any],
    workload_path: str,
    workload_name: str,
    schema_name: str,
    scenario_idx: int,
    slo_s: float,
    slo_dict: dict[str, float] | None,
    progress_dict: dict[int, tuple[int, int]] | None = None,
    rescale_factor: float | None = None,
) -> ScenarioResult:
    """Execute a single simulation inside a worker process.

    This function is the target for :class:`ProcessPoolExecutor`.  It is
    intentionally defined at module level so that it is picklable.

    The workload DataFrame is read from *workload_path* (a parquet file
    persisted under ``sampled_workloads/`` in the tuner run directory).
    """
    # Suppress worker-process console output so it does not collide with
    # the parent's rich progress bars.  Structured logs in the simulator
    # still capture useful data to disk (the structured logger has
    # propagate=False and its own level, so it is unaffected by the root
    # logger level change below).
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    logging.getLogger().setLevel(logging.CRITICAL)

    # Restrict internal parallelism (PyTorch, BLAS, etc.) so that
    # multiple workers can coexist without over-subscribing cores.
    # With spawn context the _init_worker initializer already set the
    # env vars before any heavy imports; these are kept as defence-in-depth.
    ncpus = str(inner_level_num_cpus())
    os.environ["OMP_NUM_THREADS"] = ncpus
    os.environ["MKL_NUM_THREADS"] = ncpus
    os.environ["OPENBLAS_NUM_THREADS"] = ncpus

    from autoslo.workload_definition.workload import Workload
    from autoslo.workload_execution.workload_simulator import WorkloadSimulator

    # Runtime API — effective even if PyTorch was already imported.
    import torch
    torch.set_num_threads(int(ncpus))

    # Load the pre-sampled workload from disk.
    workload_df = pd.read_parquet(workload_path)
    workload = Workload(workload_name, schema_name, df=workload_df)
    workload.set_rel_start_times_from_zero()

    if rescale_factor is not None:
        if (config_dict.get("workload_config") or {}).get("closed_loop", False):
            logging.getLogger(__name__).warning(
                "Rescaling workload times with closed_loop=True — "
                "inter-arrival gaps will shrink but closed-loop "
                "feedback may distort the intended speedup."
            )
        workload.rescale_rel_start_times(rescale_factor)

    # Build the simulator with the workload injected at construction time
    # so that all internal counters / tracking state are initialised
    # with the correct workload from the start.
    sim = WorkloadSimulator.from_config_dict(config_dict, workload=workload)

    # Build a progress callback that writes into the shared dict.
    def _progress_cb(current: int, total: int) -> None:
        if progress_dict is not None:
            progress_dict[scenario_idx] = (current, total)

    # Run the simulation.
    sim.simulate_one(
        progress_callback=_progress_cb if progress_dict is not None else None,
    )

    # Extract metrics from the output files.
    return extract_scenario_result(
        out_dir=sim._out_dir,
        scenario_idx=scenario_idx,
        slo_s=slo_s,
        slo_dict=slo_dict,
    )


# ---------------------------------------------------------------------------
# EvalSpec — describes one evaluation within a batch
# ---------------------------------------------------------------------------


@dataclass
class EvalSpec:
    """Specification for a single evaluation within a batch.

    Used by :meth:`ScenarioEvaluator.evaluate_batch` to run multiple
    configurations in a single process pool.
    """

    label: str
    config_overrides: dict[str, Any]
    grid_point: str
    out_subdir: Path


# ---------------------------------------------------------------------------
# ScenarioEvaluator
# ---------------------------------------------------------------------------


class ScenarioEvaluator:
    """Run batches of simulations in parallel and collect results.

    Parameters
    ----------
    initial_config :
        The base simulator config dict (deep-copied before each scenario).
    tuner_config :
        Tuner hyper-parameters (parallelism, aggregation metric, etc.).
    tuner_run_id :
        Unique identifier for the parent tuner run (used to build per-
        scenario ``simulator_run_id`` values).
    evolution_logger :
        :class:`StructuredLogHandler` that receives per-result records
        for the ``evolution.parquet`` ledger.
    """

    def __init__(
        self,
        initial_config: dict[str, Any],
        tuner_config: TunerConfig,
        tuner_run_id: str,
        evolution_logger: StructuredLogHandler,
    ) -> None:
        self._initial_config = initial_config
        self._tuner_config = tuner_config
        self._tuner_run_id = tuner_run_id
        self._evolution_logger = evolution_logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        workload_paths: list[Path],
        config_overrides: dict[str, Any],
        phase: str,
        grid_point: int | str,
        out_subdir: Path,
        schema_name: str | None = None,
    ) -> list[ScenarioResult]:
        """Simulate workloads in parallel and return per-scenario results.

        Parameters
        ----------
        workload_paths :
            Paths to parquet files for each scenario workload, as
            persisted under ``sampled_workloads/`` in the tuner run
            directory (e.g. ``t_000.parquet``, ``v_000.parquet``).
        config_overrides :
            Key/value overrides merged into the base config for every
            scenario (e.g. autoscaler parameters being swept).
        phase :
            Label for the current tuning phase (e.g. ``"baseline"``).
        grid_point :
            Identifier for the current grid/sweep point.
        out_subdir :
            Directory under which per-scenario output dirs are created.
        schema_name :
            Override for the schema name used to construct the
            ``Workload``.  When *None*, falls back to
            ``basic_config.schema_name`` from the initial config.
        """
        out_subdir = Path(out_subdir)
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Pre-extract SLO info from config for the result extractor.
        slo_s: float = float(
            (self._initial_config.get("slo_config") or {}).get("slo_s", 10.0)
        )
        slo_dict_filename: str | None = (
            self._initial_config.get("slo_config") or {}
        ).get("slo_dict_filename")
        # We can't easily pass the filename across processes (it depends on
        # the data path).  But the resolver in the worker already handles
        # violations during simulate_one; here we only need the dict for
        # extract_scenario_result.  Pass None — the worker will recompute.
        slo_dict: dict[str, float] | None = None
        if slo_dict_filename:
            # Try to load the dict now so extract_scenario_result gets it.
            try:
                from autoslo.blueprint_selection.slo_resolver import SloResolver

                resolver = SloResolver(slo_s, slo_dict_filename)
                slo_dict = resolver.slo_dict
            except Exception:
                logger.warning(
                    "Could not pre-load SLO dict %r; violations will use default SLO only.",
                    slo_dict_filename,
                )

        schema = schema_name or (
            (self._initial_config.get("basic_config") or {}).get("schema_name", "default")
        )

        max_workers = self._resolve_parallelism()

        work_units = self._build_work_units(
            workload_paths=workload_paths,
            config_overrides=config_overrides,
            phase=phase,
            grid_point=grid_point,
            out_subdir=out_subdir,
            schema_name=schema,
        )

        results: list[ScenarioResult] = []
        mgr = Manager()
        progress_dict = mgr.dict()

        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(inner_level_num_cpus(),),
        ) as pool:
            futures = {
                pool.submit(
                    _run_scenario,
                    config_dict=wu["config_dict"],
                    workload_path=wu["workload_path"],
                    workload_name=wu["workload_name"],
                    schema_name=wu["schema_name"],
                    scenario_idx=wu["scenario_idx"],
                    slo_s=slo_s,
                    slo_dict=slo_dict,
                    progress_dict=progress_dict,
                    rescale_factor=self._tuner_config.rescale_factor,
                ): wu["scenario_idx"]
                for wu in work_units
            }

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                main_task = progress.add_task(
                    f"[cyan]{phase} gp={grid_point}", total=len(futures)
                )
                # Map scenario_idx → Rich task id for sub-tasks.
                sub_tasks: dict[int, int] = {}
                pending = set(futures.keys())

                while pending:
                    done, pending = wait(
                        pending, timeout=0.3, return_when=FIRST_COMPLETED
                    )

                    # Update sub-task progress bars from shared dict.
                    for idx, (current, total) in list(
                        progress_dict.items()
                    ):
                        if idx not in sub_tasks:
                            sub_tasks[idx] = progress.add_task(
                                f"    scenario {idx}",
                                total=total,
                            )
                        progress.update(
                            sub_tasks[idx], completed=current, total=total
                        )

                    # Handle completed futures.
                    for future in done:
                        idx = futures[future]
                        try:
                            result = future.result()
                        except Exception:
                            logger.exception("Scenario %d failed", idx)
                            raise
                        results.append(result)
                        self._log_result(
                            result, phase, grid_point, config_overrides
                        )
                        progress.advance(main_task)

                        # Remove the sub-task for this scenario.
                        if idx in sub_tasks:
                            progress.remove_task(sub_tasks.pop(idx))
                        progress_dict.pop(idx, None)

        mgr.shutdown()

        # Sort by scenario_idx for deterministic ordering.
        results.sort(key=lambda r: r.scenario_idx)
        return results

    def evaluate_batch(
        self,
        workload_paths: list[Path],
        specs: list[EvalSpec],
        phase: str = "holdout",
        schema_name: str | None = None,
    ) -> list[list[ScenarioResult]]:
        """Evaluate multiple configs in a single pool with unified progress.

        Parameters
        ----------
        workload_paths :
            Paths to workload parquet files (shared across all specs).
        specs :
            Evaluation specifications — one per config to evaluate.
        phase :
            Label for the tuning phase (default ``"holdout"``).
        schema_name :
            Override for the workload schema name.

        Returns
        -------
        A list of :class:`ScenarioResult` lists, one per *spec*, in the
        same order as *specs*.
        """
        if not specs:
            return []

        # Pre-extract SLO info (shared across all specs).
        slo_s: float = float(
            (self._initial_config.get("slo_config") or {}).get("slo_s", 10.0)
        )
        slo_dict: dict[str, float] | None = None
        slo_dict_filename: str | None = (
            self._initial_config.get("slo_config") or {}
        ).get("slo_dict_filename")
        if slo_dict_filename:
            try:
                from autoslo.blueprint_selection.slo_resolver import SloResolver

                resolver = SloResolver(slo_s, slo_dict_filename)
                slo_dict = resolver.slo_dict
            except Exception:
                logger.warning(
                    "Could not pre-load SLO dict %r; using default SLO.",
                    slo_dict_filename,
                )

        schema = schema_name or (
            (self._initial_config.get("basic_config") or {}).get(
                "schema_name", "default"
            )
        )
        max_workers = self._resolve_parallelism()
        n_per_spec = len(workload_paths)

        # Build work units for every spec with globally-unique scenario
        # indices so that progress_dict keys don't collide.
        all_work_units: list[dict[str, Any]] = []
        # global_idx → (spec_idx, original local scenario idx)
        idx_map: dict[int, tuple[int, int]] = {}
        global_idx = 0
        for spec_idx, spec in enumerate(specs):
            units = self._build_work_units(
                workload_paths=workload_paths,
                config_overrides=spec.config_overrides,
                phase=phase,
                grid_point=spec.grid_point,
                out_subdir=Path(spec.out_subdir),
                schema_name=schema,
            )
            for wu in units:
                local_idx = wu["scenario_idx"]
                wu["scenario_idx"] = global_idx
                idx_map[global_idx] = (spec_idx, local_idx)
                all_work_units.append(wu)
                global_idx += 1

        results_by_spec: list[list[ScenarioResult]] = [[] for _ in specs]
        mgr = Manager()
        progress_dict = mgr.dict()

        ctx = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(inner_level_num_cpus(),),
        ) as pool:
            futures: dict[Any, int] = {}
            for wu in all_work_units:
                f = pool.submit(
                    _run_scenario,
                    config_dict=wu["config_dict"],
                    workload_path=wu["workload_path"],
                    workload_name=wu["workload_name"],
                    schema_name=wu["schema_name"],
                    scenario_idx=wu["scenario_idx"],
                    slo_s=slo_s,
                    slo_dict=slo_dict,
                    progress_dict=progress_dict,
                    rescale_factor=self._tuner_config.rescale_factor,
                )
                futures[f] = wu["scenario_idx"]

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                main_task = progress.add_task(
                    f"[cyan]{phase}", total=len(futures)
                )
                spec_tasks = {
                    si: progress.add_task(
                        f"  [bold]{spec.label}[/bold]", total=n_per_spec
                    )
                    for si, spec in enumerate(specs)
                }
                sub_tasks: dict[int, int] = {}
                pending = set(futures.keys())

                while pending:
                    done, pending = wait(
                        pending, timeout=0.3, return_when=FIRST_COMPLETED
                    )

                    # Update per-scenario sub-task bars.
                    for gidx, (current, total) in list(
                        progress_dict.items()
                    ):
                        if gidx not in sub_tasks:
                            si, li = idx_map[gidx]
                            sub_tasks[gidx] = progress.add_task(
                                f"      scenario {li}",
                                total=total,
                            )
                        progress.update(
                            sub_tasks[gidx], completed=current, total=total
                        )

                    for future in done:
                        gidx = futures[future]
                        si, li = idx_map[gidx]
                        spec = specs[si]
                        try:
                            result = future.result()
                        except Exception:
                            logger.exception(
                                "Scenario %d of '%s' failed",
                                li,
                                spec.label,
                            )
                            raise
                        # Restore the local scenario index.
                        result.scenario_idx = li
                        results_by_spec[si].append(result)
                        self._log_result(
                            result,
                            phase,
                            spec.grid_point,
                            spec.config_overrides,
                        )
                        progress.advance(main_task)
                        progress.advance(spec_tasks[si])

                        if gidx in sub_tasks:
                            progress.remove_task(sub_tasks.pop(gidx))
                        progress_dict.pop(gidx, None)

        mgr.shutdown()

        # Sort each spec's results by scenario_idx.
        for results_list in results_by_spec:
            results_list.sort(key=lambda r: r.scenario_idx)

        return results_by_spec

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_parallelism(self) -> int:
        p = self._tuner_config.parallelism
        if p == "auto":
            return max(1, deg_of_paralellism())
        return int(p)

    def _build_work_units(
        self,
        workload_paths: list[Path],
        config_overrides: dict[str, Any],
        phase: str,
        grid_point: int | str,
        out_subdir: Path,
        schema_name: str,
    ) -> list[dict[str, Any]]:
        """Prepare one work-unit dict per workload."""
        from autoslo.utils.config import apply_overrides

        units: list[dict[str, Any]] = []

        for idx, wl_path in enumerate(workload_paths):
            cfg = copy.deepcopy(self._initial_config)
            apply_overrides(cfg, config_overrides)

            # Set per-scenario identifiers.
            run_id = f"{self._tuner_run_id}_{phase}_{grid_point}_{idx:03d}"
            cfg.setdefault("basic_config", {})["simulator_run_id"] = run_id
            cfg.setdefault("output_config", {})["out_dir"] = str(out_subdir)
            # Ensure verbose so structured_log.parquet is written.
            cfg.setdefault("output_config", {})["verbose"] = True
            # Suppress experiment_meta writing (no experiment_name in tuner context).
            cfg.setdefault("basic_config", {})["experiment_name"] = None

            # Derive a workload name from the parquet filename (e.g. t_000).
            workload_name = Path(wl_path).stem

            units.append(
                {
                    "config_dict": cfg,
                    "workload_path": str(wl_path),
                    "workload_name": workload_name,
                    "schema_name": schema_name,
                    "scenario_idx": idx,
                }
            )
        return units

    def _log_result(
        self,
        result: ScenarioResult,
        phase: str,
        grid_point: int | str,
        config_overrides: dict[str, Any],
    ) -> None:
        """Emit a record to the evolution ledger."""
        emit_structured(
            {
                "timestamp": pd.Timestamp.now().isoformat(),
                "source": "tuner",
                "event_type": "scenario_result",
                "phase": phase,
                "grid_point": str(grid_point),
                "scenario_idx": result.scenario_idx,
                "violation_rate": result.violation_rate,
                "violation_amount_s": result.violation_amount_s,
                "violation_relative_mean": result.violation_relative_mean,
                "total_cost": result.total_cost,
                "num_queries": result.num_queries,
                "config_overrides": str(config_overrides),
            }
        )
