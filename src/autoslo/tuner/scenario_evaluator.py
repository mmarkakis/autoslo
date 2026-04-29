"""ScenarioEvaluator — runs N simulations in parallel and collects results."""

from __future__ import annotations

import itertools
import logging
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager, get_context
from pathlib import Path
from typing import Any

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

import autoslo.config.utils as cfgu
from autoslo.config.component_configs import WorkloadConfig
from autoslo.simulator.simulation_result import SimulationResult
from autoslo.simulator.workload_simulator import WorkloadSimulator
from autoslo.tuner.parallelism import (
    _init_worker,
    deg_of_parallelism,
    inner_level_num_cpus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level worker (must be picklable — no closures / lambdas)
# ---------------------------------------------------------------------------


def to_combination_idx(
    config_idx: int, workload_idx: int, num_workloads: int
) -> int:
    return config_idx * num_workloads + workload_idx


def from_combination_idx(
    combination_idx: int, num_workloads: int
) -> tuple[int, int]:
    config_idx = combination_idx // num_workloads
    workload_idx = combination_idx % num_workloads
    return config_idx, workload_idx


def _run_one_combination(
    sim_out_dir: Path,
    combination_idx: int,
    workload_config: WorkloadConfig,
    config: dict[str, Any],
    progress_dict,
) -> SimulationResult:
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

    # Runtime API — effective even if PyTorch was already imported.
    torch.set_num_threads(int(ncpus))

    # Overwrite the workload in the config with the one we want to run.
    config = cfgu.copy_and_apply_overrides(
        config, {"workload_config": workload_config.to_dict()}
    )

    # Build and run the simulator.
    def _progress_cb(current: int, total: int) -> None:
        progress_dict[combination_idx] = (current, total)

    sim = WorkloadSimulator(config, out_dir=sim_out_dir, write_text_log=False)
    result = sim.run(progress_callback=_progress_cb)
    return result


# ---------------------------------------------------------------------------
# ScenarioEvaluator
# ---------------------------------------------------------------------------


class ScenarioEvaluator:
    """Run batches of simulations in parallel and collect results.

    Parameters
    ----------
    max_workers :
        Optional explicit override for the parallel worker count. When
        ``None``, defaults to `deg_of_parallelism()`.
    """

    def __init__(
        self,
        max_workers: int | None = None,
    ) -> None:
        self._max_workers = max_workers or max(1, deg_of_parallelism())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_batch_from_overrides(
        self,
        progress_bar_label: str,
        out_dir: Path,
        workload_configs: list[WorkloadConfig],
        base_config: dict[str, Any],
        all_config_overrides: list[dict[str, Any]],
        config_labels: list[str] | None = None,
        nest_outputs_by_config: bool = True,
    ) -> list[list[SimulationResult]]:
        """
        Convenience wrapper around :meth:`evaluate_batch_from_configs` that
        accepts a list of override dicts instead of full configs, and applies
        the overrides to *base_config* to construct the full configs for each
        workload.
        """

        configs = [
            cfgu.copy_and_apply_overrides(base_config, config_overrides)
            for config_overrides in all_config_overrides
        ]
        return self.evaluate_batch_from_configs(
            progress_bar_label=progress_bar_label,
            out_dir=out_dir,
            workload_configs=workload_configs,
            configs=configs,
            config_labels=config_labels,
            nest_outputs_by_config=nest_outputs_by_config,
        )

    def evaluate_batch_from_configs(
        self,
        progress_bar_label: str,
        out_dir: str | Path,
        workload_configs: list[WorkloadConfig],
        configs: list[dict[str, Any]],
        config_labels: list[str] | None = None,
        nest_outputs_by_config: bool = True,
    ) -> list[list[SimulationResult]]:
        """
        Evaluate the cross-product of *workload_configs* and *configs* in
        parallel, with unified progress tracking and result collection.

        Parameters
        ----------
        progress_bar_label :
            Label for the tuning phase.
        out_dir :
            Directory under which per-scenario subdirectories will be created
            to hold simulation outputs.
        workload_configs :
            List of workload configurations (shared across all specs).
        configs :
            List of full config dicts for every scenario.

        Returns
        -------
        A nested list where ``results[config_idx][workload_idx]`` is the
        :class:`SimulationResult` for that combination.  The outer list
        is ordered by *configs* and the inner list by *workload_configs*.
        """

        if len(configs) == 0 or len(workload_configs) == 0:
            raise ValueError(
                "Must provide at least one config config and one workload"
            )

        if config_labels is not None and len(config_labels) != len(configs):
            raise ValueError(
                "Length of config_labels must match length of configs"
            )
        if config_labels is None:
            config_labels = [f"config_{i}" for i in range(len(configs))]

        # Set up parallel execution.
        mgr = Manager()
        progress_dict = mgr.dict()
        ctx = get_context("spawn")
        num_workloads = len(workload_configs)
        results: dict[int, dict[int, SimulationResult]] = {}
        out_dir = Path(out_dir)

        # Execute in parallel.
        with ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(inner_level_num_cpus(),),
        ) as pool:
            futures: dict[Any, int] = {}
            for workload_idx, config_idx in itertools.product(
                range(len(workload_configs)), range(len(configs))
            ):
                combination_idx = to_combination_idx(
                    config_idx, workload_idx, num_workloads=num_workloads
                )
                config_label = config_labels[config_idx]
                workload_config = workload_configs[workload_idx]
                if nest_outputs_by_config:
                    sim_out_dir = out_dir / config_label / workload_config.id()
                else:
                    sim_out_dir = (
                        out_dir / f"{config_label}#{workload_config.id()}"
                    )
                f = pool.submit(
                    _run_one_combination,
                    sim_out_dir,
                    combination_idx,
                    workload_config,
                    configs[config_idx],
                    progress_dict,
                )
                futures[f] = combination_idx

            # Add a timer column with projected remaining time.
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=False,
                refresh_per_second=1,
            ) as progress:
                main_task = progress.add_task(
                    f"[cyan]{progress_bar_label}", total=len(futures)
                )
                per_config_tasks = {
                    config_idx: progress.add_task(
                        f"  [bold]config {config_idx}[/bold]",
                        total=num_workloads,
                    )
                    for config_idx in range(len(configs))
                }
                sub_tasks: dict[int, int] = {}
                pending = set(futures.keys())

                while pending:
                    done, pending = wait(
                        pending, timeout=0.3, return_when=FIRST_COMPLETED
                    )

                    for combination_idx, (current, total) in list(
                        progress_dict.items()
                    ):
                        if combination_idx not in sub_tasks:
                            config_idx, workload_idx = from_combination_idx(
                                combination_idx,
                                num_workloads=num_workloads,
                            )
                            sub_tasks[combination_idx] = progress.add_task(
                                f"    config {config_idx} - workload {workload_idx}",
                                total=total,
                            )
                        progress.update(
                            sub_tasks[combination_idx],
                            completed=current,
                            total=total,
                        )

                    for future in done:
                        combination_idx = futures[future]
                        config_idx, workload_idx = from_combination_idx(
                            combination_idx,
                            num_workloads=num_workloads,
                        )
                        try:
                            result = future.result()
                        except Exception:
                            logger.exception(
                                "Simulation failed for config #%d, workload #%d",
                                config_idx,
                                workload_idx,
                            )
                            raise
                        # Restore the local scenario index.
                        if config_idx not in results:
                            results[config_idx] = {}
                        results[config_idx][workload_idx] = result
                        progress.advance(main_task)
                        progress.advance(per_config_tasks[config_idx])

                        if combination_idx in sub_tasks:
                            progress.remove_task(sub_tasks.pop(combination_idx))
                        progress_dict.pop(combination_idx, None)

                        if len(results[config_idx]) == num_workloads:
                            progress.remove_task(per_config_tasks[config_idx])

        mgr.shutdown()

        # Convert the sparse dict-of-dicts into a list-of-lists ordered
        # by config index (outer) and workload index (inner).
        return [
            [results[ci][wi] for wi in range(num_workloads)]
            for ci in range(len(configs))
        ]
