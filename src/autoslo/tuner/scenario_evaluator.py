"""ScenarioEvaluator — runs N simulations in parallel and collects results."""

from __future__ import annotations

import contextlib
import cProfile
import io
import itertools
import logging
import os
import pstats
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager, get_context
from pathlib import Path
from typing import Any, Optional

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
from autoslo.tuner.parallelism import (
    _init_worker,
    deg_of_parallelism,
    inner_level_num_cpus,
)
from autoslo.workload_execution.execution_result import ExecutionResult
from autoslo.workload_execution.workload_simulator import WorkloadSimulator

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


# Worker-process global: populated by _init_worker_with_configs before any
# task runs, so configs do not need to be pickled per task submission.
_worker_configs: list[dict[str, Any]] | None = None


def _init_worker_with_configs(
    inner_cpus: int, configs: list[dict[str, Any]]
) -> None:
    """Extend :func:`_init_worker` by pre-loading configs into a worker-global."""
    _init_worker(inner_cpus)
    global _worker_configs
    _worker_configs = configs


def _run_one_combination(
    sim_out_dir: Path,
    combination_idx: int,
    workload_config: WorkloadConfig,
    config_idx: int,
    render_log: bool,
    progress_dict,
) -> ExecutionResult:
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
        _worker_configs[config_idx],  # type: ignore[index]
        {"workload_config": workload_config.to_dict()},
    )

    # Build and run the simulator.
    def _progress_cb(current: int, total: int) -> None:
        if progress_dict is not None:
            progress_dict[combination_idx] = (current, total)

    with (
        open(os.devnull, "w") as _devnull,
        contextlib.redirect_stdout(_devnull),
        contextlib.redirect_stderr(_devnull),
    ):
        sim = WorkloadSimulator(
            config, out_dir=sim_out_dir, write_text_log=False
        )
        result = sim.run(progress_callback=_progress_cb, render_log=render_log)
    return result


def _run_one_combination_profiled(
    sim_out_dir: Path,
    combination_idx: int,
    workload_config: WorkloadConfig,
    config_idx: int,
    render_log: bool,
    progress_dict,
    profile_file: str,
) -> ExecutionResult:
    """Thin wrapper around :func:`_run_one_combination` that runs it under
    ``cProfile`` and dumps the stats to *profile_file* before returning."""
    profiler = cProfile.Profile()
    result = profiler.runcall(
        _run_one_combination,
        sim_out_dir,
        combination_idx,
        workload_config,
        config_idx,
        render_log,
        progress_dict,
    )
    profiler.dump_stats(profile_file)
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
        workload_first: bool = True,
        render_log: bool = False,
        verbose_progress: bool = True,
        profile_path: Optional[Path] = None,
    ) -> list[list[ExecutionResult]]:
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
            workload_first=workload_first,
            render_log=render_log,
            verbose_progress=verbose_progress,
            profile_path=profile_path,
        )

    def evaluate_batch_from_configs(
        self,
        progress_bar_label: str,
        out_dir: Path,
        workload_configs: list[WorkloadConfig],
        configs: list[dict[str, Any]],
        config_labels: list[str] | None = None,
        workload_first: bool = True,
        render_log: bool = False,
        verbose_progress: bool = True,
        profile_path: Optional[Path] = None,
    ) -> list[list[ExecutionResult]]:
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
        workload_first :
            When ``True`` (default), output directories are nested as
            ``out_dir / workload_id / config_label``.  When ``False``
            (config-first, used by the tuner), they are nested as
            ``out_dir / config_label / workload_id``.

        Returns
        -------
        A nested list where ``results[config_idx][workload_idx]`` is the
        :class:`ExecutionResult` for that combination.  The outer list
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
        if verbose_progress:
            mgr = Manager()
            progress_dict = mgr.dict()
        else:
            mgr = None
            progress_dict = None
        ctx = get_context("spawn")
        num_workloads = len(workload_configs)
        results: dict[int, dict[int, ExecutionResult]] = {}
        out_dir = Path(out_dir)

        # Execute in parallel.
        _prof_tmpdir = (
            tempfile.TemporaryDirectory() if profile_path is not None else None
        )
        prof_dir: str | None = (
            _prof_tmpdir.name if _prof_tmpdir is not None else None
        )
        profile_files: list[str] = []
        with ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=ctx,
            initializer=_init_worker_with_configs,
            initargs=(inner_level_num_cpus(), configs),
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
                if workload_first:
                    sim_out_dir = out_dir / workload_config.id() / config_label
                else:
                    sim_out_dir = out_dir / config_label / workload_config.id()
                if prof_dir is not None:
                    prof_file = str(
                        Path(prof_dir) / f"profile_{combination_idx}.prof"
                    )
                    profile_files.append(prof_file)
                    f = pool.submit(
                        _run_one_combination_profiled,
                        sim_out_dir,
                        combination_idx,
                        workload_config,
                        config_idx,
                        render_log,
                        progress_dict,
                        prof_file,
                    )
                else:
                    f = pool.submit(
                        _run_one_combination,
                        sim_out_dir,
                        combination_idx,
                        workload_config,
                        config_idx,
                        render_log,
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
                per_config_tasks = (
                    {
                        config_idx: progress.add_task(
                            f"  [bold]{config_label}[/bold]",
                            total=num_workloads,
                        )
                        for config_idx, config_label in enumerate(config_labels)
                    }
                    if verbose_progress
                    else {}
                )
                sub_tasks: dict[int, int] = {}
                sub_tasks_lock = threading.Lock()
                completed_combos: set[int] = set()
                pending = set(futures.keys())

                # Background thread refreshes per-simulation sub-task bars so
                # the main loop can block on wait() without a polling timeout,
                # eliminating result-collection latency for fast simulations.
                stop_poll = threading.Event()

                def _poll_sub_task_progress() -> None:
                    while not stop_poll.wait(0.3):
                        assert progress_dict is not None
                        for combo_idx, (current, total) in list(
                            progress_dict.items()
                        ):
                            with sub_tasks_lock:
                                if combo_idx in completed_combos:
                                    continue
                                if combo_idx not in sub_tasks:
                                    ci, wi = from_combination_idx(
                                        combo_idx, num_workloads=num_workloads
                                    )
                                    sub_tasks[combo_idx] = progress.add_task(
                                        f"    config {ci} - workload {wi}",
                                        total=total,
                                    )
                                task_id = sub_tasks[combo_idx]
                            try:
                                progress.update(
                                    task_id, completed=current, total=total
                                )
                            except KeyError:
                                pass  # task removed between lock release and update

                poll_thread: threading.Thread | None = None
                if verbose_progress:
                    poll_thread = threading.Thread(
                        target=_poll_sub_task_progress, daemon=True
                    )
                    poll_thread.start()

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)

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

                        if verbose_progress:
                            progress.advance(per_config_tasks[config_idx])
                            with sub_tasks_lock:
                                completed_combos.add(combination_idx)
                                task_id = sub_tasks.pop(combination_idx, None)
                            if task_id is not None:
                                progress.remove_task(task_id)
                        if progress_dict is not None:
                            progress_dict.pop(combination_idx, None)

                        if (
                            verbose_progress
                            and len(results[config_idx]) == num_workloads
                        ):
                            progress.remove_task(per_config_tasks[config_idx])

                if poll_thread is not None:
                    stop_poll.set()
                    poll_thread.join()

        # All workers finished; merge per-process profiles if requested.
        if _prof_tmpdir is not None:
            existing = [f for f in profile_files if Path(f).exists()]
            if existing:
                combined = pstats.Stats(existing[0], stream=io.StringIO())
                for f in existing[1:]:
                    combined.add(f)
                combined.dump_stats(str(profile_path))
            _prof_tmpdir.cleanup()

        if mgr is not None:
            mgr.shutdown()

        # Convert the sparse dict-of-dicts into a list-of-lists ordered
        # by config index (outer) and workload index (inner).
        return [
            [results[ci][wi] for wi in range(num_workloads)]
            for ci in range(len(configs))
        ]
