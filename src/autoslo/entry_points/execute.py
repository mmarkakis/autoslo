from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import WorkloadConfig
from autoslo.config.utils import copy_and_apply_overrides, make_run_id
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.path_utils import (
    append_to_run_log,
    find_most_recent_live_run_id,
    is_up_to_date,
)
from autoslo.filesystem.yaml_helpers import load_yaml, load_yaml_with_params
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.workload_execution.execution_result import ExecutionResult
from autoslo.workload_execution.workload_runner import WorkloadRunner

console = Console()


@dataclass
class _RunRecord:
    label: str
    wid: str
    run_dir: Path
    was_run: bool


def _print_summary(records: list[_RunRecord], wall_elapsed_s: float) -> None:
    n_specified = len(records)
    n_run = sum(r.was_run for r in records)

    total_sim_s = 0.0
    total_cost = 0.0
    for r in records:
        try:
            result = ExecutionResult.load(r.run_dir)
            total_sim_s += result.total_rel_time_s or 0.0
            total_cost += result.total_cost
        except Exception:
            pass

    table = Table(
        title="Execution Summary", show_header=False, box=None, padding=(0, 2)
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Wall-clock time", f"{wall_elapsed_s:.1f} s")
    table.add_row("Runs specified", str(n_specified))
    table.add_row("Runs actually executed", str(n_run))
    table.add_row("Total run time (rel_time_s)", f"{total_sim_s:.1f} s")
    table.add_row("Total cost", f"${total_cost:.4f}")
    console.print()
    console.print(table)


def main():
    t_start = time.monotonic()
    # Argument parsing.
    description = (
        "Run the workload simulator or workload runner from an "
        "execution manifest."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--execution_manifest_path",
        help=(
            "Name of the execution manifest (resolved under "
            "data/manifests/execution) or an explicit path to a .yml file."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run sequentially using the workload runner, not the simulator.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing run/output directory if present. "
            "Commands that don't manage a run directory ignore this flag."
        ),
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=1,
        help=(
            "Total number of parallel splits. Use with --split-index. Only "
            "used for live runs."
        ),
    )
    parser.add_argument(
        "--split_index",
        type=int,
        default=0,
        help=(
            "Zero-based index of the split this process should run "
            "(0 <= K < --splits). Only used for live runs."
        ),
    )
    args = parser.parse_args()

    if args.splits < 1:
        parser.error("--splits must be >= 1")
    if not (0 <= args.split_index < args.splits):
        parser.error("--split-index must satisfy 0 <= split-index < splits")


    manifest_path = Path(args.execution_manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = (
            Path(pu.get_data_path())
            / "manifests"
            / "execution"
            / manifest_path.with_suffix(".yml")
        )
    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}")

    manifest = load_yaml(manifest_path)
    entries = manifest.get("main_content", [])
    data_path = Path(pu.get_data_path())

    if not args.live:
        records = _run_simulator(entries, data_path, force=args.force)
    else:
        split_entries = [
            e
            for i, e in enumerate(entries)
            if i % args.splits == args.split_index
        ]
        if args.splits > 1:
            console.print(
                f"[bold]Split {args.split_index} of {args.splits}:[/] "
                f"{len(split_entries)} of {len(entries)} entries assigned to "
                f"this split."
            )
        records = _run_live(split_entries, force=args.force)

    console.print("\n[bold green]Execution complete.[/]")
    _print_summary(records, wall_elapsed_s=time.monotonic() - t_start)


def _run_simulator(
    entries: list[dict], data_path: Path, force: bool
) -> list[_RunRecord]:
    """Dispatch all (workload, exec_config) pairs through the simulator."""
    sim_runs_dir = data_path / "simulator_runs"
    evaluator = ScenarioEvaluator()

    # Group entries by workload_id so each workload is dispatched in one batch.
    workload_by_wid: dict[str, WorkloadConfig] = {}
    configs_by_wid: dict[str, list[dict]] = defaultdict(list)
    labels_by_wid: dict[str, list[str]] = defaultdict(list)
    records: list[_RunRecord] = []

    for entry in entries:
        workload_config = WorkloadConfig.from_config(entry)
        exec_cfg_path = resolve_config(entry["execution_config"])
        params = entry.get("params", {})
        config_label = make_run_id([exec_cfg_path.stem], params)
        wid = workload_config.id()

        out_dir = sim_runs_dir / wid / config_label
        if not force and is_up_to_date(
            out_dir / "execution_config.yml", exec_cfg_path
        ):
            console.print(
                f"[dim]Skipping '{config_label}' for '{wid}' (up to date)[/]"
            )
            records.append(
                _RunRecord(config_label, wid, out_dir, was_run=False)
            )
            continue

        if out_dir.exists():
            shutil.rmtree(out_dir)

        workload_by_wid[wid] = workload_config
        configs_by_wid[wid].append(load_yaml_with_params(exec_cfg_path, params))
        labels_by_wid[wid].append(config_label)
        records.append(_RunRecord(config_label, wid, out_dir, was_run=True))

    for wid, workload_config in workload_by_wid.items():
        configs = configs_by_wid[wid]
        labels = labels_by_wid[wid]
        console.print(
            f"\n[cyan]Dispatching {len(configs)} config(s) "
            f"for workload '{wid}'[/]"
        )
        evaluator.evaluate_batch_from_configs(
            progress_bar_label=wid,
            out_dir=sim_runs_dir,
            workload_configs=[workload_config],
            configs=configs,
            config_labels=labels,
            workload_first=True,
            render_log=True,
        )

    return records


def _run_live(entries: list[dict], force: bool) -> list[_RunRecord]:
    """
    Run each (workload, exec_config) pair sequentially against live clusters.
    """
    runs_path = Path(pu.get_runs_path())
    records: list[_RunRecord] = []
    for entry in entries:
        workload_config = WorkloadConfig.from_config(entry)
        exec_cfg_path = resolve_config(entry["execution_config"])
        params = entry.get("params", {})
        config_label = make_run_id([exec_cfg_path.stem], params)
        wid = workload_config.id()

        if not force:
            recent_run_id = find_most_recent_live_run_id(config_label, wid)
            if recent_run_id is not None and is_up_to_date(
                runs_path / recent_run_id / "config.yml", exec_cfg_path
            ):
                console.print(
                    f"[dim]Skipping '{config_label}' for '{wid}' (up to date)[/]"
                )
                records.append(
                    _RunRecord(
                        config_label,
                        wid,
                        runs_path / recent_run_id,
                        was_run=False,
                    )
                )
                continue

        cfg = load_yaml_with_params(exec_cfg_path, params)
        cfg = copy_and_apply_overrides(
            cfg, {"workload_config": workload_config.to_dict()}
        )

        console.print(
            f"\n[cyan]Running '{config_label}' for workload '{wid}'[/]"
        )
        runner = WorkloadRunner(cfg)
        append_to_run_log(
            run_id=runner.run_id, config_id=config_label, workload_id=wid
        )
        asyncio.run(runner.run())
        records.append(
            _RunRecord(
                config_label, wid, runs_path / runner.run_id, was_run=True
            )
        )

    return records


if __name__ == "__main__":
    main()
