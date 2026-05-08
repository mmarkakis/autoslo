from __future__ import annotations

import argparse
import asyncio
import shutil
from collections import defaultdict
from pathlib import Path

from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import WorkloadConfig
from autoslo.config.utils import copy_and_apply_overrides
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.path_utils import (
    append_to_run_log,
    find_most_recent_live_run_id,
    is_up_to_date,
)
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.workload_execution.workload_runner import WorkloadRunner

console = Console()


def main():
    # Argument parsing.
    description = (
        "Run the workload simulator or workload runner from an "
        "execution manifest."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "execution_manifest",
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
    args = parser.parse_args()

    manifest_path = Path(args.execution_manifest)
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
        _run_simulator(entries, data_path, force=args.force)
    else:
        _run_live(entries, force=args.force)

    console.print("\n[bold green]Execution complete.[/]")


def _run_simulator(entries: list[dict], data_path: Path, force: bool) -> None:
    """Dispatch all (workload, exec_config) pairs through the simulator."""
    sim_runs_dir = data_path / "simulator_runs"
    evaluator = ScenarioEvaluator()

    # Group entries by workload_id so each workload is dispatched in one batch.
    workload_by_wid: dict[str, WorkloadConfig] = {}
    configs_by_wid: dict[str, list[dict]] = defaultdict(list)
    labels_by_wid: dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        workload_config = WorkloadConfig.from_config(entry)
        exec_cfg_path = resolve_config(entry["execution_config"])
        config_label = exec_cfg_path.stem
        wid = workload_config.id()

        out_dir = sim_runs_dir / wid / config_label
        if not force and is_up_to_date(
            out_dir / "execution_config.yml", exec_cfg_path
        ):
            console.print(
                f"[dim]Skipping '{config_label}' for '{wid}' (up to date)[/]"
            )
            continue

        if out_dir.exists():
            shutil.rmtree(out_dir)

        workload_by_wid[wid] = workload_config
        configs_by_wid[wid].append(load_yaml(exec_cfg_path))
        labels_by_wid[wid].append(config_label)

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


def _run_live(entries: list[dict], force: bool) -> None:
    """
    Run each (workload, exec_config) pair sequentially against live clusters.
    """
    runs_path = Path(pu.get_runs_path())
    for entry in entries:
        workload_config = WorkloadConfig.from_config(entry)
        exec_cfg_path = resolve_config(entry["execution_config"])
        config_label = exec_cfg_path.stem
        wid = workload_config.id()

        if not force:
            recent_run_id = find_most_recent_live_run_id(config_label, wid)
            if recent_run_id is not None and is_up_to_date(
                runs_path / recent_run_id / "config.yml", exec_cfg_path
            ):
                console.print(
                    f"[dim]Skipping '{config_label}' for '{wid}' (up to date)[/]"
                )
                continue

        cfg = load_yaml(exec_cfg_path)
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


if __name__ == "__main__":
    main()
