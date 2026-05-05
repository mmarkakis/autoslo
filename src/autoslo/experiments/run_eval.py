"""Run simulator evaluations defined in an eval manifest.

Reads ``data/simulator_eval_specs/<name>.yml`` (or a direct path), resolves
each trial's tuned execution config, and dispatches all (config, workload)
pairs to :class:`ScenarioEvaluator`.

Output layout::

    data/simulator_runs/<workload_id>/<exec_config_id>/
        execution_config.yml
        structured_log.parquet
        billing_interval_analysis.yml
        mapping.yml

Runs are globally addressable by ``(workload_id, exec_config_id)`` and are
shared across eval manifests.  A run is skipped if its output directory
already contains ``execution_config.yml``, unless ``--force`` is passed.

Usage (from repo root)::

    python src/autoslo/experiments/run_eval.py <manifest_name_or_path> [--force]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import WorkloadConfig
from autoslo.config.utils import make_run_id
from autoslo.filesystem.config_resolver import expand_eval_baselines, resolve_config
from autoslo.filesystem.path_utils import is_up_to_date
from autoslo.filesystem.yaml_helpers import load_yaml, load_yaml_with_params
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simulator evaluations defined in an eval manifest."
    )
    parser.add_argument(
        "manifest",
        help=(
            "Name of the eval manifest (resolved under data/simulator_eval_specs/) "
            "or an explicit path to a .yml file."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run and overwrite already-completed simulation output directories.",
    )
    args = parser.parse_args()

    # Resolve the manifest path. If the argument is a direct path, use it.
    # Otherwise, resolve it under data/simulator_eval_specs/.
    p = Path(args.manifest)
    if p.is_absolute():
        manifest_path = p
    else:
        name = (
            args.manifest
            if args.manifest.endswith(".yml")
            else args.manifest + ".yml"
        )
        manifest_path = Path(pu.get_data_path()) / "simulator_eval_specs" / name
    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}")
    manifest = load_yaml(manifest_path)

    data_path = Path(pu.get_data_path())
    root_path = Path(pu.AUTOSLO_ROOT)
    sim_runs_dir = data_path / "simulator_runs"
    evaluator = ScenarioEvaluator(max_workers=manifest.get("max_workers"))

    for entry in manifest["workload_runs"]:
        workload_config = WorkloadConfig.from_config(entry)
        configs: list[dict] = []
        labels: list[str] = []

        for spec_ref in entry["trials"]:
            trial_spec_path = (
                root_path / spec_ref["spec_dir"] / "trial_spec.yml"
            )
            if not trial_spec_path.exists():
                console.print(
                    f"[yellow]Warning: trial spec not found: {trial_spec_path}[/]"
                )
                continue
            trial_spec = load_yaml(trial_spec_path)
            default_exec: str = trial_spec.get("exec_config", "")
            default_tuner: str = trial_spec.get("tuner_config", "")
            allowed_ids: set[str] = set(spec_ref.get("trial_ids") or [])

            for trial in trial_spec.get("trials", []):
                if allowed_ids and trial["trial_id"] not in allowed_ids:
                    continue

                exec_cfg: str = trial.get("exec_config", default_exec)
                tuner_cfg: str = trial.get("tuner_config", default_tuner)
                params: dict[str, str] = dict(trial.get("params") or {})
                run_id = make_run_id(
                    [
                        resolve_config(exec_cfg).stem,
                        resolve_config(tuner_cfg).stem,
                    ],
                    params,
                )
                tuned_path = (
                    data_path / "execution_configs" / "tuned" / f"{run_id}.yml"
                )

                if not tuned_path.exists():
                    console.print(
                        f"[yellow]Skipping '{trial['trial_id']}': "
                        f"tuned config not found at {tuned_path}[/]"
                    )
                    continue

                out_dir = sim_runs_dir / workload_config.id() / run_id
                if not args.force and is_up_to_date(
                    out_dir / "execution_config.yml", tuned_path
                ):
                    console.print(f"[dim]Skipping '{run_id}' (up to date)[/]")
                    continue

                if out_dir.exists():
                    shutil.rmtree(out_dir)

                configs.append(load_yaml(tuned_path))
                labels.append(run_id)

        # Baselines: exec configs run directly without a tuner step.
        # Each entry: {exec_config: "path/to/cfg.yml", params: {KEY: val}}
        # run_id = make_run_id([exec_stem], params)  (no tuner stem).
        for baseline in expand_eval_baselines(entry.get("baselines", [])):
            exec_cfg: str = baseline["exec_config"]
            params: dict[str, str] = dict(baseline.get("params") or {})
            exec_cfg_path = resolve_config(exec_cfg)
            run_id = make_run_id([exec_cfg_path.stem], params)

            out_dir = sim_runs_dir / workload_config.id() / run_id
            if not args.force and is_up_to_date(
                out_dir / "execution_config.yml", exec_cfg_path
            ):
                console.print(
                    f"[dim]Skipping baseline '{run_id}' (up to date)[/]"
                )
                continue

            if out_dir.exists():
                shutil.rmtree(out_dir)

            configs.append(load_yaml_with_params(exec_cfg_path, params))
            labels.append(run_id)

        if not configs:
            console.print(
                f"[dim]No runs to dispatch for workload '{workload_config.id()}'[/]"
            )
            continue

        console.print(
            f"\n[cyan]Dispatching {len(configs)} config(s) "
            f"for workload '{workload_config.id()}'[/]"
        )
        evaluator.evaluate_batch_from_configs(
            progress_bar_label=f"{manifest['evaluation_id']} \u2014 {workload_config.id()}",
            out_dir=sim_runs_dir,
            workload_configs=[workload_config],
            configs=configs,
            config_labels=labels,
            workload_first=True,
            render_log=True,
        )

    console.print("\n[bold green]Evaluation complete.[/]")


if __name__ == "__main__":
    main()
