"""
Run the PolicyTuner sequentially for every trial in an experiment spec.

Reads ``trial_spec.yml`` and invokes ``policy_tuner.py`` for each trial,
passing the execution config, tuner config, and ``--param KEY=VALUE`` flags
derived from the trial's ``params`` dict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.tuner.policy_tuner import AlreadyCompleteError, PolicyTuner

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PolicyTuner for each trial defined in a trial spec."
    )
    parser.add_argument(
        "spec",
        help="Path to trial_spec.yml, or to the experiment directory containing it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run all trials unconditionally, even if their tuned configs "
            "are already up to date."
        ),
    )
    args = parser.parse_args()

    given = Path(args.spec).resolve()
    if given.is_dir():
        spec_path = given / "trial_spec.yml"
    else:
        spec_path = given
    if not spec_path.exists():
        parser.error(f"Spec file not found: {spec_path}")

    spec_dir = spec_path.parent  # used by plot_experiment
    spec = load_yaml(spec_path)

    default_exec_cfg: str = spec.get("exec_config", "")
    default_tuner_cfg: str = spec.get("tuner_config", "")

    trials = sorted(
        spec.get("trials", []), key=lambda t: t.get("sort_order", 0)
    )

    if not trials:
        print("[run_trials] WARNING: no trials found in spec; nothing to run.")
        return

    total = len(trials)
    for idx, trial in enumerate(trials, 1):
        tid = trial["trial_id"]
        exec_cfg = trial.get("exec_config", default_exec_cfg)
        tuner_cfg = trial.get("tuner_config", default_tuner_cfg)
        params: dict[str, str] = dict(trial.get("params") or {})

        console.print(
            f"\n[bold]({idx}/{total})[/] Tuning trial [cyan]{tid}[/] "
            f"(exec={exec_cfg}, tuner={tuner_cfg})"
        )
        try:
            pt = PolicyTuner(
                initial_execution_config_path=exec_cfg,
                tuner_config_path=tuner_cfg,
                force=args.force,
                params=params or None,
            )
        except AlreadyCompleteError as exc:
            console.print(f"[dim]Skipping '{tid}': {exc}[/]")
            continue
        pt.tune()

    console.print(f"\n[bold green]All {total} trial(s) completed.[/]")


if __name__ == "__main__":
    main()
