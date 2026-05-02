"""
Run the PolicyTuner sequentially for every trial in an experiment spec.

Reads ``trial_spec.yml`` and invokes ``policy_tuner.py`` for each trial,
passing the execution config, tuner config, and ``--param KEY=VALUE`` flags
derived from the trial's ``params`` dict.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from autoslo.experiments.aggregate_trials import plot_experiment
from autoslo.filesystem.yaml_helpers import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PolicyTuner for each trial defined in a trial spec."
    )
    parser.add_argument(
        "spec",
        help="Path to trial_spec.yml, or to the experiment directory containing it.",
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

        cmd = [
            sys.executable,
            "src/autoslo/tuner/policy_tuner.py",
            exec_cfg,
            tuner_cfg,
            "--force",
        ]
        for key, val in sorted(params.items()):
            cmd += ["--param", f"{key}={val}"]
        print(
            f"\n[run_trials] ({idx}/{total}) trial '{tid}' — "
            f"{' '.join(cmd)}"
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"\n[run_trials] ERROR: tuner exited with code "
                f"{result.returncode} for trial '{tid}'.",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

    print(f"\n[run_trials] All {total} trial(s) completed successfully.")

    print("Plotting results...")
    plot_experiment(spec_dir)


if __name__ == "__main__":
    main()
