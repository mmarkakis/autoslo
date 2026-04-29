"""
Run the PolicyTuner sequentially for every trial in an experiment spec.

Reads ``trial_spec.yml``, infers the path of each generated config
(``configs/tuner_{id}.yml``), and invokes ``policy_tuner.py`` for each
trial in ascending ``sort_order``.
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
    parser.add_argument("spec", help="Path to trial_spec.yml")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        parser.error(f"Spec file not found: {spec_path}")

    spec_dir = spec_path.parent
    spec = load_yaml(spec_path)

    configs_dir = spec_dir / "configs"
    if not configs_dir.exists():
        parser.error(f"Configs directory not found: {configs_dir}")
    trials = sorted(
        spec.get("trials", []), key=lambda t: t.get("sort_order", 0)
    )

    if not trials:
        print("[run_trials] WARNING: no trials found in spec; nothing to run.")
        return

    # Validate that all generated configs exist before starting any run.
    missing: list[str] = []
    for trial in trials:
        cfg_path = configs_dir / f"tuner_{trial['trial_id']}.yml"
        if not cfg_path.exists():
            missing.append(str(cfg_path.relative_to(Path.cwd())))

    if missing:
        print(
            "[run_trials] ERROR: the following generated config files are missing:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print(
            "\nRun generate_trials.py first:\n"
            f"  python src/autoslo/experiments/generate_trials.py {args.spec}",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(trials)
    for idx, trial in enumerate(trials, 1):
        tid = trial["trial_id"]
        cfg_path = configs_dir / f"tuner_{tid}.yml"
        cmd = [
            sys.executable,
            "src/autoslo/tuner/policy_tuner.py",
            str(cfg_path),
        ]
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
