import argparse
import sys
from pathlib import Path

from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.utils import parse_params
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.tuner.policy_tuner import AlreadyCompleteError, PolicyTuner

console = Console()


def main():
    # Argument parsing.
    description = "Run the policy tuner from a YAML config file."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--initial_execution_config_path",
        help="Path to the YAML execution config file.",
    )
    parser.add_argument(
        "--tuner_config_path",
        help="Path to the YAML tuner config file.",
    )
    parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Substitute <KEY> placeholder in the config with VALUE. "
            "May be repeated: --param TARGET_DATE=2024-05-27."
        ),
    )
    parser.add_argument(
        "--tuning_manifest_path",
        help=(
            "Path to a YAML file describing the tuner run. Supersedes "
            "--initial_execution_config_path, --tuner_config_path, and "
            "--param in favor of a single manifest."
        ),
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
        "--verbose_progress",
        action="store_true",
        default=False,
        help=(
            "Show per-config and per-simulation sub-level progress bars in "
            "addition to the top-level bar. Defaults to False."
        ),
    )
    args = parser.parse_args()

    # Useful work.
    if args.tuning_manifest_path:
        manifest = load_yaml(args.tuning_manifest_path)

        for tuning_run_name, tuning_run_spec in manifest.get(
            "main_content", {}
        ).items():
            try:
                initial_execution_config_path = str(
                    resolve_config(
                        tuning_run_spec["configs"]["initial_execution_config"]
                    )
                )
                tuner_config_path = str(
                    resolve_config(tuning_run_spec["configs"]["tuner_config"])
                )

                pt = PolicyTuner(
                    initial_execution_config_path=initial_execution_config_path,
                    tuner_config_path=tuner_config_path,
                    force=args.force,
                    params=tuning_run_spec.get("params", {}),
                    run_id=tuning_run_name,
                    verbose_progress=args.verbose_progress,
                )
            except AlreadyCompleteError as exc:
                console.print(f"[dim]Skipping: {exc}[/]")
                continue
            pt.tune()
        return

    # If no manifest, run a single tuner with the provided config paths and
    # params.
    try:
        pt = PolicyTuner(
            initial_execution_config_path=args.initial_execution_config_path,
            tuner_config_path=args.tuner_config_path,
            force=args.force,
            params=parse_params(args.param),
            verbose_progress=args.verbose_progress,
        )
    except AlreadyCompleteError as exc:
        console.print(f"[dim]Skipping: {exc}[/]")
        sys.exit(0)
    pt.tune()


if __name__ == "__main__":
    main()
