import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from autoslo.config.utils import parse_params
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.tuner.policy_tuner import AlreadyCompleteError, PolicyTuner

console = Console()


def _print_preflight_table(rows: list[tuple[str, str, str]]) -> None:
    table = Table(title="Tune Preflight Summary")
    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Details")

    for run_name, action, details in rows:
        action_style = "green" if action == "Run" else "yellow"
        table.add_row(run_name, f"[{action_style}]{action}[/]", details)

    console.print(table)


def _confirm_execution(num_to_run: int, num_to_skip: int) -> bool:
    console.print(
        f"Planned actions: [green]{num_to_run} to run[/], "
        f"[yellow]{num_to_skip} to skip[/]."
    )
    return Confirm.ask("Proceed?", default=True)


def _print_run_header(
    run_name: str, details: str, run_index: int, total_runs: int
) -> None:
    console.print()
    console.print(
        Panel.fit(
            (
                f"[bold]Overall progress:[/] {run_index}/{total_runs}\n"
                f"[bold]Upcoming run:[/] {run_name}\n"
                f"[bold]Details:[/] {details}"
            ),
            title="Starting Tune Run",
            border_style="cyan",
        )
    )
    console.print()


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
        preflight_rows: list[tuple[str, str, str]] = []
        planned_runs: list[tuple[str, str, PolicyTuner]] = []

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
                preflight_rows.append(
                    (
                        tuning_run_name,
                        "Run",
                        (
                            f"execution={initial_execution_config_path}, "
                            f"tuner={tuner_config_path}"
                        ),
                    )
                )
                planned_runs.append(
                    (
                        tuning_run_name,
                        (
                            f"execution={initial_execution_config_path}, "
                            f"tuner={tuner_config_path}"
                        ),
                        pt,
                    )
                )
            except AlreadyCompleteError as exc:
                preflight_rows.append((tuning_run_name, "Skip", str(exc)))
                continue

        _print_preflight_table(preflight_rows)

        num_to_run = sum(1 for _, action, _ in preflight_rows if action == "Run")
        num_to_skip = len(preflight_rows) - num_to_run

        if num_to_run == 0:
            console.print("[dim]No pending runs. Nothing to do.[/]")
            return

        if not _confirm_execution(num_to_run=num_to_run, num_to_skip=num_to_skip):
            console.print("[yellow]Cancelled by user.[/]")
            return

        for run_index, (run_name, details, pt) in enumerate(
            planned_runs, start=1
        ):
            _print_run_header(
                run_name=run_name,
                details=details,
                run_index=run_index,
                total_runs=num_to_run,
            )
            pt.tune()
        return

    # If no manifest, run a single tuner with the provided config paths and
    # params.
    preflight_rows: list[tuple[str, str, str]] = []
    try:
        pt = PolicyTuner(
            initial_execution_config_path=args.initial_execution_config_path,
            tuner_config_path=args.tuner_config_path,
            force=args.force,
            params=parse_params(args.param),
            verbose_progress=args.verbose_progress,
        )
        single_run_details = (
            "execution="
            f"{args.initial_execution_config_path}, "
            f"tuner={args.tuner_config_path}"
        )
        preflight_rows.append(
            (
                "single-run",
                "Run",
                single_run_details,
            )
        )
    except AlreadyCompleteError as exc:
        preflight_rows.append(("single-run", "Skip", str(exc)))

    _print_preflight_table(preflight_rows)

    num_to_run = sum(1 for _, action, _ in preflight_rows if action == "Run")
    num_to_skip = len(preflight_rows) - num_to_run

    if num_to_run == 0:
        console.print("[dim]No pending runs. Nothing to do.[/]")
        sys.exit(0)

    if not _confirm_execution(num_to_run=num_to_run, num_to_skip=num_to_skip):
        console.print("[yellow]Cancelled by user.[/]")
        sys.exit(0)

    _print_run_header(
        run_name="single-run",
        details=single_run_details,
        run_index=1,
        total_runs=num_to_run,
    )
    pt.tune()


if __name__ == "__main__":
    main()
