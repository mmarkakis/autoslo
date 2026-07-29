import argparse
import csv
import sys
from pathlib import Path

import autoslo.filesystem.path_utils as pu
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


def _fmt_duration(elapsed_s: float) -> str:
    """Format seconds as a compact human-readable string (e.g. '1m 15s')."""
    total = int(elapsed_s)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{elapsed_s:.1f}s"


def _timing_cell(elapsed_s: float, total_s: float) -> str:
    """White duration + light-gray '- XX%' in a single Rich markup string."""
    pct = f"{elapsed_s / total_s * 100:.0f}%" if total_s > 0 else "–"
    return f"[white]{_fmt_duration(elapsed_s)}[/] [dim]- {pct}[/]"


def _load_timing_rows(out_dir: Path) -> list[dict] | None:
    """Read timing_report.csv from a tuner run directory (excluding TOTAL row)."""
    csv_path = out_dir / "timing_report.csv"
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["phase_key"] != "TOTAL"]
    return rows or None


# (run_name, preflight_action, out_dir, completed)
_PostflightEntry = tuple[str, str, Path, bool]


def _print_postflight_table(entries: list[_PostflightEntry]) -> None:
    """Print a per-run timing breakdown after all runs finish (or are skipped)."""
    # Load timing CSV for each run.
    timing: dict[str, list[dict] | None] = {
        name: _load_timing_rows(out_dir) for name, _, out_dir, _ in entries
    }

    # Collect unique phases in the order they first appear across all runs.
    phase_keys: list[str] = []
    phase_names: dict[str, str] = {}
    seen: set[str] = set()
    for name, _, _, _ in entries:
        for row in timing.get(name) or []:
            k = row["phase_key"]
            if k not in seen:
                phase_keys.append(k)
                phase_names[k] = row["phase_name"]
                seen.add(k)

    table = Table(title="Tune Postflight Summary")
    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    for k in phase_keys:
        table.add_column(phase_names[k], justify="right")
    table.add_column("Total", justify="right")

    for name, preflight_action, _, completed in entries:
        if preflight_action == "Skip":
            action_cell = "[yellow]Pre-existing[/]"
        elif completed:
            action_cell = "[green]Re-run[/]"
        else:
            action_cell = "[red]Missing[/]"

        rows = timing.get(name)
        if rows:
            total_s = sum(float(r["elapsed_s"]) for r in rows)
            by_key = {r["phase_key"]: r for r in rows}
            phase_cells = [
                (
                    _timing_cell(float(by_key[k]["elapsed_s"]), total_s)
                    if k in by_key
                    else "[dim]–[/]"
                )
                for k in phase_keys
            ]
            total_cell = f"[white]{_fmt_duration(total_s)}[/]"
        else:
            phase_cells = ["[dim]–[/]"] * len(phase_keys)
            total_cell = "[dim]–[/]"

        table.add_row(name, action_cell, *phase_cells, total_cell)

    console.print()
    console.print(table)

    # --- Action-grouped summary -------------------------------------------
    # Derive the action label for each entry (same logic as above).
    def _action_label(preflight_action: str, completed: bool) -> str:
        if preflight_action == "Skip":
            return "Pre-existing"
        return "Re-run" if completed else "Missing"

    # Preserve first-seen label order.
    label_order: list[str] = []
    seen_labels: set[str] = set()
    for _, pa, _, comp in entries:
        lbl = _action_label(pa, comp)
        if lbl not in seen_labels:
            label_order.append(lbl)
            seen_labels.add(lbl)

    # Accumulate totals per (label, phase_key).
    group_counts: dict[str, int] = {lbl: 0 for lbl in label_order}
    group_phase_s: dict[str, dict[str, float]] = {
        lbl: {k: 0.0 for k in phase_keys} for lbl in label_order
    }
    group_total_s: dict[str, float] = {lbl: 0.0 for lbl in label_order}

    for name, pa, _, comp in entries:
        lbl = _action_label(pa, comp)
        group_counts[lbl] += 1
        rows = timing.get(name)
        if rows:
            run_total = sum(float(r["elapsed_s"]) for r in rows)
            group_total_s[lbl] += run_total
            for r in rows:
                k = r["phase_key"]
                if k in group_phase_s[lbl]:
                    group_phase_s[lbl][k] += float(r["elapsed_s"])

    _LABEL_STYLE = {
        "Re-run": "green",
        "Pre-existing": "yellow",
        "Missing": "red",
    }

    summary = Table(title="Tune Postflight — Action Summary")
    summary.add_column("Action", no_wrap=True)
    summary.add_column("Count", justify="right")
    for k in phase_keys:
        summary.add_column(phase_names[k], justify="right")
    summary.add_column("Total", justify="right")

    for lbl in label_order:
        style = _LABEL_STYLE.get(lbl, "")
        action_cell = f"[{style}]{lbl}[/]" if style else lbl
        count_cell = str(group_counts[lbl])
        grp_total = group_total_s[lbl]
        if grp_total > 0:
            phase_cells = [
                _timing_cell(group_phase_s[lbl][k], grp_total)
                for k in phase_keys
            ]
            total_cell = f"[white]{_fmt_duration(grp_total)}[/]"
        else:
            phase_cells = ["[dim]–[/]"] * len(phase_keys)
            total_cell = "[dim]–[/]"
        summary.add_row(action_cell, count_cell, *phase_cells, total_cell)

    console.print(summary)


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
        postflight_entries: list[_PostflightEntry] = []
        tuner_runs_dir = pu.get_data_path() / "tuner_runs"

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
                postflight_entries.append(
                    (
                        tuning_run_name,
                        "Run",
                        tuner_runs_dir / tuning_run_name,
                        False,
                    )
                )
            except AlreadyCompleteError as exc:
                preflight_rows.append((tuning_run_name, "Skip", str(exc)))
                postflight_entries.append(
                    (
                        tuning_run_name,
                        "Skip",
                        tuner_runs_dir / tuning_run_name,
                        True,
                    )
                )
                continue

        _print_preflight_table(preflight_rows)

        num_to_run = sum(
            1 for _, action, _ in preflight_rows if action == "Run"
        )
        num_to_skip = len(preflight_rows) - num_to_run

        if num_to_run == 0:
            console.print("[dim]No pending runs. Nothing to do.[/]")
            return

        if not _confirm_execution(
            num_to_run=num_to_run, num_to_skip=num_to_skip
        ):
            console.print("[yellow]Cancelled by user.[/]")
            _print_postflight_table(postflight_entries)
            return

        completed_run_names: set[str] = set()
        for run_index, (run_name, details, pt) in enumerate(
            planned_runs, start=1
        ):
            _print_run_header(
                run_name=run_name,
                details=details,
                run_index=run_index,
                total_runs=num_to_run,
            )
            pt.tune(
                force=args.force,
            )
            completed_run_names.add(run_name)

        postflight_entries = [
            (n, a, d, True if a == "Skip" else n in completed_run_names)
            for n, a, d, _ in postflight_entries
        ]
        _print_postflight_table(postflight_entries)
        return

    # If no manifest, run a single tuner with the provided config paths and
    # params.
    preflight_rows: list[tuple[str, str, str]] = []
    single_pt: PolicyTuner | None = None
    try:
        single_pt = PolicyTuner(
            initial_execution_config_path=args.initial_execution_config_path,
            tuner_config_path=args.tuner_config_path,
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
        assert single_pt is not None
        _print_postflight_table(
            [("single-run", "Run", single_pt.out_dir, False)]
        )
        sys.exit(0)

    _print_run_header(
        run_name="single-run",
        details=single_run_details,
        run_index=1,
        total_runs=num_to_run,
    )
    assert single_pt is not None
    single_pt.tune(
        force=args.force,
    )
    _print_postflight_table([("single-run", "Run", single_pt.out_dir, True)])


if __name__ == "__main__":
    main()
