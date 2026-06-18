#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class RunTiming:
    run_name: str
    durations: dict[str, float]
    total_seconds: float | None
    published_at: datetime


def format_seconds(seconds: float) -> str:
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = (seconds % 3600) / 60
        return f"{hours}h {minutes:.1f}m"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    if seconds >= 1:
        return f"{seconds:.2f}s"
    return f"{seconds:.3f}s"


def load_run_timing(report_path: Path) -> tuple[RunTiming, list[str]]:
    durations: dict[str, float] = {}
    phase_order: list[str] = []
    total_seconds: float | None = None

    with report_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            phase_key = row["phase_key"].strip()
            elapsed_s = float(row["elapsed_s"])

            if phase_key == "TOTAL":
                total_seconds = elapsed_s
                continue

            if phase_key not in durations:
                phase_order.append(phase_key)
            durations[phase_key] = elapsed_s

    return (
        RunTiming(
            run_name=report_path.parent.name,
            durations=durations,
            total_seconds=total_seconds,
            published_at=datetime.fromtimestamp(report_path.stat().st_mtime),
        ),
        phase_order,
    )


def format_published_at(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def discover_timing_reports(
    runs_dir: Path,
) -> tuple[list[RunTiming], list[str], int]:
    run_timings: list[RunTiming] = []
    phase_order: list[str] = []
    missing_count = 0

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        report_path = run_dir / "timing_report.csv"
        if not report_path.exists():
            missing_count += 1
            continue

        run_timing, per_run_phase_order = load_run_timing(report_path)
        run_timings.append(run_timing)
        for phase_key in per_run_phase_order:
            if phase_key not in phase_order:
                phase_order.append(phase_key)

    return run_timings, phase_order, missing_count


def value_style(value: float | None, phase_min: float, phase_max: float) -> str:
    if value is None:
        return "dim"
    if phase_min == phase_max:
        return ""
    if value == phase_min:
        return "green"
    if value == phase_max:
        return "red"
    return ""


def add_main_table(
    console: Console, run_timings: list[RunTiming], phases: list[str]
) -> None:
    table = Table(title="Tuner Timing Summary")
    table.add_column("run", style="cyan", no_wrap=True)
    table.add_column("published", style="magenta", no_wrap=True)
    for phase in phases:
        table.add_column(phase, justify="right")
    table.add_column("TOTAL", justify="right", style="bold")

    phase_min_max: dict[str, tuple[float, float]] = {}
    for phase in phases:
        values = [
            rt.durations[phase] for rt in run_timings if phase in rt.durations
        ]
        if values:
            phase_min_max[phase] = (min(values), max(values))

    total_values = [
        rt.total_seconds for rt in run_timings if rt.total_seconds is not None
    ]
    total_min = min(total_values) if total_values else 0.0
    total_max = max(total_values) if total_values else 0.0

    for rt in run_timings:
        row: list[str] = [rt.run_name, format_published_at(rt.published_at)]
        for phase in phases:
            value = rt.durations.get(phase)
            if value is None:
                row.append("[dim]-[/]")
                continue

            phase_min, phase_max = phase_min_max[phase]
            style = value_style(value, phase_min, phase_max)
            rendered = format_seconds(value)
            row.append(f"[{style}]{rendered}[/]" if style else rendered)

        total_value = rt.total_seconds
        total_style = value_style(total_value, total_min, total_max)
        if total_value is None:
            row.append("[dim]-[/]")
        else:
            rendered_total = format_seconds(total_value)
            row.append(
                f"[{total_style}]{rendered_total}[/]"
                if total_style
                else rendered_total
            )

        table.add_row(*row)

    console.print(table)


def add_stats_table(
    console: Console, run_timings: list[RunTiming], phases: list[str]
) -> None:
    table = Table(title="Per-Phase Aggregate Stats")
    table.add_column("phase", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("min", justify="right")
    table.add_column("median", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("max", justify="right")
    table.add_column("range", justify="right")

    def add_stat_row(label: str, values: list[float]) -> None:
        if not values:
            table.add_row(label, "0", "-", "-", "-", "-", "-")
            return

        v_min = min(values)
        v_med = statistics.median(values)
        v_mean = statistics.mean(values)
        v_max = max(values)
        v_range = v_max - v_min

        table.add_row(
            label,
            str(len(values)),
            format_seconds(v_min),
            format_seconds(v_med),
            format_seconds(v_mean),
            format_seconds(v_max),
            format_seconds(v_range),
        )

    for phase in phases:
        add_stat_row(
            phase,
            [
                rt.durations[phase]
                for rt in run_timings
                if phase in rt.durations
            ],
        )

    add_stat_row(
        "TOTAL",
        [
            rt.total_seconds
            for rt in run_timings
            if rt.total_seconds is not None
        ],
    )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize timing_report.csv files under data/tuner_runs with Rich."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("data/tuner_runs"),
        help="Directory containing tuner run subdirectories.",
    )
    parser.add_argument(
        "--sort-by",
        default="name",
        help=(
            "Sort rows by column name (phase key, TOTAL, or name). "
            "Default: name"
        ),
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    args = parser.parse_args()

    console = Console()
    runs_dir = args.runs_dir

    if not runs_dir.exists() or not runs_dir.is_dir():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    run_timings, phases, missing_count = discover_timing_reports(runs_dir)
    if not run_timings:
        raise SystemExit(f"No timing_report.csv files found under: {runs_dir}")

    sort_key = args.sort_by
    sort_ascending = args.ascending or sort_key == "name"

    if sort_key == "name":
        run_timings.sort(key=lambda rt: rt.run_name, reverse=not sort_ascending)
    elif sort_key == "TOTAL":
        run_timings.sort(
            key=lambda rt: (
                rt.total_seconds if rt.total_seconds is not None else -1
            ),
            reverse=not sort_ascending,
        )
    else:
        run_timings.sort(
            key=lambda rt: rt.durations.get(sort_key, -1),
            reverse=not sort_ascending,
        )

    console.print(
        Panel.fit(
            (
                f"runs_dir: [bold]{runs_dir}[/]\n"
                f"run dirs with report: [bold]{len(run_timings)}[/]\n"
                f"run dirs missing report: [bold]{missing_count}[/]\n"
                "coloring: [green]fastest[/] / [red]slowest[/] per column"
            ),
            title="Timing Report Discovery",
            border_style="cyan",
        )
    )

    add_main_table(console, run_timings, phases)
    add_stats_table(console, run_timings, phases)


if __name__ == "__main__":
    main()
