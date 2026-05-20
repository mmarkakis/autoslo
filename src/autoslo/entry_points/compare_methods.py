from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _fmt(values: list[float]) -> str:
    """Format relative diffs as percentages: mean on top, dim min–max range below."""
    mean = sum(values) / len(values)
    color = "green" if mean < 0 else "red"
    result = f"[{color}]{mean:+.1f}%[/]"
    if len(values) >= 2:
        lo, hi = min(values), max(values)
        result += f"\n[dim]{lo:+.1f}% … {hi:+.1f}%[/dim]"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise a plotter-output CSV by comparing every method "
            "to a reference.  For each method, reports the mean difference "
            "in violation rate and cost across all panels where both methods appear."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the plotter CSV.")
    parser.add_argument(
        "reference", type=str, help="Label value of the reference method."
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        Console().print("[yellow]CSV is empty.[/]")
        return

    x_metric: str = rows[0]["x_metric"]

    # panel_key -> {label -> (x, y)}
    panels: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        panel_key = row["panel_title"] or f"({row['row']},{row['col']})"
        panels[panel_key][row["label"]] = (float(row["x"]), float(row["y"]))

    # Preserve label order as first seen in the CSV (excluding the reference).
    label_order: list[str] = []
    seen_labels: set[str] = set()
    for row in rows:
        lbl = row["label"]
        if lbl != args.reference and lbl not in seen_labels:
            label_order.append(lbl)
            seen_labels.add(lbl)

    x_diffs: dict[str, list[float]] = defaultdict(list)
    y_diffs: dict[str, list[float]] = defaultdict(list)

    for points in panels.values():
        if args.reference not in points:
            continue
        ref_x, ref_y = points[args.reference]
        for label, (x, y) in points.items():
            if label == args.reference:
                continue
            if ref_x != 0:
                x_diffs[label].append((x - ref_x) / ref_x * 100)
            if ref_y != 0:
                y_diffs[label].append((y - ref_y) / ref_y * 100)

    console = Console()
    if not x_diffs:
        console.print(
            f"[red]Reference method '{args.reference}' not found in any panel.[/]"
        )
        return

    table = Table(
        title=(
            f"vs. reference: [bold]{args.reference}[/]  "
            f"[dim]{args.csv_path.name}[/]"
        ),
        show_lines=True,
    )
    table.add_column("method", no_wrap=True)
    table.add_column(x_metric, justify="right")
    table.add_column("cost", justify="right")
    table.add_column("n panels", justify="right", style="dim")

    for label in label_order:
        n = len(x_diffs[label])
        table.add_row(label, _fmt(x_diffs[label]), _fmt(y_diffs[label]), str(n))

    console.print(table)


if __name__ == "__main__":
    main()
