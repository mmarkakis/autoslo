#!/usr/bin/env python3
"""compare_plot_csv_methods.py

Read a plot CSV (as produced by plot.py) and display a Rich grid showing
the percentage decrease in violation rate (x) and cost (y) when going from a
baseline method to a comparison method.

Row means appear as an extra demarcated column on the right; column means
appear as an extra demarcated row at the bottom.

Usage:
    python tools/compare_plot_csv_methods.py data/plots/main_eval_v8/main_eval_v8#live.csv
    python tools/compare_plot_csv_methods.py path/to/plot.csv --baseline RAIS --compare AutoSLO
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def _pct(val: float | None, bright: bool) -> Text:
    if val is None:
        return Text("N/A", style="dim")
    style = "white" if bright else "dim"
    return Text(f"{val:+.1f}%", style=style)


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _common_label(titles: list[str], fallback: str) -> str:
    """Return the comma-delimited parts that are common to all titles."""
    if not titles:
        return fallback
    parts_per_title = [
        [p.strip() for p in t.split(",") if p.strip()]
        for t in titles
    ]
    common_set = set(parts_per_title[0])
    for parts in parts_per_title[1:]:
        common_set &= set(parts)
    ordered = [p for p in parts_per_title[0] if p in common_set]
    return ", ".join(ordered) if ordered else fallback


def _cell(xd: float | None, yd: float | None, mean_style: bool = False) -> Text:
    base = "bold italic" if mean_style else ""
    x_style = (f"{base} white").strip() if not mean_style else "bold italic white"
    d_style = (f"{base} dim").strip() if not mean_style else "italic dim"
    t = Text()
    t.append(_pct(xd, bright=True).plain, style=x_style if xd is not None else "dim")
    t.append("  ")
    t.append(_pct(yd, bright=False).plain, style=d_style if yd is not None else "dim")
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two methods across a plot CSV grid."
    )
    parser.add_argument("csv_path", help="Path to the plot CSV file.")
    parser.add_argument(
        "--baseline",
        default="RAIS",
        help="Baseline method label (default: RAIS)",
    )
    parser.add_argument(
        "--compare",
        default="AutoSLO",
        help="Comparison method label (default: AutoSLO)",
    )
    args = parser.parse_args()

    # ── Read CSV ──────────────────────────────────────────────────────────────
    cells: dict[tuple[int, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    panel_titles: dict[tuple[int, int], str] = {}

    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r, c = int(row["row"]), int(row["col"])
            cells[(r, c)][row["label"]] = (float(row["x"]), float(row["y"]))
            panel_titles[(r, c)] = row["panel_title"]

    if not cells:
        console.print("[red]No data found in CSV.[/]")
        sys.exit(1)

    all_rows = sorted({r for r, _ in cells})
    all_cols = sorted({c for _, c in cells})

    # ── Compute deltas ────────────────────────────────────────────────────────
    x_deltas: dict[tuple[int, int], float | None] = {}
    y_deltas: dict[tuple[int, int], float | None] = {}

    for (r, c), methods in cells.items():
        base = methods.get(args.baseline)
        comp = methods.get(args.compare)
        if base is not None and comp is not None:
            bx, by = base
            cx, cy = comp
            x_deltas[(r, c)] = (cx - bx) / bx * 100 if bx != 0 else None
            y_deltas[(r, c)] = (cy - by) / by * 100 if by != 0 else None
        else:
            x_deltas[(r, c)] = None
            y_deltas[(r, c)] = None

    # ── Row / column labels (derived from panel_titles) ───────────────────────
    # Column headers: common part of panel titles sharing the same column.
    col_headers = [
        _common_label(
            [panel_titles[(r, c)] for r in all_rows if (r, c) in panel_titles],
            fallback=f"col {c}",
        )
        for c in all_cols
    ]
    # Row labels: common part of panel titles sharing the same row.
    row_labels = [
        _common_label(
            [panel_titles[(r, c)] for c in all_cols if (r, c) in panel_titles],
            fallback=f"row {r}",
        )
        for r in all_rows
    ]

    # ── Pre-compute means ─────────────────────────────────────────────────────
    row_x_means = [
        _mean([x_deltas[(r, c)] for c in all_cols if x_deltas.get((r, c)) is not None])
        for r in all_rows
    ]
    row_y_means = [
        _mean([y_deltas[(r, c)] for c in all_cols if y_deltas.get((r, c)) is not None])
        for r in all_rows
    ]
    col_x_means = [
        _mean([x_deltas[(r, c)] for r in all_rows if x_deltas.get((r, c)) is not None])
        for c in all_cols
    ]
    col_y_means = [
        _mean([y_deltas[(r, c)] for r in all_rows if y_deltas.get((r, c)) is not None])
        for c in all_cols
    ]
    all_x = [v for v in x_deltas.values() if v is not None]
    all_y = [v for v in y_deltas.values() if v is not None]
    overall_x = _mean(all_x)
    overall_y = _mean(all_y)

    # ── Main grid table ───────────────────────────────────────────────────────
    grid = Table(
        title=f"{args.compare} vs {args.baseline}  (negative = {args.compare} is better)",
        box=box.SIMPLE_HEAD,
        show_header=True,
        padding=(0, 2),
    )
    grid.add_column("", style="dim")
    for header in col_headers:
        grid.add_column(header, justify="center", min_width=12)
    grid.add_column("── Row mean ──", justify="center", min_width=12)

    for r, row_label, rxm, rym in zip(all_rows, row_labels, row_x_means, row_y_means):
        row_cells: list[Text] = []
        for c in all_cols:
            row_cells.append(_cell(x_deltas.get((r, c)), y_deltas.get((r, c))))
        row_cells.append(_cell(rxm, rym, mean_style=True))
        grid.add_row(row_label, *row_cells)

    # Separator before the means row.
    grid.add_section()

    col_mean_cells: list[Text] = []
    for cxm, cym in zip(col_x_means, col_y_means):
        col_mean_cells.append(_cell(cxm, cym, mean_style=True))
    col_mean_cells.append(_cell(overall_x, overall_y, mean_style=True))
    grid.add_row(
        Text("── Col mean ──", style="dim italic"),
        *col_mean_cells,
    )

    console.print(grid)
    console.print(
        Text(
            f"  (white = VR Δ, dim = Cost Δ;  {args.baseline}→{args.compare})",
            style="dim italic",
        )
    )


if __name__ == "__main__":
    main()
