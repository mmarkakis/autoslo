#!/usr/bin/env python3
"""cost_savings_at_threshold.py

For a given focus series and SLO threshold, show how much cheaper the focus
series is compared to the cheapest *other* series that *also* meets the
threshold.

Each cell is filled as follows:
  VIOLATES     — the focus series' x value exceeds the threshold; the cell
                 is excluded from all averages.
  no baseline  — the focus series meets the threshold but no other series in
                 that cell does; excluded from averages.
  -24.0%       — the focus series is 24% cheaper than the cheapest other
                 series that also meets the threshold (negative = focus wins).

Row means, column means, and an overall mean are computed over non-excluded
cells only.

Usage:
    python tools/cost_savings_at_threshold.py data/plots/main_eval_v8/main_eval_v8#live.csv
    python tools/cost_savings_at_threshold.py path/to/plot.csv --focus AutoSLO --threshold 0.10
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

_VIOLATES = "VIOLATES"
_NO_BASELINE = "no_baseline"


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


def _fmt_pct(val: float | None, mean_style: bool = False) -> Text:
    if val is None:
        return Text("N/A", style="bold italic dim" if mean_style else "dim")
    color = "green" if val < 0 else ("red" if val > 0 else "white")
    style = f"bold italic {color}" if mean_style else color
    return Text(f"{val:+.1f}%", style=style)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Show cost savings of a focus series vs the cheapest compliant "
            "baseline, per panel."
        )
    )
    parser.add_argument("csv_path", help="Path to the plot CSV file.")
    parser.add_argument(
        "--focus",
        default="AutoSLO",
        help="Focus series label (default: AutoSLO)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help=(
            "SLO threshold: x must be ≤ this value to be considered "
            "compliant (default: 0.10)"
        ),
    )
    args = parser.parse_args()

    # ── Read CSV ──────────────────────────────────────────────────────────────
    cells: dict[tuple[int, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    panel_titles: dict[tuple[int, int], str] = {}

    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["row"]) if row["row"] else 0
            c = int(row["col"]) if row["col"] else 0
            cells[(r, c)][row["label"]] = (float(row["x"]), float(row["y"]))
            panel_titles[(r, c)] = row["panel_title"]

    if not cells:
        console.print("[red]No data found in CSV.[/]")
        sys.exit(1)

    all_rows = sorted({r for r, _ in cells})
    all_cols = sorted({c for _, c in cells})

    # ── Compute per-cell result ───────────────────────────────────────────────
    cell_result: dict[tuple[int, int], str | float] = {}

    for (r, c), methods in cells.items():
        focus = methods.get(args.focus)
        if focus is None:
            cell_result[(r, c)] = _VIOLATES
            continue

        focus_x, focus_y = focus
        if focus_x > args.threshold:
            cell_result[(r, c)] = _VIOLATES
            continue

        # Focus meets the threshold; find cheapest other series that also does.
        best_other_y: float | None = None
        for label, (x, y) in methods.items():
            if label == args.focus:
                continue
            if x <= args.threshold:
                if best_other_y is None or y < best_other_y:
                    best_other_y = y

        if best_other_y is None:
            cell_result[(r, c)] = _NO_BASELINE
        else:
            cell_result[(r, c)] = (focus_y - best_other_y) / best_other_y * 100

    # ── Row / column labels derived from panel titles ─────────────────────────
    col_headers = [
        _common_label(
            [panel_titles[(r, c)] for r in all_rows if (r, c) in panel_titles],
            fallback=f"col {c}",
        )
        for c in all_cols
    ]
    row_labels = [
        _common_label(
            [panel_titles[(r, c)] for c in all_cols if (r, c) in panel_titles],
            fallback=f"row {r}",
        )
        for r in all_rows
    ]

    # ── Pre-compute means (numeric cells only) ────────────────────────────────
    def _numeric(positions: list[tuple[int, int]]) -> list[float]:
        return [
            cell_result[pos]  # type: ignore[misc]
            for pos in positions
            if isinstance(cell_result.get(pos), float)
        ]

    row_means = [_mean(_numeric([(r, c) for c in all_cols])) for r in all_rows]
    col_means = [_mean(_numeric([(r, c) for r in all_rows])) for c in all_cols]
    overall = _mean(_numeric(list(cells.keys())))

    # ── Build table ───────────────────────────────────────────────────────────
    grid = Table(
        title=(
            f"{args.focus} cost vs cheapest compliant baseline"
            f"  (threshold x ≤ {args.threshold})"
        ),
        box=box.SIMPLE_HEAD,
        show_header=True,
        padding=(0, 2),
    )
    grid.add_column("", style="dim")
    for header in col_headers:
        grid.add_column(header, justify="center", min_width=14)
    grid.add_column("── Row mean ──", justify="center", min_width=14)

    for r, row_label, rm in zip(all_rows, row_labels, row_means):
        row_cells: list[Text] = []
        for c in all_cols:
            res = cell_result.get((r, c))
            if res == _VIOLATES:
                row_cells.append(Text("VIOLATES", style="dim"))
            elif res == _NO_BASELINE:
                row_cells.append(Text("no baseline", style="dim"))
            else:
                row_cells.append(_fmt_pct(res if isinstance(res, float) else None))
        row_cells.append(_fmt_pct(rm, mean_style=True))
        grid.add_row(row_label, *row_cells)

    grid.add_section()

    col_mean_cells = [_fmt_pct(cm, mean_style=True) for cm in col_means]
    col_mean_cells.append(_fmt_pct(overall, mean_style=True))
    grid.add_row(Text("── Col mean ──", style="dim italic"), *col_mean_cells)

    console.print(grid)
    console.print(
        Text(
            f"  (negative = {args.focus} is cheaper;  "
            "VIOLATES / no-baseline cells excluded from averages)",
            style="dim italic",
        )
    )


if __name__ == "__main__":
    main()
