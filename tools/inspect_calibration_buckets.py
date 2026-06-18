#!/usr/bin/env python3
"""Inspect calibration bucket statistics for one or more IconqModel artifacts.

Usage:
    python tools/inspect_calibration_buckets.py <model_id> [<model_id2> ...]

For each model, prints two rich tables:
  1. Per-(cluster_rpu, concurrency_bin) cell counts and residual quantiles
     (p25, p50, p75, p90).
  2. A hyperparameter sweep over shrinkage_k x min_bucket_count showing the
     effective calibration delta at p50 per (rpu, concurrency_bin) cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

# Ensure project src is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoslo.models.iconq_model import IconqModel
from autoslo.models.residual_calibrator import (
    CONCURRENCY_BINS,
    CONCURRENCY_LABELS,
    ResidualCalibrator,
    ResidualCalibratorConfig,
)

console = Console()


def _bin_representatives(bins: list[float]) -> list[int]:
    """Return the first integer in each bin interval (bins[i], bins[i+1]]."""
    import math
    return [math.floor(b) + 1 for b in bins[:-1]]

# ---------------------------------------------------------------------------
# Color / formatting helpers
# ---------------------------------------------------------------------------
# residual_ratio = predicted / actual
#   < 1  →  model underestimates  (actual is slower — dangerous)
#   = 1  →  well-calibrated
#   > 1  →  model overestimates   (predictions are pessimistic — safe)

_COLOR_LEGEND = (
    "[bold red]■[/bold red] strong under-est (<0.67×)  "
    "[yellow]■[/yellow] mild under-est (0.67×–0.91×)  "
    "[green]■[/green] well-calibrated (0.91×–1.1×)  "
    "[cyan]■[/cyan] mild over-est (1.1×–1.5×)  "
    "[bold blue]■[/bold blue] strong over-est (>1.5×)"
)

_SWEEP_COLOR_LEGEND = (
    "[bold red]■[/bold red] strong upward correction (factor <0.67, ×≥1.5 applied)  "
    "[yellow]■[/yellow] mild upward correction (0.67–0.91)  "
    "[green]■[/green] small correction (0.91–1.1)  "
    "[cyan]■[/cyan] mild downward correction (1.1–1.5)  "
    "[bold blue]■[/bold blue] strong downward correction (factor >1.5, ×≤0.67 applied)"
)


def _ratio_color(v: float) -> str:
    if 0.91 <= v <= 1.1:
        return "green"
    elif v < 0.67:
        return "bold red"
    elif v < 1.0:
        return "yellow"
    elif v > 1.5:
        return "bold blue"
    else:
        return "cyan"


def _fmt_ratio(v: float) -> str:
    color = _ratio_color(v)
    return f"[{color}]{v:.4f}×[/{color}]"


def _bucket_cell(arr: np.ndarray, min_n: int) -> str:
    """Format a multi-line cell showing n and per-quantile residual ratios."""
    n = len(arr)
    n_str = f"[dim]n={n}[/dim]" if n < min_n else f"[bold]n={n}[/bold]"
    p25, p50, p75, p90 = (float(np.quantile(arr, q)) for q in (0.25, 0.50, 0.75, 0.90))
    lines = [
        n_str,
        f"[dim]p25[/dim] {_fmt_ratio(p25)}",
        f"[dim]p50[/dim] {_fmt_ratio(p50)}",
        f"[dim]p75[/dim] {_fmt_ratio(p75)}",
        f"[dim]p90[/dim] {_fmt_ratio(p90)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_val_df(model_id: str) -> pd.DataFrame:
    model_dir = Path(IconqModel.default_save_dir(model_id))
    parquet_path = model_dir / "final_val.parquet"
    csv_path = model_dir / "final_val.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No final_val.parquet or final_val.csv found for model {model_id!r} "
            f"at {model_dir}."
        )


def _prepare_df(
    val_df: pd.DataFrame,
    bins: list[float] = CONCURRENCY_BINS,
    labels: list[str] = CONCURRENCY_LABELS,
) -> pd.DataFrame:
    """Filter to LSTM rows, exclude lower-bound targets, compute residual_ratio."""
    df = val_df.copy()
    if "model_source" in df.columns:
        df = df[df["model_source"] == "lstm"]
    if "target_is_lower_bound" in df.columns:
        df = df[~df["target_is_lower_bound"].astype(bool)]
    df = df[(df["y"] > 0) & (df["y_pred_mean"] > 0)].copy()
    df["residual_ratio"] = (
        df["y_pred_mean"].astype(float) / df["y"].astype(float)
    )
    df["conc_bin"] = pd.cut(
        df["num_other_concurrent_queries"],
        bins=bins,
        labels=labels,
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# Table 1: bucket statistics
# ---------------------------------------------------------------------------


def _print_bucket_stats(
    model_id: str,
    df: pd.DataFrame,
    labels: list[str] = CONCURRENCY_LABELS,
    min_n: int = 20,
) -> None:
    console.print(Rule(
        f"[bold]{model_id}[/] — bucket stats  "
        "[dim]ratio = predicted / actual[/dim]"
    ))
    console.print(_COLOR_LEGEND)

    all_rpus = sorted(df["rpu"].dropna().unique().astype(int))

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("rpu", justify="right", style="bold", no_wrap=True)
    for label in labels:
        table.add_column(f"[dim]conc[/dim]\n{label}", justify="center")

    for rpu in all_rpus:
        rpu_df = df[df["rpu"] == rpu]
        row_vals: list[str] = [str(rpu)]
        for label in labels:
            cell = rpu_df[rpu_df["conc_bin"] == label]["residual_ratio"]
            n = len(cell)
            if n == 0:
                row_vals.append("[dim]n=0[/dim]")
            else:
                row_vals.append(_bucket_cell(cell.to_numpy(dtype=float), min_n))
        table.add_row(*row_vals)

    console.print(table)


# ---------------------------------------------------------------------------
# Table 2: shrinkage hyperparameter sweep
# ---------------------------------------------------------------------------

_SWEEP_K_VALUES = [20, 50, 100]
_SWEEP_MIN_BUCKET_VALUES = [10, 20, 30]


def _sweep_cell(calibrators: dict, rpu: int, conc: int) -> str:
    """Build a mini 3×3 grid (rows=min_n, cols=k) for one (rpu, conc_bin) cell."""
    # Header: k values
    header = "[dim]     " + "  ".join(f"k={k:<3}" for k in _SWEEP_K_VALUES) + "[/dim]"
    lines = [header]
    for min_b in _SWEEP_MIN_BUCKET_VALUES:
        cells = "  ".join(
            _fmt_ratio(calibrators[(k, min_b)].lookup(rpu, conc, 0.50))
            for k in _SWEEP_K_VALUES
        )
        lines.append(f"[dim]n={min_b}[/dim]  {cells}")
    return "\n".join(lines)


def _print_shrinkage_sweep(
    model_id: str,
    val_df: pd.DataFrame,
    bins: list[float] = CONCURRENCY_BINS,
    labels: list[str] = CONCURRENCY_LABELS,
) -> None:
    console.print(Rule(f"[bold]{model_id}[/] — shrinkage sweep (correction factor at p50)"))
    console.print(_SWEEP_COLOR_LEGEND)
    console.print(
        "[dim]Each cell: rows = min_bucket_count ∈ "
        f"{_SWEEP_MIN_BUCKET_VALUES}, "
        f"cols = shrinkage_k ∈ {_SWEEP_K_VALUES}[/dim]"
    )

    representatives = _bin_representatives(bins)
    prepared_df = _prepare_df(val_df, bins, labels)
    all_rpus = sorted(prepared_df["rpu"].dropna().unique().astype(int))

    # Pre-fit all 9 calibrators up front.
    calibrators: dict[tuple[int, int], ResidualCalibrator] = {}
    for k in _SWEEP_K_VALUES:
        for min_b in _SWEEP_MIN_BUCKET_VALUES:
            config = ResidualCalibratorConfig(
                shrinkage_k=k, min_bucket_count=min_b,
                concurrency_bins=bins, concurrency_labels=labels,
            )
            cal = ResidualCalibrator(config=config)
            cal.fit(val_df)
            calibrators[(k, min_b)] = cal

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("rpu", justify="right", style="bold", no_wrap=True)
    for label in labels:
        table.add_column(f"[dim]conc[/dim]\n{label}", justify="center")

    for rpu in all_rpus:
        row_vals: list[str] = [str(rpu)]
        for conc in representatives:
            row_vals.append(_sweep_cell(calibrators, rpu, conc))
        table.add_row(*row_vals)

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model_ids",
        nargs="+",
        metavar="MODEL_ID",
        help="One or more IconqModel IDs to inspect.",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip the hyperparameter sweep table (faster).",
    )
    parser.add_argument(
        "--bins",
        nargs="+",
        type=float,
        metavar="EDGE",
        default=None,
        help=(
            "Custom concurrency bin edges (interior edges only, without the "
            "implicit -inf/+inf). Example: --bins 0.5 10.5 50.5 100.5 "
            "produces bins (-inf,0], (0,10], (10,50], (50,100], (100,+inf). "
            "Labels are auto-generated from adjacent edges."
        ),
    )
    return parser.parse_args()


def _edges_to_bins_labels(
    interior_edges: list[float],
) -> tuple[list[float], list[str]]:
    """Convert interior edge values to (bins, labels) suitable for pd.cut.

    Interior edges are the split points between bins, without the outer
    sentinels. For example, [0.5, 10.5, 50.5] produces:
        bins   = [-0.5, 0.5, 10.5, 50.5, inf]
        labels = ['0', '1-10', '11-50', '51+']
    """
    edges = sorted(interior_edges)
    bins: list[float] = [-0.5] + edges + [float("inf")]
    labels: list[str] = []
    for i, edge in enumerate(edges):
        lo = 0 if i == 0 else int(edges[i - 1]) + 1
        hi = int(edge)
        labels.append(str(lo) if lo == hi else f"{lo}-{hi}")
    lo_last = int(edges[-1]) + 1
    labels.append(f"{lo_last}+")
    return bins, labels


def main() -> None:
    args = _parse_args()

    if args.bins is not None:
        bins, labels = _edges_to_bins_labels(args.bins)
        console.print(
            f"[dim]Using custom bins:[/dim] {bins}\n"
            f"[dim]Labels:[/dim] {labels}"
        )
    else:
        bins, labels = CONCURRENCY_BINS, CONCURRENCY_LABELS

    for model_id in args.model_ids:
        try:
            val_df = _load_val_df(model_id)
        except FileNotFoundError as exc:
            console.print(f"[red]Error:[/] {exc}")
            continue

        prepared_df = _prepare_df(val_df, bins, labels)
        _print_bucket_stats(model_id, prepared_df, labels, min_n=20)
        if not args.no_sweep:
            _print_shrinkage_sweep(model_id, val_df, bins, labels)


if __name__ == "__main__":
    main()
