"""
Compare a directory of sampled/forecasted workload parquet files against a
single reference workload file.

Useful for diagnosing why a forecasting policy produces training scenarios that
differ from the oracle (target-day) workload, which in turn can cause the
policy tuner to optimize for the wrong regime.

Statistics reported:
  • Summary table   — query count, workload duration, mean/p50/p90/p99 IAT,
                      unique templates, hourly-profile total-variation distance,
                      template Jensen-Shannon divergence.
  • Hourly profile  — per-hour fraction of queries, directory mean ± std vs
                      reference, delta highlighted in red/green.
  • Template table  — top-N templates by reference frequency, with directory
                      mean ± std and signed delta.

Usage
-----
  python tools/compare_workloads.py <workload_dir> <reference_file> [options]

Examples
--------
  python tools/compare_workloads.py \\
      data/tuner_runs/may27_1month_k2_thresh1_nocache_v6/02_workloads/train \\
      data/tuner_runs/may27_oracle_k2_thresh1_nocache_v6/02_workloads/train/t_0.parquet

  python tools/compare_workloads.py \\
      data/tuner_runs/may27_1day_k2_thresh1_nocache_v6/02_workloads/train \\
      data/tuner_runs/may27_oracle_k2_thresh1_nocache_v6/02_workloads/train/t_0.parquet \\
      --top-templates 20 --show-per-scenario
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# ── helpers ──────────────────────────────────────────────────────────────────


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["template_id"] = df["query_text_id"].str.split("#").str[1]
    df["hour"] = df["abs_start_time"].dt.hour
    return df


def _load_dir(directory: Path) -> list[tuple[str, pd.DataFrame]]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        console.print(f"[red]No .parquet files found in {directory}[/]")
        sys.exit(1)
    return [(f.stem, _load(f)) for f in files]


def _iat_stats(df: pd.DataFrame) -> dict[str, float]:
    """Inter-arrival time statistics from rel_start_time_s."""
    iat = df["rel_start_time_s"].sort_values().diff().dropna()
    if iat.empty:
        return dict(mean=0.0, p50=0.0, p90=0.0, p99=0.0)
    return dict(
        mean=float(iat.mean()),
        p50=float(iat.quantile(0.50)),
        p90=float(iat.quantile(0.90)),
        p99=float(iat.quantile(0.99)),
    )


def _hourly_fracs(df: pd.DataFrame) -> np.ndarray:
    """Length-24 array of per-hour query fractions (sums to 1)."""
    counts = df.groupby("hour").size().reindex(range(24), fill_value=0).values.astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _template_fracs(df: pd.DataFrame) -> pd.Series:
    """Series: template_id → fraction of total queries."""
    return df["template_id"].value_counts(normalize=True)


def _total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _jsd(p: pd.Series, q: pd.Series) -> float:
    """Jensen-Shannon divergence between two frequency distributions."""
    all_keys = p.index.union(q.index)
    p_arr = p.reindex(all_keys, fill_value=0.0).values
    q_arr = q.reindex(all_keys, fill_value=0.0).values
    # Ensure they are proper probability distributions.
    p_arr = p_arr / p_arr.sum() if p_arr.sum() > 0 else p_arr
    q_arr = q_arr / q_arr.sum() if q_arr.sum() > 0 else q_arr
    m = 0.5 * (p_arr + q_arr)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return float(0.5 * (_kl(p_arr, m) + _kl(q_arr, m)))


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


def _styled(text: str, style: str) -> str:
    """Wrap text in a rich markup style tag, or return plain text if no style."""
    if not style:
        return text
    return f"[{style}]{text}[/]"


def _delta_style(delta: float, threshold: float = 0.01) -> str:
    if delta > threshold:
        return "red"
    if delta < -threshold:
        return "green"
    return ""


# ── per-scenario stats ────────────────────────────────────────────────────────


def _scenario_stats(df: pd.DataFrame) -> dict:
    iat = _iat_stats(df)
    return dict(
        n_queries=len(df),
        duration_s=float(df["rel_start_time_s"].max() - df["rel_start_time_s"].min()),
        n_unique_templates=int(df["template_id"].nunique()),
        peak_hour_count=int(df.groupby("hour").size().max()),
        iat_mean=iat["mean"],
        iat_p50=iat["p50"],
        iat_p90=iat["p90"],
        iat_p99=iat["p99"],
    )


# ── printing ──────────────────────────────────────────────────────────────────


def _print_summary(
    dir_dfs: list[tuple[str, pd.DataFrame]],
    ref_df: pd.DataFrame,
    ref_label: str,
    dir_label: str,
) -> None:
    dfs = [df for _, df in dir_dfs]
    all_stats = [_scenario_stats(df) for df in dfs]

    keys = ["n_queries", "duration_s", "iat_mean", "iat_p50", "iat_p90", "iat_p99",
            "n_unique_templates", "peak_hour_count"]

    means = {k: float(np.mean([s[k] for s in all_stats])) for k in keys}
    stds = {k: float(np.std([s[k] for s in all_stats])) for k in keys}
    ref_stats = _scenario_stats(ref_df)

    # Scalar distances.
    dir_hourly_mean = np.mean(
        np.stack([_hourly_fracs(df) for df in dfs]), axis=0
    )
    ref_hourly = _hourly_fracs(ref_df)
    tv_dist = _total_variation(dir_hourly_mean, ref_hourly)

    dir_template_mean = (
        pd.concat([_template_fracs(df) for df in dfs], axis=1)
        .fillna(0.0)
        .mean(axis=1)
    )
    ref_template = _template_fracs(ref_df)
    jsd = _jsd(dir_template_mean, ref_template)

    table = Table(title="Workload Summary Comparison", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column(f"{dir_label} mean", justify="right")
    table.add_column(f"{dir_label} std", justify="right", style="dim")
    table.add_column(f"{ref_label}", justify="right")
    table.add_column("Δ (mean − ref)", justify="right")

    def _row(label: str, key: str, fmt: int = 2) -> None:
        m = means[key]
        s = stds[key]
        r = float(ref_stats[key])
        delta = m - r
        pct = (delta / r * 100) if r != 0 else 0.0
        style = _delta_style(abs(pct) / 100, threshold=0.10)
        sign = "+" if delta >= 0 else ""
        table.add_row(
            label,
            _fmt(m, fmt),
            _fmt(s, fmt),
            _fmt(r, fmt),
            _styled(f"{sign}{_fmt(delta, fmt)} ({sign}{pct:.1f}%)", style),
        )

    _row("Total queries", "n_queries", 0)
    _row("Duration (s)", "duration_s", 1)
    _row("IAT mean (s)", "iat_mean", 2)
    _row("IAT p50 (s)", "iat_p50", 3)
    _row("IAT p90 (s)", "iat_p90", 2)
    _row("IAT p99 (s)", "iat_p99", 1)
    _row("Unique templates", "n_unique_templates", 0)
    _row("Peak-hour queries", "peak_hour_count", 0)

    # Scalar distances row.
    table.add_section()
    table.add_row(
        "Hourly profile TV dist",
        _fmt(tv_dist, 4),
        "",
        "(0 = identical)",
        "",
    )
    table.add_row(
        "Template JSD",
        _fmt(jsd, 4),
        "",
        "(0 = identical)",
        "",
    )

    console.print(table)


def _print_hourly(
    dir_dfs: list[tuple[str, pd.DataFrame]],
    ref_df: pd.DataFrame,
    ref_label: str,
    dir_label: str,
) -> None:
    dfs = [df for _, df in dir_dfs]
    hourly_stack = np.stack([_hourly_fracs(df) for df in dfs])  # (n, 24)
    dir_mean = hourly_stack.mean(axis=0)
    dir_std = hourly_stack.std(axis=0)
    ref = _hourly_fracs(ref_df)

    # Also get raw mean counts per workload for readability.
    count_stack = np.stack([
        df.groupby("hour").size().reindex(range(24), fill_value=0).values
        for df in dfs
    ], dtype=float)
    count_mean = count_stack.mean(axis=0)
    count_std = count_stack.std(axis=0)
    ref_counts = ref_df.groupby("hour").size().reindex(range(24), fill_value=0).values.astype(float)

    table = Table(title="Hourly Query Profile", show_lines=False)
    table.add_column("Hour", justify="right", style="cyan")
    table.add_column(f"{ref_label} count", justify="right")
    table.add_column(f"{dir_label} mean count", justify="right")
    table.add_column(f"{dir_label} std", justify="right", style="dim")
    table.add_column(f"{ref_label} frac", justify="right")
    table.add_column(f"{dir_label} mean frac", justify="right")
    table.add_column("Δ frac", justify="right")

    for h in range(24):
        delta = dir_mean[h] - ref[h]
        style = _delta_style(delta, threshold=0.005)
        sign = "+" if delta >= 0 else ""
        table.add_row(
            str(h),
            f"{ref_counts[h]:.0f}",
            f"{count_mean[h]:.1f}",
            f"{count_std[h]:.1f}",
            f"{ref[h]:.4f}",
            f"{dir_mean[h]:.4f}",
            _styled(f"{sign}{delta:.4f}", style),
        )

    console.print(table)


def _print_templates(
    dir_dfs: list[tuple[str, pd.DataFrame]],
    ref_df: pd.DataFrame,
    ref_label: str,
    dir_label: str,
    top_n: int,
) -> None:
    dfs = [df for _, df in dir_dfs]

    # Build a combined template frequency matrix.
    all_templates = sorted(
        set(ref_df["template_id"].unique())
        | {t for df in dfs for t in df["template_id"].unique()}
    )
    ref_fracs = _template_fracs(ref_df).reindex(all_templates, fill_value=0.0)

    dir_frac_matrix = pd.concat(
        [_template_fracs(df).reindex(all_templates, fill_value=0.0) for df in dfs],
        axis=1,
    )
    dir_mean = dir_frac_matrix.mean(axis=1)
    dir_std = dir_frac_matrix.std(axis=1)

    # Sort by reference frequency descending, take top N.
    top_templates = ref_fracs.nlargest(top_n).index.tolist()
    other_ref = ref_fracs.drop(top_templates).sum()
    other_dir_mean = dir_mean.drop(top_templates).sum()
    other_dir_std = dir_std.drop(top_templates).mean()  # approximate

    table = Table(title=f"Template Distribution (top {top_n} by reference)", show_lines=False)
    table.add_column("Template", style="cyan", no_wrap=True)
    table.add_column(f"{ref_label} %", justify="right")
    table.add_column(f"{dir_label} mean %", justify="right")
    table.add_column(f"{dir_label} std %", justify="right", style="dim")
    table.add_column("Δ pp", justify="right")

    for t in top_templates:
        r = ref_fracs[t] * 100
        m = dir_mean[t] * 100
        s = dir_std[t] * 100
        delta = m - r
        style = _delta_style(delta / 100, threshold=0.005)
        sign = "+" if delta >= 0 else ""
        table.add_row(
            t,
            f"{r:.2f}%",
            f"{m:.2f}%",
            f"{s:.2f}%",
            _styled(f"{sign}{delta:.2f}pp", style),
        )

    table.add_section()
    delta_other = (other_dir_mean - other_ref) * 100
    sign = "+" if delta_other >= 0 else ""
    style = _delta_style(delta_other / 100, threshold=0.005)
    table.add_row(
        "(other)",
        f"{other_ref * 100:.2f}%",
        f"{other_dir_mean * 100:.2f}%",
        f"{other_dir_std * 100:.2f}%",
        _styled(f"{sign}{delta_other:.2f}pp", style),
    )

    console.print(table)


def _print_per_scenario(
    dir_dfs: list[tuple[str, pd.DataFrame]],
    ref_df: pd.DataFrame,
    ref_label: str,
) -> None:
    table = Table(title="Per-Scenario Details", show_lines=False)
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("Queries", justify="right")
    table.add_column("Duration (s)", justify="right")
    table.add_column("IAT mean (s)", justify="right")
    table.add_column("IAT p50 (s)", justify="right")
    table.add_column("IAT p90 (s)", justify="right")
    table.add_column("Unique tmpl", justify="right")

    def _add_row(label: str, df: pd.DataFrame) -> None:
        s = _scenario_stats(df)
        table.add_row(
            label,
            str(s["n_queries"]),
            _fmt(s["duration_s"], 1),
            _fmt(s["iat_mean"], 2),
            _fmt(s["iat_p50"], 3),
            _fmt(s["iat_p90"], 2),
            str(s["n_unique_templates"]),
        )

    _add_row(f"[bold]{ref_label}[/]", ref_df)
    table.add_section()
    for name, df in dir_dfs:
        _add_row(name, df)

    console.print(table)


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a directory of forecasted workloads against a single reference workload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "workload_dir",
        type=Path,
        help="Directory containing .parquet workload files (the sampled/forecasted scenarios).",
    )
    parser.add_argument(
        "reference_file",
        type=Path,
        help="Single .parquet workload file to compare against (the oracle / target-day workload).",
    )
    parser.add_argument(
        "--top-templates",
        type=int,
        default=15,
        metavar="N",
        help="Number of top templates (by reference frequency) to show in the template table (default: 15).",
    )
    parser.add_argument(
        "--show-per-scenario",
        action="store_true",
        help="Also print a per-scenario detail table for the directory workloads.",
    )
    parser.add_argument(
        "--dir-label",
        default=None,
        metavar="LABEL",
        help="Label for the directory (default: directory name).",
    )
    parser.add_argument(
        "--ref-label",
        default=None,
        metavar="LABEL",
        help="Label for the reference file (default: file stem).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    workload_dir: Path = args.workload_dir
    reference_file: Path = args.reference_file

    if not workload_dir.is_dir():
        console.print(f"[red]Not a directory: {workload_dir}[/]")
        sys.exit(1)
    if not reference_file.is_file():
        console.print(f"[red]Not a file: {reference_file}[/]")
        sys.exit(1)

    dir_label = args.dir_label or workload_dir.name
    ref_label = args.ref_label or reference_file.stem

    console.rule(f"[bold cyan]Loading workloads")
    dir_dfs = _load_dir(workload_dir)
    ref_df = _load(reference_file)
    console.print(
        f"  Directory: [bold]{workload_dir}[/] — {len(dir_dfs)} scenario(s)\n"
        f"  Reference: [bold]{reference_file}[/]"
    )

    console.rule("[bold cyan]Summary")
    _print_summary(dir_dfs, ref_df, ref_label=ref_label, dir_label=dir_label)

    console.rule("[bold cyan]Hourly Profile")
    _print_hourly(dir_dfs, ref_df, ref_label=ref_label, dir_label=dir_label)

    console.rule("[bold cyan]Template Distribution")
    _print_templates(
        dir_dfs,
        ref_df,
        ref_label=ref_label,
        dir_label=dir_label,
        top_n=args.top_templates,
    )

    if args.show_per_scenario:
        console.rule("[bold cyan]Per-Scenario Details")
        _print_per_scenario(dir_dfs, ref_df, ref_label=ref_label)


if __name__ == "__main__":
    main()
