"""
Diagnose why a forecast policy produces a lighter/heavier workload than the
actual target day, by tracing the exact counts and weights the Forecaster would
use per hour.

The tool constructs a real Forecaster (and QueryReservoir) from the provided
arguments, calls its internal per-hour methods directly, and compares predicted
counts to the oracle counts derived from the target date in the same workload
file.

Usage
-----
  python tools/diagnose_forecaster.py \\
      --workload <path/to/workload.parquet> \\
      --target-date YYYY-MM-DD \\
      --num-days N \\
      [--forecast-policy POLICY] \\
      [--decay-factor F] \\
      [--arrival-time-policy POLICY] \\
      [--min-gaps-for-deciles N] \\
      [--max-arrivals-per-hour-safety-cap N] \\
      [--fixed-queries-per-hour N] \\
      [--use-fixed-queries-per-hour] \\
      [--seed N] \\
      [--rescale-factor F]

Examples
--------
  # 1-month reservoir, default decay, same_day_exponential
  python tools/diagnose_forecaster.py \\
      --workload data/workloads/redbench_provisioned_157_0.parquet \\
      --target-date 2024-05-27 \\
      --num-days 30

  # 1-week reservoir, interarrival_deciles arrival policy
  python tools/diagnose_forecaster.py \\
      --workload data/workloads/redbench_provisioned_157_0.parquet \\
      --target-date 2024-05-27 \\
      --num-days 7 \\
      --arrival-time-policy interarrival_deciles \\
      --decay-factor 0.8
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from autoslo.config.component_configs import ForecasterConfig, ReservoirConfig
from autoslo.forecasting.forecast_policy import ArrivalTimePolicy, ForecastPolicy
from autoslo.forecasting.forecaster import Forecaster

console = Console()

_ANOMALY_THRESHOLD = 1.5  # flag a source day if its count deviates >50% from median


# ── helpers ───────────────────────────────────────────────────────────────────


def _styled(text: str, style: str) -> str:
    if not style:
        return text
    return f"[{style}]{text}[/]"


def _oracle_counts(workload_path: Path, target_date: datetime.date) -> pd.Series:
    """Load oracle (actual) per-hour query counts for target_date from the parquet."""
    df = pd.read_parquet(workload_path)
    df["_date"] = pd.to_datetime(df["abs_start_time"]).dt.date
    df["_hour"] = pd.to_datetime(df["abs_start_time"]).dt.hour
    return (
        df[df["_date"] == target_date]
        .groupby("_hour")
        .size()
        .reindex(range(24), fill_value=0)
        .rename("oracle")
    )


def _same_day_source_days(
    forecaster: Forecaster,
    target_date: datetime.date,
) -> tuple[list[tuple[datetime.date, float]], bool]:
    """
    Enumerate the same-weekday source days and their weights, as the
    SAME_DAY_EXPONENTIAL policy would.

    Mirrors the full logic of _build_bin_df_exponential and
    _n_samples_exponential, including the fallback to yesterday when the
    reservoir doesn't reach back a full week.

    Returns (source_days, is_fallback).  When is_fallback is True the single
    entry in source_days is yesterday with weight 1.0.
    """
    result: list[tuple[datetime.date, float]] = []
    weight = 1.0
    day = target_date - datetime.timedelta(days=7)
    min_date = forecaster.reservoir.min_date
    decay = forecaster.forecaster_config.decay_factor
    while day >= min_date:
        result.append((day, weight))
        weight *= decay
        day -= datetime.timedelta(days=7)

    if result:
        return result, False

    # Mirrors the fallback branch in _build_bin_df_exponential:
    # only fall back to yesterday when target_date - 7 is genuinely before
    # the reservoir's start; if it's within range but has no data, the
    # forecaster returns an empty bin_df and produces 0 samples.
    if (target_date - datetime.timedelta(days=7)) < min_date:
        yesterday = target_date - datetime.timedelta(days=1)
        return [(yesterday, 1.0)], True

    return [], False


def _hourly_reservoir_count(
    forecaster: Forecaster, day: datetime.date, hour: int
) -> int:
    """Raw query count for (day, hour) from the reservoir's count_df."""
    return int(forecaster.reservoir.bin_df(day, hour)["count"].sum())


def _predicted_count(
    forecaster: Forecaster,
    target_date: datetime.date,
    hour: int,
) -> float:
    """
    Return the predicted number of queries for (target_date, hour) by calling
    the real Forecaster methods.  Returns a float so callers can see the
    pre-rounding value.
    """
    bin_df = forecaster._build_bin_df(target_date, hour)
    if bin_df.empty:
        return 0.0
    return forecaster._n_samples(target_date, hour, bin_df)


def _gap_decile_median(
    forecaster: Forecaster,
    target_date: datetime.date,
    hour: int,
) -> float | None:
    """Return the p50 gap (seconds) from the decile model, or None if unavailable."""
    deciles = forecaster._get_gap_deciles_for_bin(target_date, hour)
    if deciles is None:
        return None
    # deciles has n_quantiles+1 entries; p50 is at index n_quantiles//2.
    n = len(deciles) - 1
    return float(deciles[n // 2])


# ── printing ──────────────────────────────────────────────────────────────────


def _print_source_days(
    forecaster: Forecaster,
    source_days: list[tuple[datetime.date, float]],
    is_fallback: bool,
) -> None:
    """Table 1 (SAME_DAY_EXPONENTIAL only): one row per source day."""
    total_weight = sum(w for _, w in source_days)

    daily_totals = [
        sum(_hourly_reservoir_count(forecaster, day, h) for h in range(24))
        for day, _ in source_days
    ]
    median_total = float(np.median(daily_totals)) if daily_totals else 0.0

    title = "Source Days (fallback: yesterday)" if is_fallback else "Source Days"
    table = Table(title=title, show_lines=False)
    table.add_column("Week", justify="right", style="cyan")
    table.add_column("Date", no_wrap=True)
    table.add_column("Weekday", no_wrap=True)
    table.add_column("Weight", justify="right")
    table.add_column("Weight share %", justify="right")
    table.add_column("Day total queries", justify="right")
    table.add_column("Anomalous?", justify="center")

    for i, ((day, weight), total) in enumerate(zip(source_days, daily_totals)):
        share_pct = 100.0 * weight / total_weight if total_weight > 0 else 0.0
        is_anomalous = median_total > 0 and (
            total > median_total * _ANOMALY_THRESHOLD
            or total < median_total / _ANOMALY_THRESHOLD
        )
        anomaly_str = _styled("YES", "red bold") if is_anomalous else "no"
        share_style = "bold" if i == 0 else ""
        table.add_row(
            str(i + 1),
            str(day),
            day.strftime("%A"),
            _styled(f"{weight:.4f}", share_style),
            _styled(f"{share_pct:.1f}%", share_style),
            _styled(str(total), "red" if is_anomalous else ""),
            anomaly_str,
        )

    table.add_section()
    table.add_row(
        "", "", "TOTAL", f"{total_weight:.4f}", "100.0%",
        str(sum(daily_totals)), "",
    )
    console.print(table)


def _print_hourly(
    forecaster: Forecaster,
    target_date: datetime.date,
    source_days: list[tuple[datetime.date, float]],
    oracle: pd.Series,
    show_deciles: bool,
) -> None:
    """Table 2: per-hour breakdown."""
    is_exp = (
        forecaster.forecast_policy == ForecastPolicy.SAME_DAY_EXPONENTIAL
        and source_days
    )

    table = Table(title="Per-Hour Count Breakdown", show_lines=False)
    table.add_column("Hour", justify="right", style="cyan")

    if is_exp:
        total_w = sum(w for _, w in source_days)
        for day, weight in source_days:
            share = weight / total_w * 100
            table.add_column(
                f"{day}\n(w={weight:.2f}, {share:.0f}%)",
                justify="right",
            )

    table.add_column("Predicted", justify="right", style="bold")
    table.add_column("Oracle", justify="right")
    table.add_column("Ratio", justify="right")
    if show_deciles:
        table.add_column("Gap p50 (s)", justify="right", style="dim")

    for h in range(24):
        predicted = _predicted_count(forecaster, target_date, h)
        oracle_count = int(oracle.iloc[h])

        ratio = predicted / oracle_count if oracle_count > 0 else float("nan")
        if np.isnan(ratio):
            ratio_str, ratio_style = "n/a", ""
        elif ratio < 0.7:
            ratio_str, ratio_style = f"{ratio:.2f}", "red bold"
        elif ratio > 1.3:
            ratio_str, ratio_style = f"{ratio:.2f}", "yellow"
        else:
            ratio_str, ratio_style = f"{ratio:.2f}", "green"

        row: list[str] = [str(h)]

        if is_exp:
            day_counts = [
                _hourly_reservoir_count(forecaster, day, h)
                for day, _ in source_days
            ]
            nonzero = [c for c in day_counts if c > 0]
            row_median = float(np.median(nonzero)) if nonzero else 0.0
            for count in day_counts:
                if row_median > 0 and count < row_median / _ANOMALY_THRESHOLD:
                    row.append(_styled(str(count), "red"))
                elif row_median > 0 and count > row_median * _ANOMALY_THRESHOLD:
                    row.append(_styled(str(count), "yellow"))
                else:
                    row.append(str(count))

        row.append(f"{predicted:.1f}")
        row.append(str(oracle_count))
        row.append(_styled(ratio_str, ratio_style))

        if show_deciles:
            gap_p50 = _gap_decile_median(forecaster, target_date, h)
            row.append(f"{gap_p50:.3f}" if gap_p50 is not None else "n/a")

        table.add_row(*row)

    console.print(table)


def _print_error_summary(
    forecaster: Forecaster,
    target_date: datetime.date,
    oracle: pd.Series,
) -> None:
    """Table 3: per-hour errors sorted by absolute under/over-estimation."""
    rows = []
    for h in range(24):
        predicted = _predicted_count(forecaster, target_date, h)
        oracle_count = int(oracle.iloc[h])
        error = predicted - oracle_count
        pct = (error / oracle_count * 100) if oracle_count > 0 else float("nan")
        rows.append((h, predicted, oracle_count, error, pct))

    rows.sort(key=lambda r: abs(r[3]), reverse=True)

    table = Table(title="Hour-Level Error Summary (sorted by |error|)", show_lines=False)
    table.add_column("Hour", justify="right", style="cyan")
    table.add_column("Predicted", justify="right")
    table.add_column("Oracle", justify="right")
    table.add_column("Error (pred − oracle)", justify="right")
    table.add_column("Error %", justify="right")

    for h, predicted, oracle_count, error, pct in rows:
        sign = "+" if error >= 0 else ""
        style = ""
        if not np.isnan(pct) and abs(pct) > 30 and oracle_count > 5:
            style = "red" if error < 0 else "yellow"
        pct_str = f"{sign}{pct:.1f}%" if not np.isnan(pct) else "n/a"
        table.add_row(
            str(h),
            f"{predicted:.1f}",
            str(oracle_count),
            _styled(f"{sign}{error:.1f}", style),
            _styled(pct_str, style),
        )

    total_predicted = sum(r[1] for r in rows)
    total_oracle = sum(r[2] for r in rows)
    total_error = total_predicted - total_oracle
    total_pct = total_error / total_oracle * 100 if total_oracle > 0 else float("nan")
    sign = "+" if total_error >= 0 else ""
    table.add_section()
    table.add_row(
        "TOTAL",
        f"{total_predicted:.1f}",
        str(total_oracle),
        _styled(f"{sign}{total_error:.1f}", "red" if total_error < 0 else "yellow"),
        _styled(f"{sign}{total_pct:.1f}%", "red" if total_error < 0 else "yellow"),
    )
    console.print(table)


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose a forecast policy by tracing per-hour counts and weights "
            "for all source days in the reservoir window."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--workload", type=Path, required=True, metavar="PATH",
        help="Path to the historical workload parquet (reservoir source and oracle).",
    )
    parser.add_argument(
        "--target-date", required=True, metavar="YYYY-MM-DD",
        help="The day being forecast (oracle counts come from this date in --workload).",
    )
    parser.add_argument(
        "--num-days", type=int, required=True, metavar="N",
        help="Reservoir window in days, ending the day before --target-date.",
    )

    # ForecasterConfig fields (all optional)
    parser.add_argument(
        "--forecast-policy", default="same_day_exponential", metavar="POLICY",
        help=(
            "Forecast policy name (default: same_day_exponential). "
            f"Choices: {[p.value for p in ForecastPolicy if p != ForecastPolicy.NONE]}"
        ),
    )
    parser.add_argument(
        "--decay-factor", type=float, default=0.5, metavar="F",
        help="Exponential decay applied per week for same_day_exponential (default: 0.5).",
    )
    parser.add_argument(
        "--arrival-time-policy", default="uniform", metavar="POLICY",
        help=(
            "Arrival time policy name (default: uniform). "
            f"Choices: {[p.value for p in ArrivalTimePolicy]}"
        ),
    )
    parser.add_argument(
        "--min-gaps-for-deciles", type=int, default=20, metavar="N",
        help="Minimum gaps required to compute deciles (default: 20).",
    )
    parser.add_argument(
        "--max-arrivals-per-hour-safety-cap", type=int, default=10_000, metavar="N",
        help="Safety cap on arrivals sampled per hour (default: 10000).",
    )
    parser.add_argument(
        "--fixed-queries-per-hour", type=int, default=100, metavar="N",
        help="Fixed queries per hour when --use-fixed-queries-per-hour is set (default: 100).",
    )

    # forecast() parameters (all optional)
    parser.add_argument(
        "--use-fixed-queries-per-hour", action="store_true",
        help="Ignore historical counts and use --fixed-queries-per-hour for every hour.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed (relevant for interarrival_deciles arrival policy; default: 42).",
    )
    parser.add_argument(
        "--rescale-factor", type=float, default=1.0, metavar="F",
        help="Rescale factor applied to workload relative times (default: 1.0).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.workload.is_file():
        console.print(f"[red]Workload file not found: {args.workload}[/]")
        sys.exit(1)

    target_date = datetime.date.fromisoformat(args.target_date)
    last_day = target_date - datetime.timedelta(days=1)
    first_day = last_day - datetime.timedelta(days=args.num_days - 1)

    # Build config objects, mirroring what PolicyTuner / SamplingConfig does.
    reservoir_config = ReservoirConfig(
        workload_name=args.workload.stem,
        last_day_date_inclusive=str(last_day),
        num_days=args.num_days,
        workload_dir=args.workload.parent,
    )
    forecaster_config = ForecasterConfig(
        forecast_policy_name=args.forecast_policy,
        decay_factor=args.decay_factor,
        arrival_time_policy_name=args.arrival_time_policy,
        min_gaps_for_deciles=args.min_gaps_for_deciles,
        max_arrivals_per_hour_safety_cap=args.max_arrivals_per_hour_safety_cap,
        fixed_queries_per_hour=args.fixed_queries_per_hour,
        rescale_factor=args.rescale_factor,
        reservoir_config=reservoir_config,
    )

    console.rule("[bold cyan]Configuration")
    decay_str = (
        f"{args.decay_factor} (weight halves every "
        f"{1 / abs(np.log2(args.decay_factor)):.1f} weeks)"
        if args.decay_factor not in (0.0, 1.0)
        else str(args.decay_factor)
    )
    console.print(
        f"  Workload             : [bold]{args.workload}[/]\n"
        f"  Target date          : [bold]{target_date}[/] ({target_date.strftime('%A')})\n"
        f"  Reservoir            : {first_day} -> {last_day}  ({args.num_days} days)\n"
        f"  Forecast policy      : {args.forecast_policy}\n"
        f"  Decay factor         : {decay_str}\n"
        f"  Arrival policy       : {args.arrival_time_policy}\n"
        f"  Rescale factor       : {args.rescale_factor}\n"
        f"  Seed                 : {args.seed}\n"
        f"  Fixed QPH mode       : {args.use_fixed_queries_per_hour}"
        + (f" ({args.fixed_queries_per_hour} qph)" if args.use_fixed_queries_per_hour else "")
    )

    console.rule("[bold cyan]Building Forecaster")
    forecaster = Forecaster(forecaster_config)

    is_exp = forecaster.forecast_policy == ForecastPolicy.SAME_DAY_EXPONENTIAL
    is_fallback = False
    source_days: list[tuple[datetime.date, float]] = []
    if is_exp:
        source_days, is_fallback = _same_day_source_days(forecaster, target_date)

    if is_exp:
        if not source_days:
            # Reservoir is within range but has no same-weekday data and
            # target_date - 7 is still >= min_date, so the forecaster would
            # produce 0 samples for every hour.
            console.print(
                "[yellow]No same-weekday source days found and no fallback applies "
                "(reservoir has no data for this weekday). Predicted counts will be 0.[/]"
            )
        elif is_fallback:
            console.print(
                f"  [yellow]Fallback mode[/]: reservoir doesn't reach back a full week; "
                f"using yesterday ({source_days[0][0]}) as the single source day."
            )
        else:
            console.print(
                f"  {len(source_days)} same-weekday source day(s): "
                + " -> ".join(str(d) for d, _ in source_days[:4])
                + (" -> ..." if len(source_days) > 4 else "")
            )

    oracle = _oracle_counts(args.workload, target_date)
    if oracle.sum() == 0:
        console.print(
            f"[yellow]Warning: no rows found for {target_date} in the workload "
            "file. Oracle counts will all be 0.[/]"
        )

    show_deciles = (
        forecaster.arrival_time_policy == ArrivalTimePolicy.INTERARRIVAL_DECILES
    )

    if is_exp:
        console.rule("[bold cyan]Source Days")
        _print_source_days(forecaster, source_days, is_fallback)

    console.rule("[bold cyan]Per-Hour Breakdown")
    _print_hourly(forecaster, target_date, source_days, oracle, show_deciles)

    console.rule("[bold cyan]Error Summary")
    _print_error_summary(forecaster, target_date, oracle)


if __name__ == "__main__":
    main()
