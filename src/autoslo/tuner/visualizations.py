"""Visualization helpers for the tuner pipeline.

Covers reservoir diagnostics, sampling fidelity, sweep analysis,
simulation diagnostics, and cross-phase summaries.  All plot functions
return ``(Figure,)`` or ``(Figure, Axes)`` for use in Jupyter or batch
report generation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Optional import — plots degrade gracefully if matplotlib is absent.
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend by default
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for tuner visualizations. "
            "Install it with: pip install matplotlib"
        )


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


# Consistent style constants.
_FIGSIZE_WIDE = (14, 5)
_FIGSIZE_MEDIUM = (10, 5)
_FIGSIZE_SQUARE = (7, 6)


# ======================================================================
# 1. Reservoir heatmap
# ======================================================================


def plot_reservoir_heatmap(
    reservoir,
    title: str = "Reservoir: arrivals per (day_of_week, hour)",
    figsize: tuple[float, float] = (14, 4),
) -> "Figure":
    """Heatmap of query arrival counts over (day_of_week, hour) bins.

    Parameters
    ----------
    reservoir :
        A :class:`~autoslo.tuner.reservoir.QueryReservoir`.
    """
    _require_mpl()

    summary = reservoir.summary()
    pivot = summary.pivot(
        index="day_of_week", columns="hour", values="count"
    ).fillna(0)

    # Ensure all hours 0-23 and days 0-6 are represented.
    full_index = range(7)
    full_cols = range(24)
    pivot = pivot.reindex(index=full_index, columns=full_cols, fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=8)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Query count")
    fig.tight_layout()
    return fig


# ======================================================================
# 2. Forecast preview bar chart
# ======================================================================


def plot_forecast_preview(
    preview_df: pd.DataFrame,
    title: str = "Expected query count per hour bin",
    figsize: tuple[float, float] = (14, 4),
) -> "Figure":
    """Bar chart of expected query counts from :meth:`WorkloadSampler.preview`.

    Parameters
    ----------
    preview_df :
        DataFrame returned by
        :meth:`~autoslo.tuner.workload_sampler.WorkloadSampler.preview`.
    """
    _require_mpl()

    fig, ax = plt.subplots(figsize=figsize)
    labels = [
        f"{row['bin_start'].strftime('%a %H:%M')}"
        for _, row in preview_df.iterrows()
    ]
    x = np.arange(len(labels))
    ax.bar(x, preview_df["expected_count"], color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Expected queries")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ======================================================================
# 3. Sampled workload arrival timeline
# ======================================================================


def plot_workload_arrivals(
    workloads: list,
    max_scenarios: int = 10,
    title: str = "Sampled workload arrival timelines",
    figsize: tuple[float, float] = (14, 6),
) -> "Figure":
    """Scatter plot showing query arrival times across sampled scenarios.

    Each row on the y-axis is one scenario; each dot is a query arrival.

    Parameters
    ----------
    workloads :
        List of :class:`~autoslo.workload_definition.workload.Workload`.
    max_scenarios :
        Maximum number of scenarios to plot (to avoid clutter).
    """
    _require_mpl()

    n = min(len(workloads), max_scenarios)
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        df = workloads[i].df
        if "rel_start_time_s" in df.columns:
            times = df["rel_start_time_s"].values / 3600.0  # hours
        else:
            ts = df["abs_start_time"]
            t0 = ts.min()
            times = (ts - t0).dt.total_seconds().values / 3600.0
        ax.scatter(
            times,
            np.full_like(times, i),
            s=4,
            alpha=0.6,
            label=workloads[i].workload_name if i < 5 else None,
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Scenario")
    ax.set_yticks(range(n))
    ax.set_yticklabels([workloads[i].workload_name for i in range(n)], fontsize=7)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ======================================================================
# 4. Query count histogram across scenarios
# ======================================================================


def plot_query_count_distribution(
    workloads: list,
    title: str = "Query count distribution across scenarios",
    figsize: tuple[float, float] = (8, 4),
) -> "Figure":
    """Histogram of total query counts across sampled workloads."""
    _require_mpl()

    counts = [len(wl.df) for wl in workloads]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(counts, bins=max(5, len(counts) // 3), color="steelblue", edgecolor="white")
    ax.axvline(np.mean(counts), color="red", linestyle="--", label=f"mean={np.mean(counts):.0f}")
    ax.axvline(np.median(counts), color="orange", linestyle="--", label=f"median={np.median(counts):.0f}")
    ax.set_xlabel("Total queries")
    ax.set_ylabel("Scenarios")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ======================================================================
# 5. Template frequency comparison (reservoir vs. sampled)
# ======================================================================


def plot_template_frequency(
    reservoir,
    workloads: list,
    top_k: int = 20,
    title: str = "Template frequency: reservoir vs. sampled",
    figsize: tuple[float, float] = (12, 5),
) -> "Figure":
    """Compare template frequencies between the reservoir and sampled workloads.

    Parameters
    ----------
    reservoir :
        A :class:`~autoslo.tuner.reservoir.QueryReservoir`.
    workloads :
        Sampled :class:`Workload` objects.
    top_k :
        Number of most-frequent templates to plot.
    """
    _require_mpl()
    from autoslo.workload_definition.query import QueryTextId

    # Reservoir template frequencies.
    res_counts = reservoir.df["query_text_id"].value_counts()
    res_freq = res_counts / res_counts.sum()

    # Sampled template frequencies (aggregate across all workloads).
    all_qtids = pd.concat([wl.df["query_text_id"] for wl in workloads])
    samp_counts = all_qtids.value_counts()
    samp_freq = samp_counts / samp_counts.sum()

    # Union of top-K from each.
    top_templates = (
        res_freq.head(top_k).index.union(samp_freq.head(top_k).index)
    )
    top_templates = sorted(top_templates)[:top_k]

    res_vals = [res_freq.get(t, 0.0) for t in top_templates]
    samp_vals = [samp_freq.get(t, 0.0) for t in top_templates]

    # Shorten labels: extract template_id if possible.
    labels = []
    for t in top_templates:
        try:
            labels.append(QueryTextId(t).template_id)
        except Exception:
            labels.append(str(t)[:20])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, res_vals, width, label="Reservoir", color="steelblue")
    ax.bar(x + width / 2, samp_vals, width, label="Sampled", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ======================================================================
# 6. Per-hour arrival rate comparison
# ======================================================================


def plot_hourly_rates(
    reservoir,
    workloads: list,
    target_start: datetime,
    target_end: datetime,
    title: str = "Hourly arrival rate: reservoir avg vs. sampled",
    figsize: tuple[float, float] = (14, 4),
) -> "Figure":
    """Compare reservoir average hourly rate to the actual sampled counts."""
    _require_mpl()

    hours_range = []
    current = target_start.replace(minute=0, second=0, microsecond=0)
    while current < target_end:
        hours_range.append(current)
        current += timedelta(hours=1)

    res_rates = []
    sampled_means = []
    sampled_stds = []

    for h_start in hours_range:
        dow = h_start.weekday()
        hour = h_start.hour
        rate = reservoir.query_rate_per_hour(dow, hour)
        res_rates.append(rate)

        # Per-workload count in this hour bin.
        counts = []
        for wl in workloads:
            df = wl.df
            mask = (df["abs_start_time"] >= h_start) & (
                df["abs_start_time"] < h_start + timedelta(hours=1)
            )
            counts.append(int(mask.sum()))
        sampled_means.append(np.mean(counts) if counts else 0)
        sampled_stds.append(np.std(counts) if counts else 0)

    x = np.arange(len(hours_range))
    labels = [h.strftime("%a %H") for h in hours_range]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, res_rates, "o-", label="Reservoir avg", color="steelblue")
    ax.errorbar(
        x,
        sampled_means,
        yerr=sampled_stds,
        fmt="s-",
        label="Sampled mean ± std",
        color="coral",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Queries / hour")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ======================================================================
# V1.1 — Windowed-Template Diagnostic Panel
# ======================================================================


def plot_windowed_template_diagnostics(
    reservoir_meta: dict[str, Any],
    reservoir_df: pd.DataFrame | None = None,
    top_k: int = 6,
    figsize: tuple[float, float] = (16, 10),
) -> "Figure":
    """Multi-panel diagnostic for windowed-template classification.

    Parameters
    ----------
    reservoir_meta :
        The ``reservoir_meta.yml`` dict (must contain ``"classifications"``).
    reservoir_df :
        Optional reservoir DataFrame — if provided, panel (b) shows folded
        arrival scatter for top-K windowed templates.
    top_k :
        Number of windowed templates to show in the folded-scatter panel.
    """
    _require_mpl()

    classifications = reservoir_meta.get("classifications", {})
    if not classifications:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No classifications available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Classify counts.
    counts = {"windowed": 0, "normal": 0, "too_few_samples": 0}
    windowed_templates: list[tuple[str, dict]] = []
    for gid, info in classifications.items():
        cls = info.get("classification", "normal")
        counts[cls] = counts.get(cls, 0) + 1
        if cls == "windowed":
            windowed_templates.append((gid, info))

    # Sort windowed by num_samples descending for top-K.
    windowed_templates.sort(key=lambda t: t[1].get("num_samples", 0), reverse=True)

    has_folded = reservoir_df is not None and len(windowed_templates) > 0
    has_periods = len(windowed_templates) > 0

    n_panels = 1 + int(has_folded) + int(has_periods)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Panel (a): classification bar chart.
    ax = axes[0]
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = ["#E07022", "#3466FF", "#D3D3D3"]
    ax.bar(labels, values, color=colors, edgecolor="white")
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
    ax.set_ylabel("Number of groups")
    ax.set_title("(a) Classification counts")

    panel_idx = 1

    # Panel (b): folded arrivals for top-K windowed templates.
    if has_folded:
        ax = axes[panel_idx]
        panel_idx += 1
        shown = windowed_templates[:top_k]
        cmap = plt.cm.tab10
        for i, (gid, info) in enumerate(shown):
            period = info.get("period_s", 3600.0)
            active = info.get("active_length_s", period * 0.3)
            start = info.get("on_window_rel_start_s", 0.0)
            group_mask = reservoir_df["repetition_id"].astype(str) == gid
            if group_mask.sum() == 0:
                group_mask = reservoir_df["query_text_id"].astype(str) == gid
            times = reservoir_df.loc[group_mask, "timestamp_within_hour"].values
            folded = np.mod(times, period)
            ax.scatter(
                folded, np.full_like(folded, i), s=6, alpha=0.5,
                color=cmap(i % 10), label=gid[:20],
            )
            # Shade active window.
            ax.axvspan(start, start + active, ymin=i / len(shown),
                       ymax=(i + 1) / len(shown), alpha=0.15,
                       color=cmap(i % 10))
        ax.set_xlabel("Time mod period (s)")
        ax.set_ylabel("Template group")
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels([g[:20] for g, _ in shown], fontsize=7)
        ax.set_title(f"(b) Folded arrivals (top {len(shown)} windowed)")

    # Panel (c): histogram of detected periods.
    if has_periods:
        ax = axes[panel_idx]
        periods = [
            info.get("period_s", 0)
            for _, info in windowed_templates
            if info.get("period_s")
        ]
        if periods:
            ax.hist(periods, bins=min(20, len(periods)), color="#E07022", edgecolor="white")
        ax.set_xlabel("Detected period (s)")
        ax.set_ylabel("Count")
        ax.set_title("(c) Period distribution")

    fig.suptitle("Windowed-Template Diagnostics", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ======================================================================
# V1.2 — Forecast Fidelity: Expected vs. Sampled
# ======================================================================


def plot_forecast_fidelity(
    preview_df: pd.DataFrame,
    sampled_workload_dir: Path,
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Expected count per hour-bin vs. actual sampled mean ± std.

    Parameters
    ----------
    preview_df :
        DataFrame from ``WorkloadSampler.preview()``, with columns
        ``bin_start``, ``expected_count``.
    sampled_workload_dir :
        Directory containing ``t_000.parquet``, ``t_001.parquet``, etc.
    """
    _require_mpl()

    parquets = sorted(sampled_workload_dir.glob("*.parquet"))
    if not parquets:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No sampled workloads found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Count queries per hour bin in each scenario.
    bin_starts = preview_df["bin_start"].values
    per_scenario: list[list[int]] = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        if "abs_start_time" not in df.columns:
            continue
        counts_for_scenario = []
        for bs in bin_starts:
            bs_ts = pd.Timestamp(bs)
            be_ts = bs_ts + pd.Timedelta(hours=1)
            n = int(((df["abs_start_time"] >= bs_ts) & (df["abs_start_time"] < be_ts)).sum())
            counts_for_scenario.append(n)
        per_scenario.append(counts_for_scenario)

    if not per_scenario:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Could not parse workloads", ha="center", va="center")
        ax.set_axis_off()
        return fig

    arr = np.array(per_scenario)  # (n_scenarios, n_bins)
    sampled_mean = arr.mean(axis=0)
    sampled_std = arr.std(axis=0)

    labels = [pd.Timestamp(b).strftime("%a %H:%M") for b in bin_starts]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - 0.2, preview_df["expected_count"].values, 0.4,
           label="Forecast expected", color="steelblue", edgecolor="white")
    ax.bar(x + 0.2, sampled_mean, 0.4, yerr=sampled_std,
           label="Sampled mean ± std", color="coral", edgecolor="white", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Query count")
    ax.set_title("Forecast Fidelity: Expected vs. Sampled")
    ax.legend()
    fig.tight_layout()
    return fig


# ======================================================================
# V1.3 — Per-Hour Template Frequency
# ======================================================================


def plot_per_hour_template_frequency(
    reservoir,
    sampled_workload_dir: Path,
    target_start: datetime,
    target_end: datetime,
    top_k: int = 10,
    max_hours: int = 6,
    figsize: tuple[float, float] = (16, 10),
) -> "Figure":
    """Template frequency comparison faceted by hour bin.

    Parameters
    ----------
    reservoir :
        A :class:`~autoslo.tuner.reservoir.QueryReservoir`.
    sampled_workload_dir :
        Directory containing sampled workload parquets.
    target_start, target_end :
        Forecast window.
    top_k :
        Number of top templates per bin.
    max_hours :
        Maximum number of hour bins to show (evenly spaced selection).
    """
    _require_mpl()

    hours_range = []
    current = target_start.replace(minute=0, second=0, microsecond=0)
    while current < target_end:
        hours_range.append(current)
        current += timedelta(hours=1)

    if len(hours_range) > max_hours:
        step = len(hours_range) // max_hours
        hours_range = hours_range[::step][:max_hours]

    # Load all sampled workloads.
    parquets = sorted(sampled_workload_dir.glob("*.parquet"))
    sampled_dfs = [pd.read_parquet(p) for p in parquets if p.stat().st_size > 0]

    n_bins = len(hours_range)
    cols = min(3, n_bins)
    rows = (n_bins + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for i, h_start in enumerate(hours_range):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        dow = h_start.weekday()
        hour = h_start.hour

        # Reservoir frequencies for this bin.
        bin_df = reservoir.bin_df(dow, hour)
        res_counts = bin_df["query_text_id"].value_counts()
        res_freq = res_counts / res_counts.sum() if len(res_counts) > 0 else res_counts

        # Sampled frequencies for this bin.
        h_end = h_start + timedelta(hours=1)
        all_qtids = []
        for sdf in sampled_dfs:
            if "abs_start_time" in sdf.columns:
                mask = (sdf["abs_start_time"] >= h_start) & (sdf["abs_start_time"] < h_end)
                all_qtids.append(sdf.loc[mask, "query_text_id"])
        if all_qtids:
            combined = pd.concat(all_qtids)
            samp_counts = combined.value_counts()
            samp_freq = samp_counts / samp_counts.sum() if len(samp_counts) > 0 else samp_counts
        else:
            samp_freq = pd.Series(dtype=float)

        top_ids = res_freq.head(top_k).index.tolist()
        short_labels = [str(t)[-6:] for t in top_ids]
        rv = [res_freq.get(t, 0.0) for t in top_ids]
        sv = [samp_freq.get(t, 0.0) for t in top_ids]

        x = np.arange(len(top_ids))
        ax.bar(x - 0.2, rv, 0.4, label="Reservoir" if i == 0 else "", color="steelblue")
        ax.bar(x + 0.2, sv, 0.4, label="Sampled" if i == 0 else "", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6)
        ax.set_title(h_start.strftime("%a %H:00"), fontsize=9)

    # Hide unused subplots.
    for i in range(n_bins, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Per-Hour Template Frequency: Reservoir vs. Sampled", fontsize=12)
    handles = [
        mpatches.Patch(color="steelblue", label="Reservoir"),
        mpatches.Patch(color="coral", label="Sampled"),
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return fig


# ======================================================================
# V1.4 — Forecast Policy Weight Decay
# ======================================================================


def plot_forecast_weight_decay(
    half_life_days: float = 14.0,
    dow_boost: float = 2.0,
    max_days: int = 60,
    figsize: tuple[float, float] = (8, 4),
) -> "Figure":
    """Visualise the recency-weighted forecast policy's weight decay.

    Parameters
    ----------
    half_life_days :
        Exponential half-life in days.
    dow_boost :
        Multiplicative boost for same day-of-week observations.
    max_days :
        Range of the x-axis (days ago).
    """
    _require_mpl()

    days = np.arange(0, max_days + 1, dtype=float)
    base_weight = np.exp(-np.log(2) * days / half_life_days)
    boosted_weight = base_weight * dow_boost

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(days, base_weight, "-", color="steelblue", label="Different weekday")
    ax.plot(days, boosted_weight, "--", color="#E07022", label=f"Same weekday (×{dow_boost})")
    ax.axvline(half_life_days, color="gray", linestyle=":", alpha=0.6,
               label=f"Half-life = {half_life_days} days")
    ax.set_xlabel("Days ago")
    ax.set_ylabel("Weight")
    ax.set_title("Recency-Weighted Forecast Policy: Weight Decay")
    ax.legend()
    ax.set_xlim(0, max_days)
    ax.set_ylim(0)
    fig.tight_layout()
    return fig


# ======================================================================
# V2.1 — Per-Scenario Metric Distributions
# ======================================================================


def plot_scenario_distributions(
    summary_path: Path,
    title: str = "Per-Scenario Metric Distributions",
    figsize: tuple[float, float] = (14, 8),
) -> "Figure":
    """Violin / strip plot of per-scenario metrics from a phase summary.

    Parameters
    ----------
    summary_path :
        Path to a ``summary.yml`` file (baseline or final).
    """
    _require_mpl()

    data = _load_yaml(summary_path)
    if not data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Summary not found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    metrics = ["violation_rate", "violation_amount_s", "violation_relative_mean", "total_cost"]
    labels = ["Viol. Rate", "Viol. Amount (s)", "Viol. Relative", "Cost ($)"]

    # Collect series.
    series: dict[str, dict[str, list[float]]] = {}
    for split_key, split_label in [("train_scenarios", "Train"), ("val_scenarios", "Val")]:
        scenarios = data.get(split_key, [])
        if scenarios:
            series[split_label] = {}
            for m in metrics:
                series[split_label][m] = [s.get(m, 0.0) for s in scenarios]

    if not series:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No scenario data in summary", ha="center", va="center")
        ax.set_axis_off()
        return fig

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)

    split_colors = {"Train": "steelblue", "Val": "coral"}

    for col, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[0][col]
        positions = []
        all_data = []
        tick_labels = []
        colors = []
        pos = 0
        for split_label, split_data in series.items():
            vals = split_data[metric]
            all_data.append(vals)
            positions.append(pos)
            tick_labels.append(split_label)
            colors.append(split_colors.get(split_label, "gray"))
            pos += 1

        parts = ax.violinplot(all_data, positions=positions, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.4)
        # Overlay individual points (jittered).
        for i, vals in enumerate(all_data):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(
                np.full(len(vals), positions[i]) + jitter, vals,
                s=12, color=colors[i], alpha=0.7, zorder=3,
            )
            # Aggregate marker (e.g. p90).
            if vals:
                p90 = float(np.quantile(vals, 0.9))
                ax.axhline(p90, color=colors[i], linestyle="--", alpha=0.5, linewidth=0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels)
        ax.set_title(label, fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ======================================================================
# V2.2 — Per-Scenario Cost Breakdown by Cluster
# ======================================================================


def plot_cost_breakdown_by_cluster(
    scenario_dirs: list[Path],
    title: str = "Cost Breakdown by Cluster",
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Stacked bar chart of per-cluster cost across scenarios.

    Parameters
    ----------
    scenario_dirs :
        List of per-scenario output directories (each containing
        ``billing_interval_analysis.yml``).
    """
    _require_mpl()

    cluster_costs: list[dict[str, float]] = []
    all_clusters: set[str] = set()
    for d in scenario_dirs:
        billing_path = d / "billing_interval_analysis.yml"
        billing = _load_yaml(billing_path)
        costs: dict[str, float] = {}
        if billing:
            for cname, cdata in billing.items():
                c = float(cdata.get("total_billed_cost", 0.0)) if isinstance(cdata, dict) else 0.0
                costs[cname] = c
                all_clusters.add(cname)
        cluster_costs.append(costs)

    if not cluster_costs or not all_clusters:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No billing data found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    sorted_clusters = sorted(all_clusters)
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(cluster_costs))
    bottom = np.zeros(len(cluster_costs))
    for i, cname in enumerate(sorted_clusters):
        vals = [cc.get(cname, 0.0) for cc in cluster_costs]
        ax.bar(x, vals, bottom=bottom, label=cname, color=cmap(i % 10), edgecolor="white")
        bottom += np.array(vals)

    ax.set_xlabel("Scenario index")
    ax.set_ylabel("Cost ($)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=min(4, len(sorted_clusters)))
    fig.tight_layout()
    return fig


# ======================================================================
# V3.1 — Violation-Window Timeline
# ======================================================================


def plot_violation_window_timeline(
    run_dir: Path,
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Violation windows as a heatmap row with checkpoint markers.

    Reads checkpoint round summaries and ``selected_checkpoints.yml``.

    Parameters
    ----------
    run_dir :
        Root tuner run directory.
    """
    _require_mpl()
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    ckpt_dir = run_dir / "checkpoints"

    # Load selected checkpoints.
    sel_path = ckpt_dir / "selected_checkpoints.yml"
    selected_cps: list[dict] = []
    if sel_path.exists():
        with open(sel_path) as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, list):
            selected_cps = raw

    # Collect violation windows from each round's base evaluation.
    # We'll re-derive them from structured logs if available, but for
    # simplicity use the candidate_results if present.
    round_dirs = sorted(ckpt_dir.glob("round_*"))

    # Gather all window-level violation data from the base structured logs.
    all_windows: list[tuple[float, float, float]] = []  # (start_s, end_s, violation_rate)
    for rd in round_dirs:
        base_dir = rd / "base"
        if not base_dir.exists():
            continue
        # Read structured logs from all scenarios in this round's base eval.
        scenario_windows: dict[float, list[float]] = {}  # start_s -> [rates across scenarios]
        for sd in sorted(base_dir.iterdir()):
            log_path = sd / "structured_log.parquet"
            if not log_path.exists():
                continue
            try:
                slog = pd.read_parquet(log_path)
            except Exception:
                continue
            completions = slog[slog["event_type"] == "completion"].copy()
            if completions.empty:
                continue
            # We need slo_s — read from run config.
            init_cfg = _load_yaml(run_dir / "initial_config.yml") or {}
            slo_s = float((init_cfg.get("slo_config") or {}).get("slo_s", 10.0))
            tuner_cfg = _load_yaml(run_dir / "tuner_config.yml") or {}
            window_s = float(tuner_cfg.get("sliding_window_s", 300.0))

            completions["window_start"] = (
                np.floor(completions["timestamp"].astype(float) / window_s) * window_s
            )
            for ws, grp in completions.groupby("window_start"):
                ws_f = float(ws)
                viol_rate = float((grp["latency_s"].astype(float) > slo_s).mean())
                scenario_windows.setdefault(ws_f, []).append(viol_rate)

        # Average across scenarios.
        for ws, rates in scenario_windows.items():
            avg_rate = float(np.mean(rates))
            all_windows.append((ws, ws + 300.0, avg_rate))
        break  # Only use round 0 base for the timeline.

    if not all_windows and not selected_cps:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No checkpoint data found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    if all_windows:
        all_windows.sort(key=lambda w: w[0])
        norm = Normalize(vmin=0, vmax=max(w[2] for w in all_windows) or 1.0)
        cmap = plt.cm.Reds
        for start, end, rate in all_windows:
            ax.barh(0, end - start, left=start, height=0.5,
                    color=cmap(norm(rate)), edgecolor="white", linewidth=0.5)
        # Colorbar.
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Violation rate", shrink=0.6)

    # Overlay checkpoint markers.
    for i, cp in enumerate(selected_cps):
        t = cp.get("time_s", 0)
        rpus = cp.get("min_rpus", [])
        ax.axvline(t, color="#3A9D5D", linestyle="--", linewidth=2, zorder=5)
        ax.text(t, 0.35, f"RPU {rpus}", fontsize=7, ha="center",
                color="#3A9D5D", fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("Simulation time (s)")
    ax.set_title("Violation Windows + Selected Checkpoints")
    fig.tight_layout()
    return fig


# ======================================================================
# V3.2 — Checkpoint-Round Trajectory
# ======================================================================


def plot_checkpoint_round_trajectory(
    run_dir: Path,
    figsize: tuple[float, float] = _FIGSIZE_MEDIUM,
) -> "Figure":
    """Line chart of validation violation & cost across checkpoint rounds.

    Parameters
    ----------
    run_dir :
        Root tuner run directory.
    """
    _require_mpl()

    ckpt_dir = run_dir / "checkpoints"
    round_dirs = sorted(ckpt_dir.glob("round_*"))

    rounds: list[int] = []
    val_violations: list[float] = []
    val_costs: list[float] = []
    selected_rpus: list[str] = []

    for rd in round_dirs:
        summary_path = rd / "candidate_results.yml"
        data = _load_yaml(summary_path)
        if not data:
            continue
        idx = int(rd.name.split("_")[-1])
        rounds.append(idx)
        val_violations.append(float(data.get("val_violation", 0)))
        val_costs.append(float(data.get("val_cost", 0)))
        cp = data.get("selected_checkpoint", {})
        rpus = cp.get("min_rpus", [])
        selected_rpus.append(str(rpus))

    if not rounds:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No checkpoint rounds found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # SLO threshold for reference line.
    init_cfg = _load_yaml(run_dir / "initial_config.yml") or {}
    slo_threshold = (init_cfg.get("slo_config") or {}).get("slo_threshold")

    fig, ax1 = plt.subplots(figsize=figsize)
    color_v = "steelblue"
    color_c = "coral"

    ax1.plot(rounds, val_violations, "o-", color=color_v, label="Val violation")
    ax1.set_xlabel("Checkpoint round")
    ax1.set_ylabel("Validation violation", color=color_v)
    ax1.tick_params(axis="y", labelcolor=color_v)

    if slo_threshold is not None:
        ax1.axhline(float(slo_threshold), color="green", linestyle=":",
                     alpha=0.6, label=f"SLO threshold ({slo_threshold})")

    # Annotate RPU selection.
    for i, (r, rpu_str) in enumerate(zip(rounds, selected_rpus)):
        ax1.annotate(rpu_str, (r, val_violations[i]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=7, ha="center")

    ax2 = ax1.twinx()
    ax2.plot(rounds, val_costs, "s--", color=color_c, alpha=0.7, label="Val cost ($)")
    ax2.set_ylabel("Validation cost ($)", color=color_c)
    ax2.tick_params(axis="y", labelcolor=color_c)

    # Combined legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title("Checkpoint Optimization Trajectory")
    ax1.set_xticks(rounds)
    fig.tight_layout()
    return fig


# ======================================================================
# V3.3 — RPU Candidate Comparison per Round
# ======================================================================


def plot_rpu_candidate_comparison(
    run_dir: Path,
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Grouped bar chart comparing RPU candidates per checkpoint round.

    Parameters
    ----------
    run_dir :
        Root tuner run directory.
    """
    _require_mpl()

    ckpt_dir = run_dir / "checkpoints"
    round_dirs = sorted(ckpt_dir.glob("round_*"))

    round_data: list[tuple[int, list[dict], dict]] = []
    for rd in round_dirs:
        data = _load_yaml(rd / "candidate_results.yml")
        if not data or "candidates" not in data:
            continue
        idx = int(rd.name.split("_")[-1])
        round_data.append((idx, data["candidates"], data.get("selected_checkpoint", {})))

    if not round_data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No candidate data found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    n_rounds = len(round_data)
    fig, axes = plt.subplots(1, n_rounds, figsize=figsize, squeeze=False)

    for col, (ridx, candidates, sel_cp) in enumerate(round_data):
        ax = axes[0][col]
        rpus = [c.get("rpu", 0) for c in candidates]
        violations = [c.get("train_violation", 0) for c in candidates]
        costs = [c.get("train_cost", 0) for c in candidates]
        sel_rpus = sel_cp.get("min_rpus", [])

        x = np.arange(len(rpus))
        width = 0.35
        ax.bar(x - width / 2, violations, width, label="Violation" if col == 0 else "",
               color="steelblue")
        ax_cost = ax.twinx()
        ax_cost.bar(x + width / 2, costs, width, label="Cost ($)" if col == 0 else "",
                    color="coral", alpha=0.7)

        # Highlight selected.
        for i, rpu in enumerate(rpus):
            if [rpu] == sel_rpus or (rpu,) == tuple(sel_rpus):
                ax.get_children()[i].set_edgecolor("green")
                ax.get_children()[i].set_linewidth(2)

        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rpus])
        ax.set_xlabel("RPU size")
        ax.set_title(f"Round {ridx}", fontsize=10)
        if col == 0:
            ax.set_ylabel("Violation")
        if col == n_rounds - 1:
            ax_cost.set_ylabel("Cost ($)")

    fig.suptitle("RPU Candidate Comparison per Checkpoint Round", fontsize=12)
    fig.tight_layout()
    return fig


# ======================================================================
# V4.1 — Sweep Scatter with Pareto Curve
# ======================================================================


def plot_sweep_pareto(
    sweep_results_path: Path,
    phase_name: str = "Sweep",
    slo_threshold: float | None = None,
    figsize: tuple[float, float] = _FIGSIZE_SQUARE,
) -> "Figure":
    """2-D scatter of grid points with Pareto frontier highlighted.

    Parameters
    ----------
    sweep_results_path :
        Path to ``sweep_results.json``.
    phase_name :
        Label for the plot title (e.g. ``"Autoscaler"``).
    slo_threshold :
        If set, shade the feasible region.
    """
    _require_mpl()

    with open(sweep_results_path) as f:
        data = json.load(f)

    grid_results = data.get("grid_results", [])
    best_idx = data.get("best_grid_point")

    if not grid_results:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No sweep results", ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # All grid points.
    all_viols = [r["train_violation_agg"] for r in grid_results]
    all_costs = [r["train_cost_agg"] for r in grid_results]
    ax.scatter(all_viols, all_costs, s=20, color="#D3D3D3", alpha=0.6, label="Grid points", zorder=2)

    # Pareto-optimal points.
    pareto_indices = [i for i, r in enumerate(grid_results) if r.get("is_pareto", False)]
    if pareto_indices:
        p_viols = [grid_results[i]["train_violation_agg"] for i in pareto_indices]
        p_costs = [grid_results[i]["train_cost_agg"] for i in pareto_indices]
        ax.scatter(p_viols, p_costs, s=50, color="#3466FF", zorder=3, label="Pareto front")

        # Connect Pareto points with a step line.
        sorted_pareto = sorted(zip(p_viols, p_costs))
        px, py = zip(*sorted_pareto)
        ax.step(px, py, where="post", color="#3466FF", linewidth=1, alpha=0.5)

        # Annotate Pareto points with abbreviated params.
        for i in pareto_indices:
            r = grid_results[i]
            params = r.get("params", {})
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            if len(param_str) > 30:
                param_str = param_str[:27] + "…"
            ax.annotate(
                param_str,
                (r["train_violation_agg"], r["train_cost_agg"]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=6, alpha=0.8,
            )

    # Selected best.
    if best_idx is not None and 0 <= best_idx < len(grid_results):
        br = grid_results[best_idx]
        ax.scatter(
            [br["train_violation_agg"]], [br["train_cost_agg"]],
            s=120, marker="*", color="#E07022", zorder=4, label="Selected",
        )

    # Feasibility band.
    if slo_threshold is not None:
        ax.axvspan(0, slo_threshold, color="#d4edda", alpha=0.3, zorder=0,
                   label=f"Feasible (≤{slo_threshold:.2f})")

    ax.set_xlabel("Primary violation")
    ax.set_ylabel("Cost ($)")
    ax.set_title(f"{phase_name} — Sweep Grid with Pareto Front")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ======================================================================
# V4.2 — Sweep Heatmap (2 params)
# ======================================================================


def plot_sweep_heatmap(
    sweep_results_path: Path,
    phase_name: str = "Sweep",
    metric: str = "train_violation_agg",
    figsize: tuple[float, float] = _FIGSIZE_MEDIUM,
) -> "Figure | None":
    """2-D heatmap when the sweep has exactly two parameters.

    Parameters
    ----------
    sweep_results_path :
        Path to ``sweep_results.json``.
    phase_name :
        Title label.
    metric :
        Which field to colour by.

    Returns ``None`` if the sweep has ≠ 2 parameters.
    """
    _require_mpl()

    with open(sweep_results_path) as f:
        data = json.load(f)

    grid_results = data.get("grid_results", [])
    best_idx = data.get("best_grid_point")

    if not grid_results:
        return None

    # Check that we have exactly 2 params.
    all_params = [r.get("params", {}) for r in grid_results]
    param_names = sorted(all_params[0].keys()) if all_params else []
    if len(param_names) != 2:
        return None

    p1, p2 = param_names
    p1_vals = sorted(set(p[p1] for p in all_params))
    p2_vals = sorted(set(p[p2] for p in all_params))

    grid = np.full((len(p2_vals), len(p1_vals)), np.nan)
    pareto_cells: list[tuple[int, int]] = []
    best_cell: tuple[int, int] | None = None

    for i, r in enumerate(grid_results):
        p = r["params"]
        c = p1_vals.index(p[p1])
        rr = p2_vals.index(p[p2])
        grid[rr, c] = r.get(metric, np.nan)
        if r.get("is_pareto"):
            pareto_cells.append((rr, c))
        if i == best_idx:
            best_cell = (rr, c)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="lower")
    ax.set_xticks(range(len(p1_vals)))
    ax.set_xticklabels([str(v) for v in p1_vals], fontsize=8)
    ax.set_yticks(range(len(p2_vals)))
    ax.set_yticklabels([str(v) for v in p2_vals], fontsize=8)
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)

    # Mark Pareto cells.
    for rr, cc in pareto_cells:
        rect = plt.Rectangle((cc - 0.5, rr - 0.5), 1, 1,
                              fill=False, edgecolor="blue", linewidth=2)
        ax.add_patch(rect)

    # Mark selected best.
    if best_cell:
        ax.plot(best_cell[1], best_cell[0], marker="*", color="#E07022",
                markersize=15, zorder=5)

    fig.colorbar(im, ax=ax, label=metric.replace("_", " ").title())
    ax.set_title(f"{phase_name} — {metric.replace('_', ' ').title()}")
    fig.tight_layout()
    return fig


# ======================================================================
# V4.3 — Train vs. Validation Agreement
# ======================================================================


def plot_train_val_agreement(
    sweep_results_path: Path,
    phase_name: str = "Sweep",
    figsize: tuple[float, float] = (6, 6),
) -> "Figure":
    """Scatter of train vs. val violation for Pareto points.

    Parameters
    ----------
    sweep_results_path :
        Path to ``sweep_results.json``.
    """
    _require_mpl()

    with open(sweep_results_path) as f:
        data = json.load(f)

    grid_results = data.get("grid_results", [])
    pareto_points = [
        r for r in grid_results
        if r.get("is_pareto") and r.get("val_violation_agg") is not None
    ]

    if not pareto_points:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No validated Pareto points", ha="center", va="center")
        ax.set_axis_off()
        return fig

    train_v = [r["train_violation_agg"] for r in pareto_points]
    val_v = [r["val_violation_agg"] for r in pareto_points]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(train_v, val_v, s=50, color="#3466FF")

    # y=x reference.
    lo = min(min(train_v), min(val_v))
    hi = max(max(train_v), max(val_v))
    margin = (hi - lo) * 0.1 or 0.01
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.3, label="y=x")

    ax.set_xlabel("Train violation")
    ax.set_ylabel("Val violation")
    ax.set_title(f"{phase_name} — Train vs. Validation Agreement")
    ax.legend()
    fig.tight_layout()
    return fig


# ======================================================================
# V5.1 — Cluster Lifecycle Gantt Chart
# ======================================================================


def plot_cluster_gantt(
    structured_log_path: Path,
    title: str = "Cluster Lifecycle",
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Gantt chart of cluster lifetimes from a structured log.

    Parameters
    ----------
    structured_log_path :
        Path to ``structured_log.parquet`` from one scenario.
    """
    _require_mpl()

    slog = pd.read_parquet(structured_log_path)

    # Extract cluster lifecycle events.
    spin_up = slog[slog["event_type"] == "spin_up_scheduled"][["timestamp", "cluster_name", "rpu"]].copy()
    ready = slog[slog["event_type"] == "cluster_ready"][["timestamp", "cluster_name"]].copy()
    tear_start = slog[slog["event_type"] == "tear_down_requested"][["timestamp", "cluster_name"]].copy()
    deactivated = slog[slog["event_type"] == "cluster_deactivated"][["timestamp", "cluster_name"]].copy()

    # Build per-cluster records.
    clusters: dict[str, dict[str, Any]] = {}
    for _, row in spin_up.iterrows():
        cname = row["cluster_name"]
        clusters.setdefault(cname, {})
        clusters[cname]["spin_up"] = float(row["timestamp"])
        clusters[cname]["rpu"] = row.get("rpu", "?")

    for _, row in ready.iterrows():
        cname = row["cluster_name"]
        clusters.setdefault(cname, {})
        clusters[cname]["ready"] = float(row["timestamp"])

    for _, row in tear_start.iterrows():
        cname = row["cluster_name"]
        if cname in clusters:
            clusters[cname]["tear_start"] = float(row["timestamp"])

    for _, row in deactivated.iterrows():
        cname = row["cluster_name"]
        if cname in clusters:
            clusters[cname]["deactivated"] = float(row["timestamp"])

    if not clusters:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No cluster lifecycle events found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Determine simulation end for clusters without tear_down.
    sim_end = float(slog["timestamp"].max())

    sorted_names = sorted(clusters.keys())
    cmap = plt.cm.Set2

    fig, ax = plt.subplots(figsize=figsize)

    for i, cname in enumerate(sorted_names):
        c = clusters[cname]
        t_spin = c.get("spin_up", 0)
        t_ready = c.get("ready", t_spin)
        t_tear = c.get("tear_start", sim_end)
        t_deact = c.get("deactivated", t_tear)
        rpu = c.get("rpu", "?")

        color = cmap(i % 8)

        # Spin-up latency (hatched).
        if t_ready > t_spin:
            ax.barh(i, t_ready - t_spin, left=t_spin, height=0.6,
                    color=color, alpha=0.3, hatch="//", edgecolor=color, linewidth=0.5)
        # Active period.
        ax.barh(i, t_tear - t_ready, left=t_ready, height=0.6,
                color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        # Draining phase (dashed border).
        if t_deact > t_tear:
            ax.barh(i, t_deact - t_tear, left=t_tear, height=0.6,
                    color=color, alpha=0.2, edgecolor=color, linestyle="--", linewidth=1)

        ax.text(t_spin + 2, i, f"RPU {rpu}", fontsize=7, va="center")

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Simulation time (s)")
    ax.set_title(title)

    # Legend.
    legend_patches = [
        mpatches.Patch(facecolor="gray", alpha=0.3, hatch="//", label="Spinning up"),
        mpatches.Patch(facecolor="gray", alpha=0.8, label="Active"),
        mpatches.Patch(facecolor="gray", alpha=0.2, linestyle="--", label="Draining"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ======================================================================
# V5.2 — Headroom & Active-Queries Time-Series
# ======================================================================


def plot_headroom_timeseries(
    structured_log_path: Path,
    eta_crit: float | None = None,
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Headroom, active queries, and active clusters over time.

    Parameters
    ----------
    structured_log_path :
        Path to ``structured_log.parquet``.
    eta_crit :
        If set, shade the danger zone where headroom ≤ eta_crit.
    """
    _require_mpl()

    slog = pd.read_parquet(structured_log_path)
    ticks = slog[slog["event_type"] == "capacity_tick"].copy()

    if ticks.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No capacity_tick events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    t = ticks["timestamp"].astype(float).values
    headroom = ticks["headroom"].astype(float).values
    n_queries = ticks["num_active_queries"].astype(float).values
    n_clusters = ticks["num_active_clusters"].astype(float).values

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(t, headroom, "-", color="steelblue", label="Headroom", linewidth=1)
    ax1.set_xlabel("Simulation time (s)")
    ax1.set_ylabel("Headroom", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    if eta_crit is not None:
        ax1.axhline(eta_crit, color="red", linestyle=":", alpha=0.5, label=f"η_crit = {eta_crit}")
        ax1.fill_between(t, 0, eta_crit, alpha=0.05, color="red")

    # Overlay spin-up events.
    spin_ups = slog[slog["event_type"] == "spin_up_scheduled"]
    for _, su in spin_ups.iterrows():
        ax1.axvline(float(su["timestamp"]), color="green", linestyle="--",
                     alpha=0.3, linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(t, n_queries, "-", color="coral", alpha=0.7, label="Active queries", linewidth=0.8)
    ax2.plot(t, n_clusters, "-", color="#E07022", alpha=0.7, label="Active clusters", linewidth=0.8)
    ax2.set_ylabel("Count", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title("Headroom & Cluster Activity")
    fig.tight_layout()
    return fig


# ======================================================================
# V5.3 — Per-Query Latency vs. SLO
# ======================================================================


def plot_latency_vs_slo(
    structured_log_path: Path,
    slo_s: float = 10.0,
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Scatter of per-query latency against the SLO line.

    Parameters
    ----------
    structured_log_path :
        Path to ``structured_log.parquet``.
    slo_s :
        SLO threshold in seconds.
    """
    _require_mpl()

    slog = pd.read_parquet(structured_log_path)
    completions = slog[slog["event_type"] == "completion"].copy()

    if completions.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No completion events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    t = completions["timestamp"].astype(float).values
    lat = completions["latency_s"].astype(float).values
    violated = lat > slo_s

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(t[~violated], lat[~violated], s=8, color="steelblue", alpha=0.5, label="Within SLO")
    ax.scatter(t[violated], lat[violated], s=12, color="#C9302C", alpha=0.7, label="Violation")
    ax.axhline(slo_s, color="green", linestyle="--", linewidth=1.5, label=f"SLO = {slo_s}s")

    ax.set_xlabel("Completion time (s)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Per-Query Latency vs. SLO")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ======================================================================
# V5.4 — Routing Distribution Over Time
# ======================================================================


def plot_routing_distribution(
    structured_log_path: Path,
    bin_s: float = 60.0,
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Stacked area chart of query routing distribution across clusters.

    Parameters
    ----------
    structured_log_path :
        Path to ``structured_log.parquet``.
    bin_s :
        Time bin width in seconds.
    """
    _require_mpl()

    slog = pd.read_parquet(structured_log_path)
    routing = slog[slog["event_type"] == "routing"].copy()

    if routing.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No routing events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    routing["time_bin"] = np.floor(routing["timestamp"].astype(float) / bin_s) * bin_s
    pivot = routing.groupby(["time_bin", "cluster_name"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.tab10
    clusters = pivot.columns.tolist()
    x = pivot.index.values

    bottom = np.zeros(len(x))
    for i, cname in enumerate(clusters):
        vals = pivot[cname].values.astype(float)
        ax.fill_between(x, bottom, bottom + vals, alpha=0.7,
                        color=cmap(i % 10), label=cname)
        bottom += vals

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Queries routed")
    ax.set_title("Routing Distribution Over Time")
    ax.legend(fontsize=7, ncol=min(4, len(clusters)))
    fig.tight_layout()
    return fig


# ======================================================================
# V5.5 — Billing Utilisation Gantt
# ======================================================================


def plot_billing_utilisation(
    billing_path: Path,
    figsize: tuple[float, float] = _FIGSIZE_WIDE,
) -> "Figure":
    """Thin horizontal bars for billed intervals, coloured by utilisation.

    Parameters
    ----------
    billing_path :
        Path to ``billing_interval_analysis.yml``.
    """
    _require_mpl()
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    data = _load_yaml(billing_path)
    if not data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No billing data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    sorted_clusters = sorted(data.keys())
    max_queries = 1
    for cname in sorted_clusters:
        cd = data[cname]
        if isinstance(cd, dict):
            for bi in cd.get("billed_intervals", []):
                nq = len(bi.get("query_ids", []))
                if nq > max_queries:
                    max_queries = nq

    norm = Normalize(vmin=0, vmax=max_queries)
    cmap = plt.cm.YlOrRd

    fig, ax = plt.subplots(figsize=figsize)

    for i, cname in enumerate(sorted_clusters):
        cd = data[cname]
        if not isinstance(cd, dict):
            continue
        for bi in cd.get("billed_intervals", []):
            begin = float(bi.get("begin_s", 0))
            end = float(bi.get("end_s", begin + 60))
            nq = len(bi.get("query_ids", []))
            ax.barh(i, end - begin, left=begin, height=0.6,
                    color=cmap(norm(nq)), edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(sorted_clusters)))
    ax.set_yticklabels(sorted_clusters, fontsize=8)
    ax.set_xlabel("Simulation time (s)")
    ax.set_title("Billing Intervals — Utilisation")
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Queries in interval")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ======================================================================
# V6.1 — Tuner Evolution Strip Chart
# ======================================================================


def plot_evolution_strip(
    evolution_path: Path,
    primary_metric: str = "violation_rate",
    figsize: tuple[float, float] = (16, 6),
) -> "Figure":
    """Strip chart of scenario results across all tuner phases.

    Parameters
    ----------
    evolution_path :
        Path to ``evolution.parquet``.
    primary_metric :
        Column to plot on the y-axis.
    """
    _require_mpl()

    df = pd.read_parquet(evolution_path)
    if primary_metric not in df.columns:
        primary_metric = "violation_rate"

    # Phase ordering.
    phase_order = ["baseline", "checkpoints", "autoscaler", "routing", "final", "holdout"]
    df["phase"] = pd.Categorical(df["phase"], categories=phase_order, ordered=True)
    df = df.sort_values(["phase", "grid_point"])

    # Create a combined label for x-axis grouping.
    df["group"] = df["phase"].astype(str) + "\n" + df["grid_point"].astype(str)
    groups = df["group"].unique()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top: violation metric.
    ax = axes[0]
    group_positions: dict[str, int] = {g: i for i, g in enumerate(groups)}
    for _, row in df.iterrows():
        x = group_positions[row["group"]]
        jitter = np.random.default_rng(int(row.get("scenario_idx", 0))).uniform(-0.2, 0.2)
        ax.scatter(x + jitter, row[primary_metric], s=10, alpha=0.5, color="steelblue")

    # Overlay aggregate per group.
    for g, gdf in df.groupby("group", observed=True):
        x = group_positions[g]
        vals = gdf[primary_metric].dropna()
        if len(vals) > 0:
            p90 = float(np.quantile(vals, 0.9))
            ax.plot(x, p90, "_", color="red", markersize=12, markeredgewidth=2)

    ax.set_ylabel(primary_metric.replace("_", " ").title())
    ax.set_title("Tuner Evolution — Per-Scenario Results")

    # Bottom: cost.
    ax2 = axes[1]
    for _, row in df.iterrows():
        x = group_positions[row["group"]]
        jitter = np.random.default_rng(int(row.get("scenario_idx", 0)) + 999).uniform(-0.2, 0.2)
        ax2.scatter(x + jitter, row.get("total_cost", 0), s=10, alpha=0.5, color="coral")

    for g, gdf in df.groupby("group", observed=True):
        x = group_positions[g]
        vals = gdf["total_cost"].dropna()
        if len(vals) > 0:
            p90 = float(np.quantile(vals, 0.9))
            ax2.plot(x, p90, "_", color="red", markersize=12, markeredgewidth=2)

    ax2.set_ylabel("Cost ($)")
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels(groups, rotation=45, ha="right", fontsize=6)

    # Draw vertical separators between phases.
    prev_phase = None
    for g in groups:
        phase = g.split("\n")[0]
        if prev_phase is not None and phase != prev_phase:
            x_pos = group_positions[g] - 0.5
            for a in axes:
                a.axvline(x_pos, color="gray", linestyle=":", alpha=0.4)
        prev_phase = phase

    fig.tight_layout()
    return fig


# ======================================================================
# V6.2 — Holdout Comparison Bar Chart
# ======================================================================


def plot_holdout_comparison(
    holdout_summary_path: Path,
    slo_threshold: float | None = None,
    figsize: tuple[float, float] = _FIGSIZE_MEDIUM,
) -> "Figure":
    """Grouped bar chart comparing baseline, tuned, and static baselines.

    Parameters
    ----------
    holdout_summary_path :
        Path to ``holdout/summary.yml``.
    slo_threshold :
        If set, draw a threshold line on the violation axis.
    """
    _require_mpl()

    data = _load_yaml(holdout_summary_path)
    if not data:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Holdout summary not found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    entries: list[tuple[str, float, float]] = []  # (label, violation, cost)

    bv = data.get("baseline_violation")
    bc = data.get("baseline_cost")
    if bv is not None and bc is not None:
        entries.append(("Baseline", float(bv), float(bc)))

    tv = data.get("tuned_violation")
    tc = data.get("tuned_cost")
    if tv is not None and tc is not None:
        entries.append(("Tuned", float(tv), float(tc)))

    for sb in data.get("static_baselines", []):
        entries.append((sb.get("label", "Static"), float(sb.get("violation", 0)),
                        float(sb.get("cost", 0))))

    if not entries:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No holdout data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels = [e[0] for e in entries]
    violations = [e[1] for e in entries]
    costs = [e[2] for e in entries]

    fig, ax1 = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.35

    bars_v = ax1.bar(x - width / 2, violations, width, label="Violation", color="steelblue")
    ax1.set_ylabel("Violation", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    if slo_threshold is not None:
        ax1.axhline(slo_threshold, color="green", linestyle="--", alpha=0.6,
                     label=f"Threshold ({slo_threshold:.2f})")

    ax2 = ax1.twinx()
    bars_c = ax2.bar(x + width / 2, costs, width, label="Cost ($)", color="coral")
    ax2.set_ylabel("Cost ($)", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Holdout: Baseline vs. Tuned vs. Static")
    fig.tight_layout()
    return fig


# ======================================================================
# V6.3 — Phase-wise Improvement Waterfall
# ======================================================================


def plot_phase_waterfall(
    run_dir: Path,
    metric: str = "violation",
    figsize: tuple[float, float] = _FIGSIZE_MEDIUM,
) -> "Figure":
    """Waterfall chart showing incremental improvement per tuning phase.

    Parameters
    ----------
    run_dir :
        Root tuner run directory.
    metric :
        ``"violation"`` or ``"cost"``.
    """
    _require_mpl()

    # Read baseline and final summaries.
    baseline = _load_yaml(run_dir / "baseline" / "summary.yml")
    final = _load_yaml(run_dir / "final" / "summary.yml")

    if not baseline or not final:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Baseline or final summary missing", ha="center", va="center")
        ax.set_axis_off()
        return fig

    key = "val_violation_agg" if metric == "violation" else "val_cost_agg"
    fallback_key = "train_violation_agg" if metric == "violation" else "train_cost_agg"

    baseline_val = float(baseline.get(key) or baseline.get(fallback_key, 0))
    final_val = float(final.get(key) or final.get(fallback_key, 0))

    # Try to get intermediate phase results.
    phases = [("Baseline", baseline_val)]

    # Checkpoints: look for the last round's validation.
    ckpt_dir = run_dir / "checkpoints"
    ckpt_rounds = sorted(ckpt_dir.glob("round_*")) if ckpt_dir.exists() else []
    if ckpt_rounds:
        last_round = _load_yaml(ckpt_rounds[-1] / "candidate_results.yml")
        if last_round:
            ckpt_key = "val_violation" if metric == "violation" else "val_cost"
            ckpt_val = float(last_round.get(ckpt_key, baseline_val))
            phases.append(("Checkpoints", ckpt_val))

    # Autoscaler sweep.
    as_path = run_dir / "autoscaler" / "sweep_results.json"
    if as_path.exists():
        with open(as_path) as f:
            as_data = json.load(f)
        best = as_data.get("best_grid_point")
        if best is not None:
            gr = as_data["grid_results"][best]
            as_key = "val_violation_agg" if metric == "violation" else "val_cost_agg"
            as_val = gr.get(as_key, gr.get("train_violation_agg" if metric == "violation" else "train_cost_agg"))
            if as_val is not None:
                phases.append(("Autoscaler", float(as_val)))

    # Routing sweep.
    rt_path = run_dir / "routing" / "sweep_results.json"
    if rt_path.exists():
        with open(rt_path) as f:
            rt_data = json.load(f)
        best = rt_data.get("best_grid_point")
        if best is not None:
            gr = rt_data["grid_results"][best]
            rt_key = "val_violation_agg" if metric == "violation" else "val_cost_agg"
            rt_val = gr.get(rt_key, gr.get("train_violation_agg" if metric == "violation" else "train_cost_agg"))
            if rt_val is not None:
                phases.append(("Routing", float(rt_val)))

    phases.append(("Final", final_val))

    labels = [p[0] for p in phases]
    values = [p[1] for p in phases]
    deltas = [0] + [values[i] - values[i - 1] for i in range(1, len(values))]

    fig, ax = plt.subplots(figsize=figsize)

    running = values[0]
    for i in range(len(labels)):
        if i == 0:
            ax.bar(i, values[i], color="steelblue", edgecolor="white")
        elif i == len(labels) - 1:
            ax.bar(i, values[i], color="steelblue", edgecolor="white")
        else:
            delta = deltas[i]
            color = "#3A9D5D" if delta < 0 else "#C9302C"
            bottom = running + delta if delta < 0 else running
            height = abs(delta)
            ax.bar(i, height, bottom=bottom, color=color, edgecolor="white")
            running += delta

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    metric_label = "Violation" if metric == "violation" else "Cost ($)"
    ax.set_ylabel(metric_label)
    ax.set_title(f"Phase-wise Improvement Waterfall — {metric_label}")
    fig.tight_layout()
    return fig


# ======================================================================
# V7.1 — History-Window Sensitivity Curves
# ======================================================================


def plot_history_sensitivity(
    run_root: Path,
    scenarios: list[str] | None = None,
    labels: dict[str, str] | None = None,
    figsize: tuple[float, float] = _FIGSIZE_MEDIUM,
) -> "Figure":
    """Line chart of holdout metrics across history-window lengths.

    Parameters
    ----------
    run_root :
        Root directory containing sub-directories per scenario
        (e.g. ``prev_day/``, ``prev_week/``, ``prev_month/``).
    scenarios :
        Names of sub-directories to include (default: auto-detect).
    labels :
        Human-readable labels for each scenario.
    """
    _require_mpl()

    if scenarios is None:
        scenarios = sorted(
            [d.name for d in run_root.iterdir() if d.is_dir() and (d / "holdout" / "summary.yml").exists()]
        )
    if labels is None:
        labels = {s: s for s in scenarios}

    viols_baseline: list[float] = []
    viols_tuned: list[float] = []
    costs_baseline: list[float] = []
    costs_tuned: list[float] = []
    x_labels: list[str] = []

    for sc in scenarios:
        holdout = _load_yaml(run_root / sc / "holdout" / "summary.yml")
        if not holdout:
            continue
        x_labels.append(labels.get(sc, sc))
        viols_baseline.append(float(holdout.get("baseline_violation", 0)))
        viols_tuned.append(float(holdout.get("tuned_violation", 0)))
        costs_baseline.append(float(holdout.get("baseline_cost", 0)))
        costs_tuned.append(float(holdout.get("tuned_cost", 0)))

    if not x_labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No holdout data found", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x = np.arange(len(x_labels))
    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(x, viols_baseline, "o--", color="#D3D3D3", label="Baseline viol.")
    ax1.plot(x, viols_tuned, "o-", color="steelblue", label="Tuned viol.")
    ax1.set_ylabel("Violation", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, costs_baseline, "s--", color="#D3D3D3", alpha=0.6, label="Baseline cost")
    ax2.plot(x, costs_tuned, "s-", color="coral", label="Tuned cost")
    ax2.set_ylabel("Cost ($)", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("History length")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    ax1.set_title("History-Window Sensitivity")
    fig.tight_layout()
    return fig
