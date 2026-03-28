"""Visualization helpers for the tuner's reservoir, forecast, and sampled workloads."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional import — plots degrade gracefully if matplotlib is absent.
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend by default
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for tuner visualizations. "
            "Install it with: pip install matplotlib"
        )


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
