from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_trendline(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: object,
    x_log: bool,
    y_log: bool,
) -> None:
    """Fit a simple linear trend in transformed space and plot it."""
    if x.size < 2 or y.size < 2:
        return

    x_fit = np.log10(x) if x_log else x
    y_fit = np.log10(y) if y_log else y
    valid = np.isfinite(x_fit) & np.isfinite(y_fit)
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    if x_fit.size < 2 or np.unique(x_fit).size < 2:
        return

    try:
        slope, intercept = np.polyfit(x_fit, y_fit, 1)
    except Exception:
        return

    lo = float(np.min(x_fit))
    hi = float(np.max(x_fit))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    x_line_fit = np.linspace(lo, hi, 100)
    y_line_fit = slope * x_line_fit + intercept

    x_line = np.power(10.0, x_line_fit) if x_log else x_line_fit
    y_line = np.power(10.0, y_line_fit) if y_log else y_line_fit
    ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=1.5, alpha=0.9)


def _interarrival_by_query(arrivals_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-query interarrival times from query arrival timestamps.

    Interarrival is defined as current arrival minus previous arrival. For the
    first query in arrival order, we use the next arrival minus current arrival
    to avoid dropping that point.
    """
    if arrivals_df.empty:
        return pd.DataFrame(columns=["query_id", "interarrival_s"])

    base_cols = ["query_id", "arrival_s"]
    if "cluster_name" in arrivals_df.columns:
        base_cols.append("cluster_name")

    tmp = arrivals_df[base_cols].copy()
    tmp["query_id"] = tmp["query_id"].astype(str)
    tmp["arrival_s"] = pd.to_numeric(tmp["arrival_s"], errors="coerce")
    if "cluster_name" in tmp.columns:
        tmp["cluster_name"] = tmp["cluster_name"].fillna("").astype(str)
    else:
        tmp["cluster_name"] = "__all_clusters__"
    tmp = tmp.dropna(subset=["arrival_s"])
    if tmp.empty:
        return pd.DataFrame(columns=["query_id", "interarrival_s"])

    frames: list[pd.DataFrame] = []
    for _, cluster_tmp in tmp.groupby("cluster_name", observed=True):
        # Keep first arrival for each query id if duplicates exist.
        cluster_tmp = cluster_tmp.sort_values(["arrival_s", "query_id"]).drop_duplicates(
            subset=["query_id"], keep="first"
        )
        cluster_tmp = cluster_tmp.reset_index(drop=True)

        arr = cluster_tmp["arrival_s"].to_numpy(dtype=float)
        inter = np.empty_like(arr)
        if arr.size == 1:
            inter[0] = np.nan
        else:
            inter[1:] = arr[1:] - arr[:-1]
            inter[0] = arr[1] - arr[0]

        out = cluster_tmp[["query_id"]].copy()
        out["interarrival_s"] = inter
        frames.append(out)

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["query_id", "interarrival_s"])
    )


def _effective_concurrency_by_query(query_windows_df: pd.DataFrame) -> pd.DataFrame:
    """Compute avg number of other active queries over each query's lifetime."""
    out_cols = ["query_id", "effective_concurrency"]
    required = {"query_id", "arrival_s", "completion_s"}
    if not required.issubset(query_windows_df.columns):
        return pd.DataFrame(columns=out_cols)

    base_cols = ["query_id", "arrival_s", "completion_s"]
    if "cluster_name" in query_windows_df.columns:
        base_cols.append("cluster_name")

    queries = query_windows_df[base_cols].copy()
    queries["query_id"] = queries["query_id"].astype(str)
    queries["arrival_s"] = pd.to_numeric(queries["arrival_s"], errors="coerce")
    queries["completion_s"] = pd.to_numeric(
        queries["completion_s"], errors="coerce"
    )
    if "cluster_name" in queries.columns:
        queries["cluster_name"] = queries["cluster_name"].fillna("").astype(str)
    else:
        queries["cluster_name"] = "__all_clusters__"

    queries = (
        queries.dropna(subset=["arrival_s", "completion_s"])
        .query("completion_s > arrival_s")
        .sort_values(["cluster_name", "arrival_s", "completion_s", "query_id"])
        .drop_duplicates("query_id", keep="first")
        .reset_index(drop=True)
    )

    if queries.empty:
        return pd.DataFrame(columns=out_cols)

    frames: list[pd.DataFrame] = []
    for _, cluster_queries in queries.groupby("cluster_name", observed=True):
        cluster_queries = cluster_queries.reset_index(drop=True)
        cluster_starts = cluster_queries["arrival_s"].to_numpy(dtype=float)
        cluster_ends = cluster_queries["completion_s"].to_numpy(dtype=float)
        durations = cluster_ends - cluster_starts
        overlap_time = np.zeros(len(cluster_queries), dtype=float)

        events: list[tuple[float, str, int]] = []
        for i, (start_s, end_s) in enumerate(zip(cluster_starts, cluster_ends)):
            events.append((float(start_s), "start", i))
            events.append((float(end_s), "end", i))

        # End before start gives half-open intervals: [arrival, completion).
        events.sort(key=lambda e: (e[0], e[1] == "start"))

        active: set[int] = set()
        prev_t = events[0][0]
        for t, kind, i in events:
            if t > prev_t and active:
                dt = t - prev_t
                num_others = len(active) - 1
                for q_idx in active:
                    overlap_time[q_idx] += num_others * dt

            if kind == "end":
                active.discard(i)
            else:
                active.add(i)

            prev_t = t

        frames.append(
            pd.DataFrame(
                {
                    "query_id": cluster_queries["query_id"].to_numpy(),
                    "effective_concurrency": overlap_time / durations,
                }
            )
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=out_cols)


def plot_factor_error_vs_interarrival(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    arrivals_df: pd.DataFrame,
    *,
    title: str,
    color_col: Optional[str] = "rpu",
    alpha: float = 0.35,
    point_size: float = 10.0,
) -> pd.DataFrame:
    """Scatter plot of factor error against per-query interarrival time.

    Returns the plotting dataframe (query_id, interarrival_s, factor_error,
    optional color_col) after cleaning.
    """
    if "query_id" not in results_df.columns:
        raise ValueError("results_df must include 'query_id'.")
    if "factor_error" not in results_df.columns:
        raise ValueError("results_df must include 'factor_error'.")
    if not {"query_id", "arrival_s"}.issubset(arrivals_df.columns):
        raise ValueError("arrivals_df must include columns: 'query_id', 'arrival_s'.")

    inter_df = _interarrival_by_query(arrivals_df)

    plot_df = results_df.copy()
    plot_df["query_id"] = plot_df["query_id"].astype(str)
    plot_df = plot_df.merge(inter_df, on="query_id", how="left")

    keep_cols = ["query_id", "interarrival_s", "factor_error"]
    if color_col is not None and color_col in plot_df.columns:
        keep_cols.append(color_col)
    plot_df = plot_df[keep_cols].copy()

    plot_df["interarrival_s"] = pd.to_numeric(
        plot_df["interarrival_s"], errors="coerce"
    )
    plot_df["factor_error"] = pd.to_numeric(
        plot_df["factor_error"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["interarrival_s", "factor_error"])
    plot_df = plot_df[(plot_df["interarrival_s"] > 0) & (plot_df["factor_error"] > 0)]

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No valid points",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return plot_df

    if color_col is not None and color_col in plot_df.columns:
        is_categorical = isinstance(plot_df[color_col].dtype, pd.CategoricalDtype)
        groups = (
            plot_df.groupby(color_col, observed=True)
            if is_categorical
            else plot_df.groupby(color_col)
        )
        for key, sub in groups:
            color = None
            ax.scatter(
                sub["interarrival_s"],
                sub["factor_error"],
                s=point_size,
                alpha=alpha,
                label=str(key),
                color=color,
            )
            color = ax.collections[-1].get_facecolor()[0]
            _plot_trendline(
                ax,
                sub["interarrival_s"].to_numpy(dtype=float),
                sub["factor_error"].to_numpy(dtype=float),
                color=color,
                x_log=True,
                y_log=True,
            )
        ax.legend(title=color_col.upper(), fontsize=8, title_fontsize=8)
    else:
        ax.scatter(
            plot_df["interarrival_s"],
            plot_df["factor_error"],
            s=point_size,
            alpha=alpha,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Interarrival time (s)")
    ax.set_ylabel("Factor error (predicted / actual)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    return plot_df


def plot_factor_error_vs_effective_concurrency(
    ax: plt.Axes,
    results_df: pd.DataFrame,
    query_windows_df: pd.DataFrame,
    *,
    title: str,
    color_col: Optional[str] = "rpu",
    alpha: float = 0.35,
    point_size: float = 10.0,
) -> pd.DataFrame:
    """Scatter plot of factor error against per-query effective concurrency."""
    if "query_id" not in results_df.columns:
        raise ValueError("results_df must include 'query_id'.")
    if "factor_error" not in results_df.columns:
        raise ValueError("results_df must include 'factor_error'.")

    eff_df = _effective_concurrency_by_query(query_windows_df)
    plot_df = results_df.copy()
    plot_df["query_id"] = plot_df["query_id"].astype(str)
    plot_df = plot_df.merge(eff_df, on="query_id", how="left")

    keep_cols = ["query_id", "effective_concurrency", "factor_error"]
    if color_col is not None and color_col in plot_df.columns:
        keep_cols.append(color_col)
    plot_df = plot_df[keep_cols].copy()

    plot_df["effective_concurrency"] = pd.to_numeric(
        plot_df["effective_concurrency"], errors="coerce"
    )
    plot_df["factor_error"] = pd.to_numeric(
        plot_df["factor_error"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["effective_concurrency", "factor_error"])
    plot_df = plot_df[
        (plot_df["effective_concurrency"] >= 0) & (plot_df["factor_error"] > 0)
    ]

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No valid points",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return plot_df

    if color_col is not None and color_col in plot_df.columns:
        is_categorical = isinstance(plot_df[color_col].dtype, pd.CategoricalDtype)
        groups = (
            plot_df.groupby(color_col, observed=True)
            if is_categorical
            else plot_df.groupby(color_col)
        )
        for key, sub in groups:
            color = None
            ax.scatter(
                sub["effective_concurrency"],
                sub["factor_error"],
                s=point_size,
                alpha=alpha,
                label=str(key),
                color=color,
            )
            color = ax.collections[-1].get_facecolor()[0]
            _plot_trendline(
                ax,
                sub["effective_concurrency"].to_numpy(dtype=float),
                sub["factor_error"].to_numpy(dtype=float),
                color=color,
                x_log=False,
                y_log=True,
            )
        ax.legend(title=color_col.upper(), fontsize=8, title_fontsize=8)
    else:
        ax.scatter(
            plot_df["effective_concurrency"],
            plot_df["factor_error"],
            s=point_size,
            alpha=alpha,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Effective concurrency (avg other active queries)")
    ax.set_ylabel("Factor error (predicted / actual)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.35)
    return plot_df
