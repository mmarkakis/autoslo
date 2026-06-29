"""Comparison scatter plots for multiple IconqModels.

Provides two scatter plots that share the same visual language:

1. Q-error by split       — ``plot_qerror_by_split``
   X = model, columns within each group = train / val / test split.
   Y = Q-error (linear); markers at p50 / p90 / p95.

2. Factor error by RPU    — ``plot_factor_error_by_rpu``
   X = model, columns within each group = RPU value (test set only).
   Y = factor error (log scale); markers at p5 / p10 / p50 / p90 / p95.

Encoding shared by both plots
------------------------------
* **color** identifies the grouping dimension (split or RPU value).  A fixed
  palette is used so colors are stable across invocations.
* **Marker shape** identifies the percentile tier:
  - circle (o):        p50
  - triangle-up (^):   p90
  - square (s):        p95
  - diamond (D):       p10  (RPU plot only)
  - triangle-down (v): p5   (RPU plot only)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from autoslo.models.iconq_model import DataSplit, IconqModel
from autoslo.nn.lstm_state import AfterLSTMState
from autoslo.visualizations.colors import Palette
from autoslo.visualizations.iconq_model_performance import (
    _add_cluster_rpu,
    _add_observation_type,
)
from autoslo.workload_definition.query import ClusterAwareQueryId, Query
from autoslo.workload_definition.workload import Workload

# ── Model manifest entry ──────────────────────────────────────────────────────


@dataclass
class ModelEntry:
    """One row from the plotting manifest, describing a single model."""

    model_id: str
    """IconqModel ID (subdirectory under ``data/iconq_models/``)."""
    label: str
    """Tick label displayed on the X-axis."""
    annotate: bool = False
    """When True, every point for this model is annotated with its numeric value."""
    color: str | None = None
    """Optional Palette attribute name (e.g. ``'light_green'``).  When set,
    overrides the automatic color assigned from ``Palette.as_list()``."""

    def resolved_color(self, fallback_idx: int, fallback_list: list[str]) -> str:
        """Return the color for this model.

        If *color* is set, resolve it as ``getattr(Palette, color)``.
        Otherwise return ``fallback_list[fallback_idx % len(fallback_list)]``.
        """
        if self.color is not None:
            return getattr(Palette, self.color)
        return fallback_list[fallback_idx % len(fallback_list)]

# ── Constants ──────────────────────────────────────────────────────────────────

# Marker shapes per percentile tier (shared by both plots).
_MARKERS: dict[str, str] = {
    "p5": "v",  # triangle-down
    "p10": "D",  # diamond
    "p50": "o",  # circle
    "p90": "s",  # square
    "p95": "^",  # triangle-up
}
_MARKER_SIZE = 20  # markersize for ax.plot
_FONTSIZE = 20

# X-axis geometry.
_ITEM_SPACING = 0.35  # distance between items within a model group
_GROUP_GAP = 0.35  # extra gap between model groups

# RPU columns with fewer than this many test-set samples are excluded.
_MIN_RPU_SAMPLES_DEFAULT = 30

_DEFAULT_OUTPUT_DIR = "data/plots/iconq_comparison"


# ── Palette helpers ───────────────────────────────────────────────────────────


def _extend_rpu_palette(rpus: list[int]) -> dict[int, str]:
    """Return a color for every RPU value, extending beyond the built-in palette."""
    base = dict(Palette.rpu_to_color())
    # Extra colors for RPU values not in the built-in palette (e.g. 64, 128, 256).
    extras = [
        Palette.dark_yellow,
        Palette.dark_purple,
        Palette.gray,
        Palette.alt_light_green,
        Palette.alt_light_blue,
        Palette.alt_light_red,
    ]
    result: dict[int, str] = {}
    extra_idx = 0
    for rpu in sorted(rpus):
        if rpu in base:
            result[rpu] = base[rpu]
        else:
            result[rpu] = extras[extra_idx % len(extras)]
            extra_idx += 1
    return result


# ── Per-group computation helpers ─────────────────────────────────────────────


def _compute_percentiles(
    df: pd.DataFrame,
    col: str,
    quantiles: list[float],
) -> list[float] | None:
    """Return percentile values for *col* at *quantiles*, or None if no usable data."""
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return None
    return [float(vals.quantile(q)) for q in quantiles]


def _normal_df(split_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to non-aborted (normal) observations."""
    df = _add_observation_type(split_df)
    return df[df["observation_type"] == "normal"].copy()


# ── Figure sizing ─────────────────────────────────────────────────────────────


def _figsize(n_models: int, n_bars_per_group: int) -> tuple[float, float]:
    """Return a (width, height) that scales gracefully with bar count."""
    width = max(
        6.0, min(18.0, n_models * max(4, n_bars_per_group) * 0.35 + 2.0)
    )
    return width, 7


# ── Plot 1: Q-error by split ──────────────────────────────────────────────────


def plot_qerror_by_split(
    models: list[ModelEntry],
    all_split_dfs: dict[str, dict[DataSplit, pd.DataFrame]],
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    highlight_best: bool = True,
    annotate_best: bool = True,
    show_title: bool = True,
) -> tuple[Figure, str]:
    """Scatter plot comparing Q-error percentiles across splits.

    X-axis: splits (train / val / test); columns within each group = models.
    Y-axis: Q-error (linear).  Marker shape encodes percentile (p50/p90/p95).
    """
    model_ids = [m.model_id for m in models]
    n_models = len(model_ids)
    colors = Palette.as_list()

    splits = list(DataSplit)
    n_splits = len(splits)
    quantiles = [0.50, 0.90, 0.95]
    percentile_names = ["p50", "p90", "p95"]

    # ── Compute percentiles ──────────────────────────────────────────────────
    data: dict[tuple[str, str], list[float] | None] = {}
    max_p95 = 1.0
    for model_id in model_ids:
        for split in splits:
            df = _normal_df(all_split_dfs[model_id][split])
            result = _compute_percentiles(df, "q_error", quantiles)
            data[(model_id, split.value)] = result
            if result is not None:
                max_p95 = max(max_p95, result[-1])

    # ── Scatter positions ────────────────────────────────────────────────────
    group_width = n_models * _ITEM_SPACING + _GROUP_GAP
    group_centers = np.array([j * group_width for j in range(n_splits)])
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * _ITEM_SPACING

    # ── Best per (x-axis group, shape) = (split, percentile) ─────────────────
    best_model_for: dict[tuple[str, str], str] = {}
    for split in splits:
        for k, pct_name in enumerate(percentile_names):
            best_m: str | None = None
            best_dist = float("inf")
            for mid in model_ids:
                r = data[(mid, split.value)]
                if r is None:
                    continue
                d = abs(r[k] - 1.0)
                if d < best_dist:
                    best_dist = d
                    best_m = mid
            if best_m is not None:
                best_model_for[(split.value, pct_name)] = best_m

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=_figsize(n_splits, n_models))

    for i, (model_id, m) in enumerate(zip(model_ids, models)):
        color = m.resolved_color(i, colors)
        for j, split in enumerate(splits):
            result = data[(model_id, split.value)]
            if result is None:
                continue
            x = group_centers[j] + offsets[i]
            for pct_name, value in zip(percentile_names, result):
                is_best = best_model_for.get((split.value, pct_name)) == model_id
                ax.plot(
                    x,
                    value,
                    marker=_MARKERS[pct_name],
                    color=color,
                    linestyle="none",
                    markersize=_MARKER_SIZE,
                    markeredgecolor="black",
                    markeredgewidth=2.0 if (is_best and highlight_best) else 0.0,
                )
                if (is_best and annotate_best) or m.annotate:
                    ax.annotate(
                        f"{value:.2f}",
                        xy=(x, value),
                        xytext=(0, 6),
                        textcoords="offset points",
                        fontsize=_FONTSIZE,
                        color="black" if (is_best and annotate_best) else color,
                        rotation=90,
                        ha="center",
                        va="bottom",
                    )

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xticks(group_centers)
    ax.set_xticklabels([s.value.capitalize() for s in splits], fontsize=_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_FONTSIZE)
    ax.set_ylabel("Q-Error", fontsize=_FONTSIZE)
    ax.set_ylim(bottom=1, top=max_p95 * 1.15)
    ax.set_xlim(
        group_centers[0] - group_width * 0.5,
        group_centers[-1] + group_width * 0.5,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=1.0, zorder=0)
    if show_title:
        ax.set_title(
            "Q-Error Percentiles by Model and Split  (normal observations)",
            fontsize=_FONTSIZE,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    color_handles = [
        Line2D(
            [0], [0],
            marker=_MARKERS["p50"],
            color=m.resolved_color(i, colors),
            linestyle="none",
            markersize=_MARKER_SIZE,
            label=m.label,
        )
        for i, m in enumerate(models)
    ]
    marker_handles = [
        Line2D(
            [0], [0],
            marker=_MARKERS[p],
            color="gray",
            linestyle="none",
            markersize=_MARKER_SIZE,
            label=p,
        )
        for p in reversed(percentile_names)
    ]
    highlight_handles: list[Line2D] = []
    if highlight_best:
        highlight_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="white",
                linestyle="none",
                markersize=_MARKER_SIZE,
                markeredgecolor="black",
                markeredgewidth=2.0,
                label="Best",
            )
        )
    legend_rows = [color_handles, marker_handles] + (
        [highlight_handles] if highlight_handles else []
    )
    n_legend_rows = len(legend_rows)
    bottom_pad = 0.04 + n_legend_rows * 0.11
    fig.tight_layout(rect=[0, bottom_pad, 1, 1])
    _y_start = bottom_pad - 0.03
    _y_step = -(bottom_pad - 0.03) / n_legend_rows
    for i, row in enumerate(legend_rows):
        leg = ax.legend(
            handles=row,
            ncol=len(row),
            loc="upper center",
            bbox_to_anchor=(0.5, _y_start + i * _y_step),
            bbox_transform=fig.transFigure,
            borderaxespad=0,
            fontsize=_FONTSIZE,
            columnspacing=0.4,
            handletextpad=0.05,
        )
        if i < len(legend_rows) - 1:
            ax.add_artist(leg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(output_dir, "qerror_by_split.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 2: Factor error by RPU (test set only) ───────────────────────────────


def plot_factor_error_by_rpu(
    models: list[ModelEntry],
    all_split_dfs: dict[str, dict[DataSplit, pd.DataFrame]],
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    min_rpu_samples: int = _MIN_RPU_SAMPLES_DEFAULT,
    highlight_best: bool = True,
    annotate_best: bool = True,
    show_title: bool = True,
) -> tuple[Figure, str]:
    """Scatter plot of factor error percentiles by RPU (test set).

    X-axis: RPU values; columns within each group = models.
    Y-axis: factor error on a log scale.  Markers at p5, p10, p50, p90, p95;
    shape encodes percentile.  A dashed line marks 1.0 (perfect prediction).
    RPU values with fewer than *min_rpu_samples* normal test observations for
    every model are excluded; a note is added to the figure subtitle.
    """
    model_ids = [m.model_id for m in models]
    quantiles = [0.05, 0.10, 0.50, 0.90, 0.95]
    percentile_names = ["p5", "p10", "p50", "p90", "p95"]

    # ── Collect per-(model, RPU) sub-DataFrames ──────────────────────────────
    rpu_dfs: dict[tuple[str, int], pd.DataFrame] = {}
    all_rpus_seen: set[int] = set()
    qualified_rpus: set[int] = set()

    for model_id in model_ids:
        test_df = _add_cluster_rpu(all_split_dfs[model_id][DataSplit.TEST])
        test_df = _normal_df(test_df)
        test_df["rpu"] = pd.to_numeric(test_df["rpu"], errors="coerce").astype(
            "Int64"
        )
        test_df = test_df.dropna(subset=["rpu"])
        for rpu_val, sub in test_df.groupby("rpu", observed=True):
            rpu_int = int(str(rpu_val))
            all_rpus_seen.add(rpu_int)
            if len(sub) >= min_rpu_samples:
                qualified_rpus.add(rpu_int)
                rpu_dfs[(model_id, rpu_int)] = sub.copy()

    excluded_rpus = sorted(all_rpus_seen - qualified_rpus)
    if not qualified_rpus:
        raise ValueError(
            f"No RPU values have >= {min_rpu_samples} test-set samples in any model. "
            "Lower --min-rpu-samples or check the test set size."
        )

    sorted_rpus = sorted(qualified_rpus)
    n_models = len(model_ids)
    n_rpus = len(sorted_rpus)
    colors = Palette.as_list()

    # ── Compute percentiles ──────────────────────────────────────────────────
    data: dict[tuple[str, int], list[float] | None] = {}
    for model_id in model_ids:
        for rpu_val in sorted_rpus:
            rpu_sub = rpu_dfs.get((model_id, rpu_val))
            if rpu_sub is None or rpu_sub.empty:
                data[(model_id, rpu_val)] = None
                continue
            data[(model_id, rpu_val)] = _compute_percentiles(
                rpu_sub, "factor_error", quantiles
            )

    # ── Scatter positions ────────────────────────────────────────────────────
    group_width = n_models * _ITEM_SPACING + _GROUP_GAP
    group_centers = np.array([j * group_width for j in range(n_rpus)])
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * _ITEM_SPACING

    # ── Best per (x-axis group, shape) = (rpu, percentile) ───────────────────
    best_model_for: dict[tuple[int, str], str] = {}
    for rpu_val in sorted_rpus:
        for k, pct_name in enumerate(percentile_names):
            best_m: str | None = None
            best_dist = float("inf")
            for mid in model_ids:
                r = data[(mid, rpu_val)]
                if r is None:
                    continue
                d = abs(float(np.log(r[k])))
                if d < best_dist:
                    best_dist = d
                    best_m = mid
            if best_m is not None:
                best_model_for[(rpu_val, pct_name)] = best_m

    # ── Draw ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=_figsize(n_rpus, n_models))

    for i, (model_id, m) in enumerate(zip(model_ids, models)):
        color = m.resolved_color(i, colors)
        for j, rpu_val in enumerate(sorted_rpus):
            result = data[(model_id, rpu_val)]
            if result is None:
                continue
            x = group_centers[j] + offsets[i]
            for pct_name, value in zip(percentile_names, result):
                is_best = best_model_for.get((rpu_val, pct_name)) == model_id
                ax.plot(
                    x,
                    value,
                    marker=_MARKERS[pct_name],
                    color=color,
                    linestyle="none",
                    markersize=_MARKER_SIZE,
                    markeredgecolor="black",
                    markeredgewidth=2.0 if (is_best and highlight_best) else 0.0,
                )
                if (is_best and annotate_best) or m.annotate:
                    ax.annotate(
                        f"{value:.2f}",
                        xy=(x, value),
                        xytext=(0, 6),
                        textcoords="offset points",
                        fontsize=_FONTSIZE,
                        color="black" if (is_best and annotate_best) else color,
                        rotation=90,
                        ha="center",
                        va="bottom",
                    )

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{r} RPU" for r in sorted_rpus], fontsize=_FONTSIZE)
    ax.tick_params(axis="y", labelsize=_FONTSIZE)
    ax.set_ylabel("Predicted / Actual", fontsize=_FONTSIZE)
    ax.set_yscale("log")
    ax.set_xlim(
        group_centers[0] - group_width * 0.5,
        group_centers[-1] + group_width * 0.5,
    )
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=1.0, zorder=0)
    subtitle_parts = ["test set · normal observations only"]
    if excluded_rpus:
        subtitle_parts.append(
            f"RPU {excluded_rpus} excluded (< {min_rpu_samples} samples in all models)"
        )
    if show_title:
        ax.set_title(
            "Factor Error (predicted / actual) by Model and RPU\n"
            + " · ".join(subtitle_parts),
            fontsize=_FONTSIZE,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    model_handles = [
        Line2D(
            [0], [0],
            marker=_MARKERS["p50"],
            color=m.resolved_color(i, colors),
            linestyle="none",
            markersize=_MARKER_SIZE,
            label=m.label,
        )
        for i, m in enumerate(models)
    ]
    marker_handles = [
        Line2D(
            [0], [0],
            marker=_MARKERS[p],
            color="gray",
            linestyle="none",
            markersize=_MARKER_SIZE,
            label=p,
        )
        for p in reversed(percentile_names)
    ]
    highlight_handles: list[Line2D] = []
    if highlight_best:
        highlight_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="white",
                linestyle="none",
                markersize=_MARKER_SIZE,
                markeredgecolor="black",
                markeredgewidth=2.0,
                label="Best",
            )
        )
    legend_rows = [model_handles, marker_handles] + (
        [highlight_handles] if highlight_handles else []
    )
    n_legend_rows = len(legend_rows)
    bottom_pad = 0.04 + n_legend_rows * 0.11
    fig.tight_layout(rect=[0, bottom_pad, 1, 1])
    _y_start = bottom_pad - 0.03
    _y_step = -(bottom_pad - 0.03) / n_legend_rows
    for i, row in enumerate(legend_rows):
        leg = ax.legend(
            handles=row,
            ncol=len(row),
            loc="upper center",
            bbox_to_anchor=(0.5, _y_start + i * _y_step),
            bbox_transform=fig.transFigure,
            borderaxespad=0,
            fontsize=_FONTSIZE,
            columnspacing=0.4,
            handletextpad=0.05,
        )
        if i < len(legend_rows) - 1:
            ax.add_artist(leg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(output_dir, "factor_error_by_rpu.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 3: Inference time by arrival ─────────────────────────────────────────


def plot_inference_time_by_arrival(
    models: list[ModelEntry],
    workload: Workload,
    rpu: int,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    max_arrivals: int | None = None,
    show_title: bool = True,
) -> tuple[Figure, str]:
    """Line plot of per-arrival inference time as the active query set grows.

    For each model and each query arrival *i* (in ``rel_start_time_s`` order),
    measures the wall-clock time of all inference calls triggered by that
    arrival:

    * **Stateful models** (``supports_stateful_inference = True``):
      one batched :meth:`predict_incremental_batch` over the *i-1* already-active
      queries followed by one :meth:`predict_initial_batch` for the new query.
    * **Non-stateful models**: one :meth:`predict_from_query_groups` call that
      re-infers all *i* queries simultaneously.

    Queries never complete — the active set grows monotonically — so the cost
    plotted reflects the marginal burden of each new arrival.

    Before the timing loop each model's workload stage predictions are
    populated via
    :meth:`~autoslo.workload_definition.workload.Workload.populate_featurizations_and_isolated_predictions`
    so that featurization is correct.

    Parameters
    ----------
    models:
        Ordered list of model entries (loaded by the caller from the manifest).
    workload:
        The workload whose queries drive the benchmark.  Sorted by
        ``rel_start_time_s`` internally.
    rpu:
        RPU value for the virtual cluster used during inference.
    output_dir:
        Directory where the PNG is written.
    max_arrivals:
        If set, only the first *max_arrivals* queries (by arrival order) are
        processed.  Useful to keep run time bounded.

    Returns
    -------
    (fig, save_path)
    """
    cluster_name = f"autoslo-{rpu}-0-0"

    # Sort queries by arrival time and optionally cap.
    all_queries: list[Query] = sorted(
        workload.queries(), key=lambda q: q.rel_start_time_s
    )
    if max_arrivals is not None:
        all_queries = all_queries[:max_arrivals]

    n_arrivals = len(all_queries)
    arrival_indices = list(range(1, n_arrivals + 1))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"inference_timing_{workload.workload_config.id()}.csv",
    )

    # ── Load cached results (if any) ─────────────────────────────────────────
    model_times: dict[str, list[float]] = {}
    if os.path.exists(csv_path):
        cached = pd.read_csv(csv_path, index_col="arrival_index")
        if len(cached) == n_arrivals:
            for col in cached.columns:
                model_times[col] = cached[col].tolist()

    # ── Run timing for models not yet cached ─────────────────────────────────
    colors = [
        Palette.light_blue,
        Palette.light_green,
        Palette.light_red,
        Palette.light_purple,
    ]

    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        for m in models:
            if m.model_id in model_times:
                progress.add_task(
                    f"{m.label}  (RPU={rpu}, cached)", total=1, completed=1
                )
                continue

            model = IconqModel.load(m.model_id, inference_mode=True)

            # Populate stage predictions so featurization is accurate.
            workload.populate_featurizations_and_isolated_predictions(
                model, [rpu]
            )
            # Re-fetch queries with populated stage predictions.
            queries: list[Query] = sorted(
                workload.queries(), key=lambda q: q.rel_start_time_s
            )
            if max_arrivals is not None:
                queries = queries[:max_arrivals]

            times_ms: list[float] = []
            active_queries: list[Query] = []
            lstm_states: dict[str, AfterLSTMState] = {}

            task = progress.add_task(
                f"{m.label}  (RPU={rpu})", total=len(queries)
            )

            if model.supports_stateful_inference:
                for q in queries:
                    caqi_new = ClusterAwareQueryId.make(
                        cluster_name, q.query_id
                    )
                    t0 = time.perf_counter()

                    # Incremental update for all currently active queries.
                    if lstm_states:
                        inc_items: list[
                            tuple[AfterLSTMState, ClusterAwareQueryId]
                        ] = [
                            (
                                lstm_states[aq.query_id],
                                ClusterAwareQueryId.make(
                                    cluster_name, aq.query_id
                                ),
                            )
                            for aq in active_queries
                        ]
                        for caqi, (
                            _pred,
                            new_state,
                        ) in model.predict_incremental_batch(
                            inc_items, q
                        ).items():
                            lstm_states[caqi.query_id] = new_state

                    # Initial prediction for the new query.
                    for caqi, (_pred, state) in model.predict_initial_batch(
                        [(caqi_new, q, list(active_queries))]
                    ).items():
                        lstm_states[q.query_id] = state

                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    times_ms.append(elapsed_ms)
                    active_queries.append(q)
                    progress.advance(task)

            else:
                for q in queries:
                    all_now: list[Query] = active_queries + [q]
                    base_to_neighbors: dict[Query, list[Query]] = {
                        qj: [qk for qk in all_now if qk.query_id != qj.query_id]
                        for qj in all_now
                    }
                    t0 = time.perf_counter()
                    model.predict_from_query_groups(
                        {cluster_name: base_to_neighbors}
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    times_ms.append(elapsed_ms)
                    active_queries.append(q)
                    progress.advance(task)

            model_times[m.model_id] = times_ms

            # Persist after every newly completed model so partial runs can resume.
            csv_df = pd.DataFrame(
                model_times,
                index=pd.RangeIndex(1, n_arrivals + 1, name="arrival_index"),
            )
            csv_df.to_csv(csv_path)

    # ── Draw ──────────────────────────────────────────────────────────────────
    _ROLLING = 10

    fig, ax = plt.subplots(figsize=(9, 7))

    for m_idx, m in enumerate(models):
        color = m.resolved_color(m_idx, colors)
        raw = model_times[m.model_id]
        rolling = (
            pd.Series(raw).rolling(_ROLLING, min_periods=1).mean().tolist()
        )

        # Raw: thin and dim.
        ax.plot(
            arrival_indices,
            raw,
            color=color,
            linewidth=3.0,
            alpha=0.25,
            linestyle=":",
        )
        # Rolling average: full opacity.
        ax.plot(
            arrival_indices,
            rolling,
            color=color,
            linewidth=3.0,
            alpha=1.0,
        )

    ax.set_xlabel("Query arrival index", fontsize=_FONTSIZE)
    ax.set_ylabel("Inference time (ms)", fontsize=_FONTSIZE)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=_FONTSIZE)
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=1.0, zorder=0)
    if show_title:
        ax.set_title(
            f"Inference time per arrival (RPU={rpu})", fontsize=_FONTSIZE
        )

    # Legend row 1: model colors.
    color_handles = [
        Line2D(
            [0],
            [0],
            color=m.resolved_color(i, colors),
            linewidth=3.0,
            label=m.label,
        )
        for i, m in enumerate(models)
    ]
    # Legend row 2: line style explanation.
    style_handles = [
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=3.0,
            alpha=0.25,
            label="Raw",
            linestyle=":",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=3.0,
            alpha=1.0,
            label=f"{_ROLLING}-pt rolling avg",
        ),
    ]

    _legend_rows = [color_handles, style_handles]
    _y_start = -0.17
    _y_step = -0.12
    for i, _row in enumerate(_legend_rows):
        _leg = ax.legend(
            handles=_row,
            ncol=len(_row),
            loc="upper center",
            bbox_to_anchor=(0.5, _y_start + i * _y_step),
            borderaxespad=0,
            fontsize=_FONTSIZE,
            columnspacing=0.8,
            handletextpad=0.1,
        )
        if i < len(_legend_rows) - 1:
            ax.add_artist(_leg)

    fig.tight_layout()

    save_path = os.path.join(output_dir, "inference_time_by_arrival.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path
