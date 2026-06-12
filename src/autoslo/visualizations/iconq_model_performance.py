"""Diagnostic plots for IconqModel prediction performance.

Provides eight diagnostic plots (each with one panel per data split):

1. Q-Error CDF               — ``plot_qerror_cdf``
2. Predicted vs. Actual      — ``plot_predicted_vs_actual``
3. Q-Error percentiles over  — ``plot_qerror_over_epochs``
   training epochs
4. Q-Error vs. Concurrency   — ``plot_qerror_vs_concurrency``
5. Q-Error heatmap           — ``plot_qerror_heatmap``
   (template × concurrency)
6. Error by cluster RPU      — ``plot_error_by_cluster_rpu``
    (magnitude + direction)
7. Error CDF by cluster RPU  — ``plot_error_cdf_by_cluster_rpu``
    (magnitude + direction)
8. Signed error heatmap      — ``plot_signed_error_heatmap_rpu_x_concurrency``
    (RPU × concurrency)
9. Template contribution     — ``plot_template_contribution_breakdown``
    (error × frequency weighting)

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from autoslo.clusters.cluster import Cluster
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.models.iconq_model import DataSplit, IconqModel
from autoslo.visualizations.colors import Palette
from autoslo.visualizations.prediction_error_cdf import (
    add_monospace_summary_box,
    build_direction_summary_lines,
    build_percentile_summary_lines,
    plot_grouped_cdf,
)
from autoslo.workload_definition.query import QueryTextId

# ── Constants ──────────────────────────────────────────────────────────────────

_SPLIT_COLORS: dict[str, str] = {
    "train": Palette.dark_blue,
    "val": Palette.dark_orange,
    "test": Palette.dark_green,
}

_PERCENTILE_COLORS: dict[int, str] = {
    50: Palette.dark_blue,
    90: Palette.dark_orange,
    95: Palette.dark_red,
    99: Palette.dark_purple,
}
_PERCENTILE_LINE_STYLES: dict[int, str] = {
    50: "-",
    90: "--",
    95: "-.",
    99: ":",
}

# Concurrency bins used by plots 4 and 5.
_CONC_BINS = [-0.5, 0.5, 25.5, 75.5, 150.5, 250.5, float("inf")]
_CONC_LABELS = ["0", "1-25", "26-75", "76-150", "151-250", "251+"]

# Cells in the heatmap with fewer than this many samples are masked.
_MIN_HEATMAP_SAMPLES = 5


# ── Helpers ───────────────────────────────────────────────────────────────────


def _add_concurrency_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a ``conc_bin`` categorical column."""
    out = df.copy()
    out["conc_bin"] = pd.cut(
        out["num_other_concurrent_queries"],
        bins=_CONC_BINS,
        labels=_CONC_LABELS,
    )
    return out


def _add_cluster_rpu(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a numeric ``cluster_rpu`` column."""
    out = df.copy()

    if "cluster_rpu" in out.columns:
        out["cluster_rpu"] = pd.to_numeric(
            out["cluster_rpu"], errors="coerce"
        ).astype("Int64")
        return out

    if "cluster_name" in out.columns:
        out["cluster_rpu"] = (
            out["cluster_name"]
            .astype(str)
            .apply(lambda name: Cluster.rpu_for_cluster_name(name))
            .astype("Int64")
        )
        return out

    if "query_id" in out.columns:
        # Fallback for cluster-aware ids serialized as "<cluster>#<query_id>".
        parsed = out["query_id"].astype(str).str.split("#", n=1).str[0]
        if parsed.str.startswith("autoslo-").all():
            out["cluster_rpu"] = parsed.apply(
                lambda name: Cluster.rpu_for_cluster_name(name)
            ).astype("Int64")
            return out

    raise ValueError(
        "cluster_rpu metadata is missing from final split DataFrames. "
        "Re-run model evaluation so final_{train,val,test}.csv include "
        "cluster_name/cluster_rpu columns."
    )


def plot_all(iconq_model_id: str) -> None:
    """
    Run all five plots for *model*.

    Calls :func:`build_split_predictions` once and reuses the result for all
    plots that accept a ``split_dfs`` argument.
    """

    split_dfs = IconqModel.optimized_load_final_dfs_per_split(iconq_model_id)

    fns = [
        plot_qerror_cdf,
        plot_predicted_vs_actual,
        plot_qerror_over_epochs,
        plot_qerror_vs_concurrency,
        plot_qerror_heatmap,
    ]

    for fn in fns:
        fig, save_path = fn(split_dfs, iconq_model_id)
        plt.close(fig)
        print(f"  saved: {save_path}")


# ── Plot 1: Q-Error CDF ───────────────────────────────────────────────────────


def plot_qerror_cdf(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """Q-Error cumulative distribution function, one panel per split.

    X-axis: Q-Error (log scale).
    Y-axis: Fraction of queries with Q-Error ≤ x.
    Vertical lines mark the P50/P90/P95 percentiles.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    percentiles = [50, 90, 95]

    for ax, split in zip(axes, DataSplit):
        df = split_dfs[split]
        q_errs = np.sort(df["q_error"].values)
        cdf = np.arange(1, len(q_errs) + 1) / len(q_errs)
        ax.plot(q_errs, cdf, color=_SPLIT_COLORS[split.value], lw=2)

        for p in percentiles:
            x_at_p = np.interp(p, cdf * 100, q_errs)
            ax.plot(
                [x_at_p, x_at_p],
                [0, 1],
                linestyle=_PERCENTILE_LINE_STYLES[p],
                color=Palette.gray,
                lw=0.9,
                label=f"p{p}: {x_at_p:.2f}",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Q-Error")
        ax.set_ylabel("Fraction ≤ x" if split == DataSplit.TRAIN else "")
        ax.set_ylim(0, 1)
        ax.set_title(f"{split.value.capitalize()}  (N={len(df):,})")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"{iconq_model_id} - Q-Error CDF", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id), "qerror_cdf.png"
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 2: Predicted vs. Actual scatter ──────────────────────────────────────


def plot_predicted_vs_actual(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """
    Log-log scatter of predicted vs. actual latency, one panel per split.

    Points are coloured by ``num_other_concurrent_queries``.  Reference lines
    mark perfect prediction and the ±2× band.
    """
    # Compute limits.
    nocq = "num_other_concurrent_queries"
    vmax = max(1, max(split_dfs[s][nocq].max() for s in DataSplit))
    norm = SymLogNorm(linthresh=1, vmin=0, vmax=vmax)
    all_latencies = pd.concat(
        [split_dfs[s][["y", "y_pred_mean"]] for s in DataSplit]
    )
    lim_min = max(float(all_latencies.min().min()) * 0.8, 1e-6)
    lim_max = float(all_latencies.max().max()) * 1.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    scatter_handles = []

    for ax, split in zip(axes, DataSplit):
        df = split_dfs[split]
        sc = ax.scatter(
            df["y"],
            df["y_pred_mean"],
            c=df[nocq],
            cmap="viridis",
            norm=norm,
            s=8,
            alpha=0.5,
        )
        scatter_handles.append(sc)

        # Reference lines.
        xs = np.array([lim_min, lim_max])
        ax.plot(xs, xs, "k-", lw=1.2, label="Perfect (y = x)")
        _do = Palette.dark_orange
        ax.plot(xs, xs * 2, "--", color=_do, lw=0.8, label="2× band")
        ax.plot(xs, xs / 2, "--", color=_do, lw=0.8)

        # Setup.
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Actual latency (s)")
        if split == DataSplit.TRAIN:
            ax.set_ylabel("Predicted latency (s)")
        ax.set_title(f"{split.value.capitalize()}  (N={len(df):,})")
        ax.grid(True, which="both", linestyle=":", alpha=0.3)
        ax.legend(fontsize=7)

    cb = fig.colorbar(
        scatter_handles[-1],
        ax=list(axes),
        location="right",
        fraction=0.015,
        pad=0.04,
    )
    cb.set_label("# other concurrent queries")

    fig.suptitle(f"{iconq_model_id} - Predicted vs. Actual", fontsize=12)
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id), "predicted_vs_actual.png"
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 3: Q-Error percentiles over training epochs ──────────────────────────


def plot_qerror_over_epochs(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """
    Two-panel epoch-trajectory plot.

    Left panel  — Val Q-Error percentile trajectories (P50/P75/P90/P99) over
                  all training epochs, read from ``val_predictions_<epoch>.csv``
                  files.  A vertical dashed line marks the best-val-loss epoch
                  (the saved checkpoint).  The P25–P75 band is shaded.

    Right panel — Final-epoch Q-Error percentile comparison across all three
                  splits (train / val / test) shown as a grouped bar chart.
    """
    model_dir = Path(IconqModel.default_save_dir(iconq_model_id))

    # Discover epoch CSV files.
    epoch_csv_files: dict[int, Path] = {}
    for f in model_dir.glob("val_predictions_*.csv"):
        m = re.search(r"val_predictions_(\d+)\.csv$", f.name)
        if m:
            epoch_csv_files[int(m.group(1))] = f

    fig, (ax_val, ax_final) = plt.subplots(1, 2, figsize=(14, 5))

    # Read all epoch CSVs in one pass.
    # NOTE: Some CSVs contain corrupted rows where isolated-query y values
    # were serialised as raw PyTorch tensors (e.g. "tensor(20.9451)").
    # pd.to_numeric(..., errors="coerce") silently drops those rows.
    epoch_stats: dict[int, dict] = {}
    for epoch, csv_path in sorted(epoch_csv_files.items()):
        df_ep = pd.read_csv(
            csv_path,
            usecols=lambda c: c.strip() in {"q_error", "individual_loss"},
        )
        q_errs = (
            pd.to_numeric(df_ep["q_error"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
            if "q_error" in df_ep.columns
            else np.empty(0, dtype=float)
        )
        if "individual_loss" in df_ep.columns:
            loss_numeric = pd.to_numeric(
                df_ep["individual_loss"], errors="coerce"
            )
            mean_loss = (
                float(loss_numeric.mean())
                if not loss_numeric.isna().all()
                else float("nan")
            )
        else:
            mean_loss = float("nan")

        pc_or_nan = lambda p: (
            float(np.percentile(q_errs, p)) if len(q_errs) else float("nan")
        )
        epoch_stats[epoch] = {
            "mean_loss": mean_loss,
            "p25": pc_or_nan(25),
            "p50": pc_or_nan(50),
            "p90": pc_or_nan(90),
            "p95": pc_or_nan(95),
            "p99": pc_or_nan(99),
        }

    epochs = sorted(epoch_stats.keys())

    # Best epoch: lowest mean validation loss.
    finite_epochs = [
        e for e in epochs if not np.isnan(epoch_stats[e]["mean_loss"])
    ]
    best_epoch = (
        min(finite_epochs, key=lambda e: epoch_stats[e]["mean_loss"])
        if finite_epochs
        else epochs[-1]
    )

    # Draw val trajectory.
    percentile_specs = [
        (25, None, None, None),  # band only
        (50, _PERCENTILE_COLORS[50], "-", "P50"),
        (90, _PERCENTILE_COLORS[90], "-", "P90"),
        (95, _PERCENTILE_COLORS[95], "-", "P95"),
        (99, _PERCENTILE_COLORS[99], "-", "P99"),
    ]
    for p, color, ls, label in percentile_specs:
        key = f"p{p}"
        ys = [epoch_stats[e][key] for e in epochs]
        if color is not None:
            ax_val.plot(
                epochs, ys, color=color, linestyle=ls, lw=1.8, label=label
            )

    ax_val.axvline(
        best_epoch,
        color=Palette.dark_red,
        linestyle="--",
        lw=1.2,
        label=f"Best epoch ({best_epoch})",
    )
    ax_val.set_yscale("log")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Q-Error")
    ax_val.set_title("Val Q-Error percentiles over epochs")
    ax_val.legend(fontsize=8)
    ax_val.grid(True, which="both", linestyle=":", alpha=0.4)

    # Right panel: grouped bar chart for final-epoch split comparison.
    percentile_names = ["P50", "P90", "P95", "P99"]
    percentile_keys = [50, 90, 95, 99]
    x_pos = np.arange(len(percentile_names))
    bar_width = 0.25

    for i, split in enumerate(DataSplit):
        df = split_dfs[split]
        if df.empty:
            continue
        q_errs = df["q_error"].dropna().values
        vals = [
            float(np.percentile(q_errs, p)) if len(q_errs) else float("nan")
            for p in percentile_keys
        ]
        offsets = x_pos + (i - 1) * bar_width
        ax_final.bar(
            offsets,
            vals,
            bar_width,
            label=f"{split.value.capitalize()} (N={len(df):,})",
            color=_SPLIT_COLORS[split.value],
            alpha=0.85,
        )

        # Annotate bars with percentile values.
        for x, v in zip(offsets, vals):
            if not np.isnan(v):
                ax_final.text(
                    x,
                    v,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                    alpha=0.7,
                )

    ax_final.set_xticks(x_pos)
    ax_final.set_xticklabels(percentile_names)
    ax_final.set_ylabel("Q-Error")
    ax_final.set_yscale("log")
    ax_final.set_title("Final-epoch Q-Error by split")
    ax_final.legend(fontsize=8)
    ax_final.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(f"{iconq_model_id} - Q-Error over Epochs", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id), "qerror_over_epochs.png"
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 4: Q-Error vs. Concurrency ───────────────────────────────────────────


def plot_qerror_vs_concurrency(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """Box plots of Q-Error grouped by concurrency bin, one panel per split.

    The 0-concurrent bin (isolated queries predicted by the stage model) is
    hatched to distinguish it from bins predicted by the LSTM.  A connected
    line shows the median trend across bins.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, split in zip(axes, DataSplit):
        df = _add_concurrency_bins(split_dfs[split])
        groups = [
            df.loc[df["conc_bin"] == lbl, "q_error"].dropna().values
            for lbl in _CONC_LABELS
        ]

        # Build non-empty groups with their original positions.
        non_empty_data: list[np.ndarray] = []
        non_empty_positions: list[int] = []
        for j, g in enumerate(groups):
            if len(g) > 0:
                non_empty_data.append(g)
                non_empty_positions.append(
                    j + 1
                )  # boxplot uses 1-based positions

        if non_empty_data:
            bp = ax.boxplot(
                non_empty_data,
                positions=non_empty_positions,
                widths=0.6,
                patch_artist=True,
                medianprops=dict(color="black", lw=1.5),
                whiskerprops=dict(lw=0.8),
                capprops=dict(lw=0.8),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                manage_ticks=False,
            )
            for patch, pos in zip(bp["boxes"], non_empty_positions):
                patch.set_facecolor(_SPLIT_COLORS[split.value])
                patch.set_alpha(0.7)
                if _CONC_LABELS[pos - 1] == "0":
                    patch.set_hatch("//")

            medians = [float(np.median(g)) for g in non_empty_data]
            ax.plot(
                non_empty_positions,
                medians,
                "o-",
                color=Palette.dark_red,
                lw=1.0,
                ms=4,
                label="Median trend",
            )
            ax.legend(fontsize=8)

        ax.set_yscale("log")
        ax.set_xticks(range(1, len(_CONC_LABELS) + 1))
        ax.set_xticklabels(_CONC_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("# concurrent queries")
        ax.set_ylabel("Q-Error" if split == DataSplit.TRAIN else "")
        ax.set_title(
            f"{split.value.capitalize()}  (N={len(split_dfs[split]):,})"
        )
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(f"{iconq_model_id} - Q-Error vs. Concurrency", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id), "qerror_vs_concurrency.png"
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 5: Q-Error heatmap (template × concurrency bin) ──────────────────────


def plot_qerror_heatmap(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """Heatmap of median Q-Error per (query template × concurrency bin).

    Rows are sorted by the val-split median Q-Error (worst templates at top)
    and the same row order is used for all three panels so comparisons are
    easy.  Cells with fewer than *min_samples* observations are greyed out.
    """
    # Enrich all splits with template and concurrency-bin columns.
    enriched: dict[DataSplit, pd.DataFrame] = {}
    for split in DataSplit:
        df = _add_concurrency_bins(split_dfs[split]).copy()
        df["template"] = df["query_text_id"].apply(
            lambda x: QueryTextId(x).template_id
        )
        enriched[split] = df

    # Collect all templates, sorted by val median Q-Error (descending).
    val_template_medians = (
        enriched[DataSplit.VAL]
        .groupby("template")["q_error"]
        .median()
        .sort_values(ascending=False)
    )
    val_sorted_templates = val_template_medians.index.tolist()

    all_templates = sorted(
        set().union(*(set(df["template"]) for df in enriched.values())),
        key=lambda t: (
            val_sorted_templates.index(t)
            if t in val_sorted_templates
            else len(val_sorted_templates)
        ),
    )

    n_templates = len(all_templates)
    n_bins = len(_CONC_LABELS)
    fig_height = min(max(6, n_templates * 0.28), 22)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad("lightgray")

    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height))

    template_index = {t: i for i, t in enumerate(all_templates)}

    for ax, split in zip(axes, DataSplit):
        df = enriched[split]
        grid = np.full((n_templates, n_bins), np.nan)

        for j, bin_label in enumerate(_CONC_LABELS):
            bin_df = df[df["conc_bin"] == bin_label]
            for template, grp in bin_df.groupby("template", observed=True)[
                "q_error"
            ]:
                i = template_index.get(template)
                if i is not None and len(grp) >= _MIN_HEATMAP_SAMPLES:
                    grid[i, j] = float(grp.median())

        vmax = max(
            3.0,
            (
                float(np.nanpercentile(grid, 95))
                if not np.all(np.isnan(grid))
                else 3.0
            ),
        )

        im = ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            cmap=cmap,
            vmin=1.0,
            vmax=vmax,
            origin="upper",
        )

        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(_CONC_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_templates))
        ax.set_yticklabels(all_templates, fontsize=6)
        ax.set_xlabel("# concurrent queries")
        ax.set_ylabel("Template" if split == DataSplit.TRAIN else "")
        ax.set_title(f"{split.value.capitalize()}  (N={len(df):,})")

        # Annotate cell sample counts when there are ≤30 templates.
        if n_templates <= 30:
            for i, template in enumerate(all_templates):
                for j, bin_label in enumerate(_CONC_LABELS):
                    n = int(
                        (
                            (df["template"] == template)
                            & (df["conc_bin"] == bin_label)
                        ).sum()
                    )
                    if n > 0 and not np.isnan(grid[i, j]):
                        ax.text(
                            j,
                            i,
                            str(n),
                            ha="center",
                            va="center",
                            fontsize=5,
                            color="black",
                            alpha=0.5,
                        )

        fig.colorbar(
            im, ax=ax, label="Median Q-Error", fraction=0.046, pad=0.04
        )

    fig.suptitle(
        f"{iconq_model_id} - Q-Error Heatmap (Template x Concurrency)",
        fontsize=12,
    )
    fig.tight_layout()
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id), "qerror_heatmap.png"
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 6: Template contribution breakdown ──────────────────────────────────


def plot_template_contribution_breakdown(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
    run_id: Optional[str] = None,
    top_k: int = 12,
) -> tuple[Figure, str]:
    """Template-level error/frequency/contribution breakdown per split.

    For each split and template:
    - error: median Q-Error
    - frequency: fraction of samples in the split
    - contribution: frequency * error

    If *run_id* is provided, overlays run template frequency and
    run-mix contribution (run frequency * split template error) to help
    diagnose whether degraded performance is due to submitting more
    difficult templates.
    """

    # Prepare split-level template statistics.
    split_stats: dict[DataSplit, pd.DataFrame] = {}
    for split in DataSplit:
        df = split_dfs[split].copy()
        if "query_text_id" not in df.columns or "q_error" not in df.columns:
            split_stats[split] = pd.DataFrame(
                columns=["template", "error", "count", "freq", "contribution"]
            )
            continue

        df["template"] = df["query_text_id"].astype(str).apply(
            lambda x: QueryTextId(x).template_id
        )
        grouped = (
            df.groupby("template", observed=True)
            .agg(
                error=("q_error", "median"),
                count=("q_error", "size"),
            )
            .reset_index()
        )
        total = int(grouped["count"].sum())
        grouped["freq"] = grouped["count"] / max(total, 1)
        grouped["contribution"] = grouped["freq"] * grouped["error"]
        split_stats[split] = grouped

    # Optional run-level template frequencies from submitted queries.
    run_freq_df: pd.DataFrame | None = None
    if run_id:
        try:
            run_log = StructuredLog.load(run_id)
            arrivals = run_log.df
            if {"event_type", "query_text_id"}.issubset(arrivals.columns):
                arrivals = arrivals[
                    (arrivals["event_type"] == "arrival")
                    & arrivals["query_text_id"].notna()
                ].copy()
                if not arrivals.empty:
                    arrivals["template"] = arrivals["query_text_id"].astype(
                        str
                    ).apply(lambda x: QueryTextId(x).template_id)
                    run_freq_df = (
                        arrivals.groupby("template", observed=True)
                        .size()
                        .rename("count")
                        .reset_index()
                    )
                    run_total = int(run_freq_df["count"].sum())
                    run_freq_df["run_freq"] = run_freq_df["count"] / max(
                        run_total, 1
                    )
        except Exception:
            run_freq_df = None

    # Choose a global template order by average contribution across splits.
    all_stats = pd.concat(
        [df.assign(split=split.value) for split, df in split_stats.items()],
        ignore_index=True,
    )
    if all_stats.empty:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No template-level data", ha="center", va="center")
        ax.set_axis_off()
        save_path = os.path.join(
            IconqModel.default_save_dir(iconq_model_id),
            "template_contribution_breakdown.png",
        )
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig, save_path

    template_rank = (
        all_stats.groupby("template", observed=True)["contribution"]
        .mean()
        .sort_values(ascending=False)
    )
    templates = template_rank.head(top_k).index.tolist()

    fig, axes = plt.subplots(
        3, 3, figsize=(20, 12), constrained_layout=True, sharey="row"
    )
    col_titles = [
        "Template Error (median Q-Error)",
        "Template Frequency",
        "Template Contribution (freq × error)",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title)

    for row_idx, split in enumerate(DataSplit):
        s_df = split_stats[split].copy()
        s_df = s_df[s_df["template"].isin(templates)]
        s_df = s_df.set_index("template").reindex(templates).fillna(0.0)

        y = np.arange(len(templates))
        err_ax = axes[row_idx, 0]
        freq_ax = axes[row_idx, 1]
        contrib_ax = axes[row_idx, 2]

        err_ax.barh(
            y,
            s_df["error"].to_numpy(dtype=float),
            color=_SPLIT_COLORS[split.value],
            alpha=0.85,
        )
        freq_ax.barh(
            y,
            s_df["freq"].to_numpy(dtype=float),
            color=_SPLIT_COLORS[split.value],
            alpha=0.85,
            label=f"{split.value} mix",
        )
        contrib_ax.barh(
            y,
            s_df["contribution"].to_numpy(dtype=float),
            color=_SPLIT_COLORS[split.value],
            alpha=0.85,
            label=f"{split.value} mix",
        )

        # Optional run-mix overlays to assess query-mix shift.
        if run_freq_df is not None and not run_freq_df.empty:
            run_map = run_freq_df.set_index("template")["run_freq"]
            run_freq = (
                run_map.reindex(templates).fillna(0.0).to_numpy(dtype=float)
            )
            split_err = s_df["error"].to_numpy(dtype=float)
            run_contrib = run_freq * split_err

            freq_ax.scatter(
                run_freq,
                y,
                marker="D",
                s=32,
                color="black",
                label=f"run {run_id} mix",
                zorder=3,
            )
            contrib_ax.scatter(
                run_contrib,
                y,
                marker="D",
                s=32,
                color="black",
                label=f"run {run_id} mix",
                zorder=3,
            )

            split_weighted = float(np.sum(s_df["contribution"].to_numpy(dtype=float)))
            run_weighted = float(np.sum(run_contrib))
            delta_pct = (
                100.0 * (run_weighted - split_weighted) / max(split_weighted, 1e-9)
            )
            contrib_ax.text(
                0.98,
                0.03,
                (
                    f"split weighted={split_weighted:.3f}\n"
                    f"run-mix weighted={run_weighted:.3f}\n"
                    f"delta={delta_pct:+.1f}%"
                ),
                transform=contrib_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.9,
                },
            )

        for ax in (err_ax, freq_ax, contrib_ax):
            ax.set_yticks(y)
            ax.set_yticklabels(templates, fontsize=8)
            ax.invert_yaxis()
            ax.grid(True, axis="x", linestyle=":", alpha=0.35)

        err_ax.set_ylabel(
            f"{split.value.capitalize()}\nTemplate ID",
            fontsize=9,
        )
        freq_ax.set_ylabel("Template ID", fontsize=9)
        contrib_ax.set_ylabel("Template ID", fontsize=9)

        # Add row context on the left-most panel so viewers know which split
        # each row of bars corresponds to.
        err_ax.text(
            0.02,
            1.02,
            f"Split: {split.value.capitalize()}",
            transform=err_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

        if row_idx == len(DataSplit) - 1:
            err_ax.set_xlabel("Median Q-Error")
            freq_ax.set_xlabel("Frequency")
            contrib_ax.set_xlabel("Contribution")

        freq_ax.legend(fontsize=7, loc="lower right")
        contrib_ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        (
            f"{iconq_model_id} - Template Contribution Breakdown"
            + (f" (run mix: {run_id})" if run_id else "")
        ),
        fontsize=13,
    )
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id),
        "template_contribution_breakdown.png",
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 7: Error by cluster RPU ─────────────────────────────────────────────


def plot_error_by_cluster_rpu(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """Error magnitude and direction grouped by cluster RPU, per split.

    Top row: Q-Error boxplots by RPU (log scale, magnitude).
    Bottom row: signed log10 prediction ratio boxplots by RPU
        (positive = over-prediction, negative = under-prediction).
    """
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10), sharex="col", constrained_layout=True
    )

    for col_idx, split in enumerate(DataSplit):
        mag_ax = axes[0, col_idx]
        dir_ax = axes[1, col_idx]

        df = _add_cluster_rpu(split_dfs[split])
        df = df.dropna(subset=["cluster_rpu", "y", "y_pred_mean", "q_error"])

        if df.empty:
            for ax in (mag_ax, dir_ax):
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
            continue

        df["cluster_rpu"] = df["cluster_rpu"].astype(int)
        df["signed_log_ratio"] = np.log10(
            np.maximum(df["y_pred_mean"].astype(float), 1e-9)
            / np.maximum(df["y"].astype(float), 1e-9)
        )

        rpus = sorted(df["cluster_rpu"].unique().tolist())
        x_positions = np.arange(1, len(rpus) + 1)

        mag_data = [
            df.loc[df["cluster_rpu"] == r, "q_error"].to_numpy(dtype=float)
            for r in rpus
        ]
        dir_data = [
            df.loc[df["cluster_rpu"] == r, "signed_log_ratio"].to_numpy(
                dtype=float
            )
            for r in rpus
        ]

        # Magnitude boxplots (Q-Error).
        mag_bp = mag_ax.boxplot(
            mag_data,
            positions=x_positions,
            widths=0.6,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.3),
            whiskerprops=dict(lw=0.8),
            capprops=dict(lw=0.8),
            flierprops=dict(marker=".", markersize=2, alpha=0.25),
        )
        for patch in mag_bp["boxes"]:
            patch.set_facecolor(_SPLIT_COLORS[split.value])
            patch.set_alpha(0.75)
        mag_ax.set_yscale("log")
        mag_ax.set_ylabel("Q-Error" if split == DataSplit.TRAIN else "")
        mag_ax.set_title(f"{split.value.capitalize()}  (N={len(df):,})")
        mag_ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        # Direction boxplots (signed log ratio).
        dir_bp = dir_ax.boxplot(
            dir_data,
            positions=x_positions,
            widths=0.6,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.3),
            whiskerprops=dict(lw=0.8),
            capprops=dict(lw=0.8),
            flierprops=dict(marker=".", markersize=2, alpha=0.25),
        )
        for patch in dir_bp["boxes"]:
            patch.set_facecolor(_SPLIT_COLORS[split.value])
            patch.set_alpha(0.75)

        dir_ax.axhline(0.0, color=Palette.dark_red, linestyle="--", lw=1.0)
        dir_ax.set_ylabel(
            "log10(pred/actual)" if split == DataSplit.TRAIN else ""
        )
        dir_ax.set_xlabel("Cluster RPU")
        dir_ax.set_xticks(x_positions)
        dir_ax.set_xticklabels([str(r) for r in rpus])
        dir_ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"{iconq_model_id} - Error Magnitude/Direction by Cluster RPU",
        fontsize=12,
    )
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id), "error_by_cluster_rpu.png"
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 8: Error CDF by cluster RPU ─────────────────────────────────────────


def plot_error_cdf_by_cluster_rpu(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """CDF views of error magnitude and direction grouped by cluster RPU.

    Top row: Q-Error CDF by RPU (log x-axis).
    Bottom row: signed log10(pred/actual) CDF by RPU.
    """
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10), sharey="row", constrained_layout=True
    )
    palette = dict(Palette.rpu_to_color())

    for col_idx, split in enumerate(DataSplit):
        mag_ax = axes[1, col_idx]
        dir_ax = axes[0, col_idx]

        df = _add_cluster_rpu(split_dfs[split])
        df = df.dropna(subset=["cluster_rpu", "y", "y_pred_mean", "q_error"])

        if df.empty:
            for ax in (mag_ax, dir_ax):
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            continue

        df["cluster_rpu"] = df["cluster_rpu"].astype(int)
        df["factor_error"] = np.maximum(
            df["y_pred_mean"].astype(float), 1e-9
        ) / np.maximum(df["y"].astype(float), 1e-9)

        plot_grouped_cdf(
            mag_ax,
            df,
            value_col="q_error",
            group_col="cluster_rpu",
            palette=palette,
            xlabel="Q-Error",
            ylabel="Fraction <= x" if split == DataSplit.TRAIN else "",
            log_x=True,
            legend_fontsize=7,
        )

        plot_grouped_cdf(
            dir_ax,
            df,
            value_col="factor_error",
            group_col="cluster_rpu",
            palette=palette,
            title=f"{split.value.capitalize()}  (N={len(df):,})",
            xlabel="Factor error (predicted / actual)",
            ylabel="Fraction <= x" if split == DataSplit.TRAIN else "",
            log_x=True,
            legend_fontsize=7,
        )

        q_summary_lines = build_percentile_summary_lines(
            df,
            group_col="cluster_rpu",
            value_col="q_error",
            quantiles=(0.50, 0.90, 0.95),
            group_header="RPU",
        )
        add_monospace_summary_box(mag_ax, q_summary_lines, fontsize=7)

        direction_summary_lines = build_direction_summary_lines(
            df,
            group_col="cluster_rpu",
            actual_col="y",
            predicted_col="y_pred_mean",
            group_header="RPU",
        )
        add_monospace_summary_box(dir_ax, direction_summary_lines, fontsize=7)

    fig.suptitle(
        f"{iconq_model_id} - Error CDF by Cluster RPU",
        fontsize=12,
    )
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id),
        "error_cdf_by_cluster_rpu.png",
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path


# ── Plot 9: Signed error heatmap (RPU × concurrency) ───────────────────────


def plot_signed_error_heatmap_rpu_x_concurrency(
    split_dfs: dict[DataSplit, pd.DataFrame],
    iconq_model_id: str,
) -> tuple[Figure, str]:
    """3x3 heatmap grid: rows=P25/P50/P75, cols=train/val/test.

    Within each panel: rows=RPU, cols=concurrency bins,
    value=percentile(log10(pred/actual)) over queries in that cell.
    """
    enriched: dict[DataSplit, pd.DataFrame] = {}
    all_rpus: set[int] = set()
    percentiles = [25, 50, 75]

    for split in DataSplit:
        df = _add_cluster_rpu(_add_concurrency_bins(split_dfs[split]))
        df = df.dropna(subset=["cluster_rpu", "conc_bin", "y", "y_pred_mean"])
        if not df.empty:
            df = df.copy()
            df["cluster_rpu"] = df["cluster_rpu"].astype(int)
            df["signed_log_ratio"] = np.log10(
                np.maximum(df["y_pred_mean"].astype(float), 1e-9)
                / np.maximum(df["y"].astype(float), 1e-9)
            )
            all_rpus.update(int(r) for r in df["cluster_rpu"].unique().tolist())
        enriched[split] = df

    sorted_rpus = sorted(all_rpus)
    n_rpus = len(sorted_rpus)
    n_bins = len(_CONC_LABELS)

    fig_height = min(max(5, n_rpus * 0.45), 14)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, max(9, fig_height * 2.2)),
        sharex=True,
        sharey=True,
    )

    grids: dict[tuple[DataSplit, int], np.ndarray] = {}
    for split in DataSplit:
        df = enriched[split]
        for p in percentiles:
            grid = np.full((n_rpus, n_bins), np.nan)
            if not df.empty:
                for i, rpu in enumerate(sorted_rpus):
                    for j, bin_label in enumerate(_CONC_LABELS):
                        vals = df.loc[
                            (df["cluster_rpu"] == rpu)
                            & (df["conc_bin"] == bin_label),
                            "signed_log_ratio",
                        ].to_numpy(dtype=float)
                        if len(vals) > 0:
                            grid[i, j] = float(np.percentile(vals, p))
            grids[(split, p)] = grid

    all_vals = (
        np.concatenate(
            [g[~np.isnan(g)] for g in grids.values() if np.any(~np.isnan(g))]
        )
        if any(np.any(~np.isnan(g)) for g in grids.values())
        else np.array([])
    )
    if len(all_vals) > 0:
        vmax = max(0.1, float(np.nanpercentile(np.abs(all_vals), 95)))
    else:
        vmax = 1.0

    cmap = plt.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")

    for row_idx, p in enumerate(percentiles):
        for col_idx, split in enumerate(DataSplit):
            ax = axes[row_idx, col_idx]
            grid = grids[(split, p)]

            im = ax.imshow(
                np.ma.masked_invalid(grid),
                aspect="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                origin="upper",
            )

            ax.set_xticks(range(n_bins))
            if row_idx == len(percentiles) - 1:
                ax.set_xticklabels(
                    _CONC_LABELS,
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
                ax.set_xlabel("# concurrent queries")
            else:
                ax.set_xticklabels([])

            ax.set_yticks(range(n_rpus))
            if col_idx == 0:
                ax.set_yticklabels([str(r) for r in sorted_rpus], fontsize=8)
                ax.set_ylabel(f"P{p}\nCluster RPU")
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title(
                    f"{split.value.capitalize()}  (N={len(enriched[split]):,})"
                )

            # Add sample count annotations for easier trust calibration.
            df = enriched[split]
            if not df.empty:
                for i, rpu in enumerate(sorted_rpus):
                    for j, bin_label in enumerate(_CONC_LABELS):
                        n = int(
                            (
                                (df["cluster_rpu"] == rpu)
                                & (df["conc_bin"] == bin_label)
                            ).sum()
                        )
                        if n > 0 and not np.isnan(grid[i, j]):
                            ax.text(
                                j,
                                i,
                                str(n),
                                ha="center",
                                va="center",
                                fontsize=5,
                                color="black",
                                alpha=0.5,
                            )

    fig.colorbar(
        im,
        ax=axes,
        label="log10(pred/actual)",
        fraction=0.02,
        pad=0.02,
    )

    fig.suptitle(
        (
            f"{iconq_model_id} - Signed Error Heatmap "
            "(RPU x Concurrency, P25/P50/P75)"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    save_path = os.path.join(
        IconqModel.default_save_dir(iconq_model_id),
        "signed_error_heatmap_rpu_x_concurrency.png",
    )
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig, save_path
