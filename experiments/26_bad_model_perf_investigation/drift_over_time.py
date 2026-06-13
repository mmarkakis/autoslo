from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from autoslo.filesystem.path_utils import get_runs_path
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.models.iconq_model import IconqModel
from autoslo.visualizations.colors import Palette
from autoslo.workload_execution.trace import Trace

_RUN_LOG_COLS = ["run_id", "config_id", "workload_id"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build prediction-accuracy drift-over-time plots by RPU from "
            "run logs. Optionally compute counterfactual predictions from "
            "IconQ models."
        )
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Ignore any existing cache and rebuild from run_log.csv.",
    )
    parser.add_argument(
        "--iconq_model_ids",
        nargs="+",
        default=[],
        help=(
            "Optional space-separated list of IconQ model IDs to compute "
            "counterfactual predictions. Predictions are cached in the "
            "mega_df as iconq_{model_id}_* columns."
        ),
    )
    return parser.parse_args()


def _load_run_ids_from_run_log() -> list[str]:
    run_log_path = Path(get_runs_path()) / "run_log.csv"
    if not run_log_path.exists():
        raise FileNotFoundError(f"run_log.csv not found at {run_log_path}")

    run_log_df = pd.read_csv(run_log_path, dtype={"run_id": str})
    for col in ["run_id", "config_id"]:
        if col not in run_log_df.columns:
            raise ValueError(f"run_log.csv is missing required column: {col}")

    # Filter only run_ids where the config id includes neither "training" nor "benchmarking."
    run_ids = sorted(
        run_log_df[
            ~run_log_df["config_id"].str.contains(
                "training|benchmarking", na=False
            )
        ]["run_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not run_ids:
        raise ValueError("run_log.csv has no run_id rows.")
    return run_ids


def _load_run_log_metadata() -> pd.DataFrame:
    """Return a DataFrame with run_id, workload_id, and redshift_version columns."""
    run_log_path = Path(get_runs_path()) / "run_log.csv"
    run_log_df = pd.read_csv(run_log_path, dtype={"run_id": str})
    run_log_df["run_id"] = run_log_df["run_id"].astype(str)
    if "workload_id" not in run_log_df.columns:
        run_log_df["workload_id"] = ""
    run_log_df["workload_id"] = run_log_df["workload_id"].fillna("").astype(str)

    run_log_df["redshift_version"] = run_log_df["run_id"].apply(
        lambda rid: Trace.redshift_version_for_run(rid)
    )
    return run_log_df[["run_id", "workload_id", "redshift_version"]].drop_duplicates(subset="run_id")


def _load_cache(cache_path: Path, refresh_cache: bool) -> pd.DataFrame:
    if refresh_cache or not cache_path.exists():
        return pd.DataFrame()

    cached = pd.read_parquet(cache_path)
    if "run_id" not in cached.columns:
        return pd.DataFrame()
    cached["run_id"] = cached["run_id"].astype(str)
    return cached


def _collect_missing_run_metrics(
    run_ids: list[str],
    cached_df: pd.DataFrame,
    iconq_model_ids: list[str] | None = None,
) -> pd.DataFrame:
    cached_run_ids = (
        set(cached_df["run_id"].astype(str).unique().tolist())
        if not cached_df.empty and "run_id" in cached_df.columns
        else set()
    )
    missing_run_ids = [
        run_id for run_id in run_ids if run_id not in cached_run_ids
    ]

    appended_frames: list[pd.DataFrame] = []
    for run_id in tqdm(missing_run_ids, desc="Loading missing run metrics"):
        try:
            run_df = StructuredLog.load(run_id).prediction_accuracy_df()
        except FileNotFoundError:
            print(f"Skipping run {run_id}: structured log not found")
            continue
        except Exception as exc:
            print(
                f"Skipping run {run_id}: failed to parse structured log ({exc})"
            )
            continue

        if run_df.empty:
            print(f"Skipping run {run_id}: prediction_accuracy_df is empty")
            continue

        run_df = run_df.copy()
        # Preserve query_id as a column (from index) so we can match counterfactual predictions
        run_df.reset_index(names=["query_id"], inplace=True)
        run_df["run_id"] = str(run_id)
        appended_frames.append(run_df)

    if appended_frames:
        new_df = pd.concat(appended_frames, ignore_index=True)
        if cached_df.empty:
            mega_df = new_df
        else:
            mega_df = pd.concat([cached_df, new_df], ignore_index=True)
    else:
        mega_df = cached_df

    # Augment with counterfactual predictions from IconQ models
    if iconq_model_ids and not mega_df.empty:
        mega_df = _add_counterfactual_predictions(
            mega_df, missing_run_ids if appended_frames else [], iconq_model_ids
        )

    return mega_df


def _add_counterfactual_predictions(
    mega_df: pd.DataFrame,
    new_run_ids: list[str],
    iconq_model_ids: list[str],
) -> pd.DataFrame:
    """
    For each IconQ model, compute counterfactual predictions and add columns
    to mega_df. Only computes for newly loaded runs (in new_run_ids) to avoid
    redundant computation on cached data.

    For model M, adds columns:
    - iconq_{M}_predicted_latency: predicted runtime in seconds
    - iconq_{M}_factor_error: predicted / actual
    - iconq_{M}_q_error: max(predicted/actual, actual/predicted)

    Parameters:
        mega_df: The main dataframe with (run_id, query_id, actual_latency, ...)
        new_run_ids: Run IDs that were newly loaded (empty list if using cache)
        iconq_model_ids: List of IconQ model IDs to compute predictions for

    Returns:
        mega_df augmented with new columns for each model's predictions.
    """
    if not new_run_ids:
        # No new data to compute counterfactuals for; but still add empty columns
        # if they don't exist (to ensure cache consistency)
        for model_id in iconq_model_ids:
            pred_col = f"iconq_{model_id}_predicted_latency"
            factor_col = f"iconq_{model_id}_factor_error"
            q_error_col = f"iconq_{model_id}_q_error"
            for col in [pred_col, factor_col, q_error_col]:
                if col not in mega_df.columns:
                    mega_df[col] = None
        return mega_df

    new_run_ids_set = set(new_run_ids)
    new_rows = mega_df[mega_df["run_id"].isin(new_run_ids_set)]

    for model_id in iconq_model_ids:
        pred_col = f"iconq_{model_id}_predicted_latency"
        factor_col = f"iconq_{model_id}_factor_error"
        q_error_col = f"iconq_{model_id}_q_error"

        # Initialize columns if they don't exist
        for col in [pred_col, factor_col, q_error_col]:
            if col not in mega_df.columns:
                mega_df[col] = None

        print(f"Loading IconQ model {model_id}...")
        try:
            model = IconqModel.load(model_id=model_id)
        except Exception as exc:
            print(f"  Failed to load model {model_id}: {exc}")
            continue

        # Compute predictions for each new run
        for run_id in tqdm(
            sorted(new_run_ids_set),
            desc=f"Counterfactual predictions ({model_id})",
        ):
            try:
                dataset = model.build_dataset_from_run_id(run_id)
                predictions = model.predict_from_dataset(dataset)

                # Map predictions back to mega_df rows by (run_id, query_id)
                for cluster_aware_qid, model_pred in predictions.items():
                    query_id = cluster_aware_qid.query_id
                    pred_latency_s = model_pred.overall_mean_s()

                    # Find matching row(s) in mega_df
                    mask = (
                        (mega_df["run_id"] == run_id)
                        & (mega_df["query_id"] == query_id)
                    )
                    if not mask.any():
                        continue

                    for idx in mega_df[mask].index:
                        actual_latency_s = mega_df.loc[idx, "actual_latency"]
                        if actual_latency_s <= 0:
                            continue

                        mega_df.at[idx, pred_col] = pred_latency_s
                        mega_df.at[idx, factor_col] = (
                            pred_latency_s / actual_latency_s
                        )
                        mega_df.at[idx, q_error_col] = max(
                            pred_latency_s / actual_latency_s,
                            actual_latency_s / pred_latency_s,
                        )

            except Exception as exc:
                print(
                    f"  Failed to predict {model_id} on run {run_id}: {exc}"
                )
                continue

    return mega_df


def _aggregate_percentiles(
    mega_df: pd.DataFrame,
    metric: str,
    quantile: float,
) -> pd.DataFrame:
    summary = (
        mega_df.groupby(["run_id", "rpu"], observed=True)[metric]
        .quantile(quantile)
        .rename("value")
        .reset_index()
    )
    summary["run_id"] = summary["run_id"].astype(str)
    summary["rpu"] = summary["rpu"].astype(int)
    return summary


def _plot_drift_over_time(
    mega_df: pd.DataFrame,
    run_ids: list[str],
    run_log_meta: pd.DataFrame,
    output_path: Path,
    show: bool,
    model_id: str | None = None,
) -> None:
    """Plot prediction accuracy drift over time.

    Parameters:
        mega_df: DataFrame with prediction accuracy data
        run_ids: List of run IDs for x-axis ordering
        run_log_meta: Run metadata (workload, version)
        output_path: Path to save plot
        show: Whether to display plot interactively
        model_id: Optional model ID for counterfactual predictions.
                  If None, uses historical predictions (factor_error, q_error).
                  If provided, uses iconq_{model_id}_factor_error, etc.
    """
    # Derive column names based on model_id
    if model_id is None:
        factor_col = "factor_error"
        q_error_col = "q_error"
    else:
        factor_col = f"iconq_{model_id}_factor_error"
        q_error_col = f"iconq_{model_id}_q_error"

    if mega_df.empty:
        raise ValueError("No prediction accuracy rows available to plot.")

    required_cols = {"run_id", "rpu", factor_col, q_error_col}
    missing_cols = required_cols - set(mega_df.columns)
    if missing_cols:
        raise ValueError(
            f"mega dataframe missing columns: {sorted(missing_cols)}"
        )

    mega_df = mega_df.copy()
    mega_df["run_id"] = mega_df["run_id"].astype(str)
    mega_df["rpu"] = pd.to_numeric(mega_df["rpu"], errors="coerce").astype(
        "Int64"
    )
    mega_df = mega_df.dropna(subset=["rpu", factor_col, q_error_col])
    mega_df["rpu"] = mega_df["rpu"].astype(int)

    if mega_df.empty:
        raise ValueError(
            "No valid rows after cleaning run_id/rpu/error columns."
        )

    # Per-run metadata: workload_id and redshift_version.
    meta = run_log_meta.set_index("run_id")
    run_to_workload = meta["workload_id"].to_dict()
    run_to_version = meta["redshift_version"].to_dict()

    # Version transition positions: where redshift_version changes across sorted run_ids.
    version_transitions: list[tuple[float, str, str]] = []
    prev_ver = run_to_version.get(run_ids[0])
    for i, rid in enumerate(run_ids[1:], start=1):
        ver = run_to_version.get(rid)
        if ver != prev_ver and prev_ver is not None and ver is not None:
            version_transitions.append((i - 0.5, prev_ver, ver))
        if ver is not None:
            prev_ver = ver

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True, sharey='row')

    positions = list(range(len(run_ids)))
    percentile_specs = [(0.50, "P50"), (0.90, "P90"), (0.95, "P95")]
    metrics = [(factor_col, "Factor Error"), (q_error_col, "Q-Error")]
    palette = dict(Palette.rpu_to_color())

    def _workload_prefix(wl: str) -> str:
        return wl.split("_")[0] if wl else ""

    _MARKERS = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "*"]
    sorted_workload_prefixes = sorted(
        set(_workload_prefix(wl) for wl in run_to_workload.values())
    )
    prefix_to_marker = {
        pfx: _MARKERS[i % len(_MARKERS)]
        for i, pfx in enumerate(sorted_workload_prefixes)
    }

    for row_idx, (metric, metric_name) in enumerate(metrics):
        for col_idx, (quantile, q_label) in enumerate(percentile_specs):
            ax = axes[row_idx, col_idx]
            summary = _aggregate_percentiles(
                mega_df, metric=metric, quantile=quantile
            )

            for rpu in sorted(summary["rpu"].unique().tolist()):
                sub = summary[summary["rpu"] == rpu]
                value_by_run = {
                    str(run_id): float(value)
                    for run_id, value in zip(sub["run_id"], sub["value"])
                }
                y_values = [
                    value_by_run.get(run_id, float("nan")) for run_id in run_ids
                ]
                color = palette.get(int(rpu), "black")

                # Draw the connecting line without markers.
                ax.plot(
                    positions,
                    y_values,
                    linewidth=1.5,
                    color=color,
                    label=str(rpu),
                    zorder=2,
                )
                # Draw each point individually so the marker encodes workload_id.
                for pos, run_id in zip(positions, run_ids):
                    yval = value_by_run.get(run_id, float("nan"))
                    if yval != yval:  # NaN check
                        continue
                    wl = _workload_prefix(run_to_workload.get(run_id, ""))
                    marker = prefix_to_marker.get(wl, "o")
                    ax.plot(
                        pos,
                        yval,
                        marker=marker,
                        markersize=4.5,
                        color=color,
                        linewidth=0,
                        zorder=3,
                    )
            
            # Draw a  horizontal line at 1.0 for reference.
            ax.axhline(
                1.0,
                color="0.35",
                linestyle="--",
                linewidth=1.1,
                label="Perfect"
            )

            # Draw a vertical line at each Redshift version transition and
            # annotate with the before/after version strings.
            y_min, y_max = ax.get_ylim()
            for x_trans, ver_before, ver_after in version_transitions:
                ax.axvline(
                    x_trans,
                    color=Palette.light_gray,
                    linestyle="--",
                    linewidth=1.0,
                    zorder=4,
                )
                ax.text(
                    x_trans - 0.15,
                    0.98,
                    ver_before,
                    transform=ax.get_xaxis_transform(),
                    ha="right",
                    va="top",
                    fontsize=5,
                    color="0.35",
                    rotation=90,
                )
                ax.text(
                    x_trans + 0.15,
                    0.98,
                    ver_after,
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="top",
                    fontsize=5,
                    color="0.35",
                    rotation=90,
                )

            ax.set_title(f"{metric_name} {q_label}")
            ax.set_xlabel("run_id")
            if col_idx == 0:
                ax.set_ylabel(metric_name)
            ax.set_yscale("log")

            tick_step = max(1, len(run_ids) // 20)
            tick_positions = positions[::tick_step]
            tick_labels = [run_ids[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

            ax.grid(True, which="major", axis="both", linestyle=":", alpha=0.4)
            ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.2)

            # Show minor tick marks at each integer multiple within each decade
            # (2×, 3×, …, 9×) but suppress their labels.
            ax.yaxis.set_minor_locator(
                plt.matplotlib.ticker.LogLocator(
                    base=10.0, subs=tuple(range(2, 10)), numticks=100
                )
            )
            ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())

            if row_idx == 0 and col_idx == 0:
                rpu_legend = ax.legend(
                    title="RPU", fontsize=7, loc="upper left"
                )
                ax.add_artist(rpu_legend)

                # Workload legend — marker shapes, neutral color.
                workload_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker=prefix_to_marker[pfx],
                        color="0.4",
                        linewidth=0,
                        markersize=5,
                        label=pfx if pfx else "(none)",
                    )
                    for pfx in sorted_workload_prefixes
                ]
                ax.legend(
                    handles=workload_handles,
                    title="Workload",
                    fontsize=6,
                    loc="lower left",
                )

    fig.suptitle("Prediction Error Drift Over Time by RPU", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_counterfactual_vs_log_scatter(
    mega_df: pd.DataFrame,
    model_id: str,
    output_path: Path,
    show: bool,
) -> None:
    """Plot counterfactual predictions vs log-derived predictions.

    X-axis: log-derived predicted latency (from structured log)
    Y-axis: counterfactual IconQ predicted latency
    Color: query RPU using Palette.rpu_to_color()
    """
    pred_col = f"iconq_{model_id}_predicted_latency"
    required_cols = {"predicted_latency", pred_col, "rpu"}
    missing_cols = required_cols - set(mega_df.columns)
    if missing_cols:
        raise ValueError(
            f"mega dataframe missing columns for scatter: {sorted(missing_cols)}"
        )

    df = mega_df.copy()
    df["predicted_latency"] = pd.to_numeric(
        df["predicted_latency"], errors="coerce"
    )
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df["rpu"] = pd.to_numeric(df["rpu"], errors="coerce")
    df = df.dropna(subset=["predicted_latency", pred_col, "rpu"])
    df = df[(df["predicted_latency"] > 0) & (df[pred_col] > 0)]

    if df.empty:
        raise ValueError(
            f"No valid rows for scatterplot of model {model_id}."
        )

    df["rpu"] = df["rpu"].astype(int)
    palette = dict(Palette.rpu_to_color())

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    for rpu in sorted(df["rpu"].unique().tolist()):
        sub = df[df["rpu"] == rpu]
        ax.scatter(
            sub["predicted_latency"],
            sub[pred_col],
            s=10,
            alpha=0.6,
            color=palette.get(int(rpu), "black"),
            label=str(rpu),
        )

    # Reference y=x line.
    min_v = min(df["predicted_latency"].min(), df[pred_col].min())
    max_v = max(df["predicted_latency"].max(), df[pred_col].max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        linewidth=1.1,
        color="0.35",
        label="y=x",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Log-derived predicted latency (s)")
    ax.set_ylabel(f"Counterfactual predicted latency (s) - model {model_id}")
    ax.set_title("Counterfactual vs Log-Derived Predictions")
    ax.grid(True, which="major", axis="both", linestyle=":", alpha=0.4)
    ax.grid(True, which="minor", axis="both", linestyle=":", alpha=0.2)
    ax.legend(title="RPU", fontsize=7, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved scatterplot to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_predicted_vs_actual_scatter(
    mega_df: pd.DataFrame,
    model_id: str | None,
    output_path: Path,
    show: bool,
) -> None:
    """Plot predicted vs actual latencies for one prediction source.

    When model_id is None, uses log-derived predictions.
    Otherwise uses the specified counterfactual IconQ model predictions.
    """
    if model_id is None:
        pred_col = "predicted_latency"
        source_label = "log"
    else:
        pred_col = f"iconq_{model_id}_predicted_latency"
        source_label = f"iconq_{model_id}"

    required_cols = {"actual_latency", pred_col, "rpu"}
    missing_cols = required_cols - set(mega_df.columns)
    if missing_cols:
        raise ValueError(
            "mega dataframe missing columns for predicted-vs-actual scatter: "
            f"{sorted(missing_cols)}"
        )

    df = mega_df.copy()
    df["actual_latency"] = pd.to_numeric(df["actual_latency"], errors="coerce")
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df["rpu"] = pd.to_numeric(df["rpu"], errors="coerce")
    df = df.dropna(subset=["actual_latency", pred_col, "rpu"])
    df = df[(df["actual_latency"] > 0) & (df[pred_col] > 0)]

    if df.empty:
        raise ValueError(
            f"No valid rows for predicted-vs-actual scatter of {source_label}."
        )

    df["rpu"] = df["rpu"].astype(int)
    palette = dict(Palette.rpu_to_color())

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    for rpu in sorted(df["rpu"].unique().tolist()):
        sub = df[df["rpu"] == rpu]
        ax.scatter(
            sub["actual_latency"],
            sub[pred_col],
            s=10,
            alpha=0.6,
            color=palette.get(int(rpu), "black"),
            label=str(rpu),
        )

    min_v = min(df["actual_latency"].min(), df[pred_col].min())
    max_v = max(df["actual_latency"].max(), df[pred_col].max())
    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        linewidth=1.1,
        color="0.35",
        label="y=x",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual latency (s)")
    ax.set_ylabel("Predicted latency (s)")
    ax.set_title(f"Predicted vs Actual Latency ({source_label})")
    ax.grid(True, which="major", axis="both", linestyle=":", alpha=0.4)
    ax.grid(True, which="minor", axis="both", linestyle=":", alpha=0.2)
    ax.legend(title="RPU", fontsize=7, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved predicted-vs-actual scatterplot to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _get_available_model_ids(mega_df: pd.DataFrame) -> list[str | None]:
    """
    Get list of model IDs with valid predictions in mega_df.
    Always includes None (for historical predictions) plus any counterfactual model IDs.

    Returns:
        List starting with None, followed by sorted counterfactual model IDs.
    """
    model_ids = set()
    for col in mega_df.columns:
        if col.startswith("iconq_") and col.endswith("_factor_error"):
            # Extract model_id from "iconq_{model_id}_factor_error"
            model_id = col[6:-13]  # Remove "iconq_" prefix and "_factor_error" suffix
            model_ids.add(model_id)
    return [None] + sorted(model_ids)


def main() -> None:
    args = _parse_args()

    script_dir = Path(__file__).resolve().parent
    cache_path = script_dir / "prediction_accuracy_mega_df.parquet"
    plot_path = script_dir / "prediction_error_drift_over_time.png"

    run_ids = _load_run_ids_from_run_log()
    run_log_meta = _load_run_log_metadata()
    cached_df = _load_cache(cache_path, refresh_cache=args.refresh_cache)
    mega_df = _collect_missing_run_metrics(
        run_ids, cached_df, iconq_model_ids=args.iconq_model_ids or None
    )

    if mega_df.empty:
        raise ValueError("No prediction accuracy data was collected from runs.")

    # Keep only runs currently present in run_log.csv while preserving cache value.
    mega_df["run_id"] = mega_df["run_id"].astype(str)
    mega_df = mega_df[mega_df["run_id"].isin(set(run_ids))].copy()
    mega_df.to_parquet(cache_path, index=False)
    print(f"Saved mega dataframe to: {cache_path} ({len(mega_df):,} rows)")

    # Generate plots for all available models (None=historical + any counterfactual models)
    model_ids = _get_available_model_ids(mega_df)
    for model_id in model_ids:
        if model_id is None:
            output_path = script_dir / "prediction_error_drift_over_time.png"
        else:
            output_path = (
                script_dir / f"prediction_error_drift_over_time_iconq_{model_id}.png"
            )

        try:
            _plot_drift_over_time(
                mega_df,
                run_ids=run_ids,
                run_log_meta=run_log_meta,
                output_path=output_path,
                show=args.show,
                model_id=model_id,
            )

            if model_id is None:
                scatter_actual_output_path = (
                    script_dir / "prediction_scatter_log_vs_actual.png"
                )
            else:
                scatter_actual_output_path = (
                    script_dir
                    / f"prediction_scatter_iconq_{model_id}_vs_actual.png"
                )
            _plot_predicted_vs_actual_scatter(
                mega_df=mega_df,
                model_id=model_id,
                output_path=scatter_actual_output_path,
                show=args.show,
            )

            if model_id is not None:
                scatter_output_path = (
                    script_dir
                    / f"prediction_scatter_iconq_{model_id}_vs_log.png"
                )
                _plot_counterfactual_vs_log_scatter(
                    mega_df=mega_df,
                    model_id=model_id,
                    output_path=scatter_output_path,
                    show=args.show,
                )
        except Exception as exc:
            print(f"Failed to generate plot for model {model_id}: {exc}")


if __name__ == "__main__":
    main()
