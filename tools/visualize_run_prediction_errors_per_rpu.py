from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autoslo.clusters.cluster import Cluster
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.visualizations.colors import Palette


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-RPU prediction errors from a structured log and "
            "save the figure to the run directory."
        )
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID (e.g. 1780875805584).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Structured log source: run ID, run directory, or explicit "
            "structured_log.parquet path. If omitted, --run-id is required."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output PNG path. Defaults to <run_dir>/prediction_errors_per_rpu.png "
            "when source resolves to a file path."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively.",
    )
    return parser.parse_args()


def _safe_details_latency(details: Any) -> float | None:
    if details is None:
        return None
    if isinstance(details, dict):
        value = details.get("latency_s")
    else:
        try:
            parsed = json.loads(details)
        except (TypeError, json.JSONDecodeError):
            return None
        value = parsed.get("latency_s") if isinstance(parsed, dict) else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_results_df(log_df: pd.DataFrame) -> pd.DataFrame:
    query_data: dict[Any, dict[str, float | int | None]] = {}
    for _, row in log_df.iterrows():
        query_id = row.get("query_id")
        event_type = row.get("event_type")
        if query_id is None or event_type is None:
            continue

        if query_id not in query_data:
            query_data[query_id] = {
                "arrival_time": None,
                "completion_time": None,
                "predicted_latency": None,
                "rpu": None,
            }

        if event_type == "arrival":
            query_data[query_id]["arrival_time"] = float(row["rel_time_s"])
        elif event_type == "completion":
            query_data[query_id]["completion_time"] = float(row["rel_time_s"])
        elif event_type in {"query_routed", "latency_update"}:
            pred = _safe_details_latency(row.get("details"))
            if pred is not None:
                query_data[query_id]["predicted_latency"] = pred
            if event_type == "query_routed" and row.get("cluster_name") is not None:
                query_data[query_id]["rpu"] = Cluster.rpu_for_cluster_name(
                    row["cluster_name"]
                )

    results_df = pd.DataFrame.from_dict(query_data, orient="index")
    if results_df.empty:
        return results_df

    results_df["actual_latency"] = (
        results_df["completion_time"] - results_df["arrival_time"]
    )
    results_df = results_df.dropna(
        subset=["actual_latency", "predicted_latency", "rpu"]
    )
    results_df = results_df[
        (results_df["actual_latency"] > 0) & (results_df["predicted_latency"] > 0)
    ]

    results_df["abs_error"] = (
        results_df["actual_latency"] - results_df["predicted_latency"]
    ).abs()
    results_df["factor_error"] = (
        results_df["predicted_latency"] / results_df["actual_latency"]
    )
    results_df["q_error"] = results_df[["factor_error"]].apply(
        lambda row: max(row["factor_error"], 1.0 / row["factor_error"]),
        axis=1,
    )
    results_df["rpu"] = results_df["rpu"].astype(int)
    return results_df


def _plot_error_cdf(
    ax: plt.Axes,
    data: pd.DataFrame,
    palette: dict[int, str],
    x_col: str,
    title: str,
    xlabel: str,
) -> None:
    sns.ecdfplot(
        data=data,
        x=x_col,
        hue="rpu",
        ax=ax,
        palette=palette,
        log_scale=True,
        linewidth=2.0,
    )

    ax.axvline(1, color="0.35", linestyle="--", linewidth=1.1, label="Perfect")
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)


def _default_output_path(log: StructuredLog, source_hint: str) -> Path:
    if log.path is not None:
        return log.path.parent / "prediction_errors_per_rpu.png"
    return Path("data") / "runs" / source_hint / "prediction_errors_per_rpu.png"


def main() -> None:
    args = _parse_args()
    source = args.source or args.run_id
    if source is None:
        raise ValueError("Provide --run-id or --source.")

    log = StructuredLog.load(source)
    results_df = _build_results_df(log.df)
    if results_df.empty:
        raise ValueError("No complete query records with predictions were found.")

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    palette = {
        4: Palette.light_green,
        8: Palette.light_blue,
        16: Palette.light_red,
        32: Palette.light_purple,
    }

    _plot_error_cdf(
        axes[0],
        results_df,
        palette,
        "factor_error",
        "Factor Error by RPU",
        "Factor error: predicted / actual",
    )
    _plot_error_cdf(
        axes[1],
        results_df,
        palette,
        "q_error",
        "Q-Error by RPU",
        "Q-error: max(actual / predicted, predicted / actual)",
    )

    q_summary = ["RPU   P50    P90    P95"]
    for rpu, group in results_df.groupby("rpu", sort=True):
        qs = group["q_error"].quantile([0.50, 0.90, 0.95])
        q_summary.append(
            f"{rpu:>3}  {qs.loc[0.50]:>4.2f}  {qs.loc[0.90]:>5.2f}  {qs.loc[0.95]:>5.2f}"
        )

    axes[1].text(
        0.97,
        0.04,
        "\n".join(q_summary),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.85,
        },
    )

    direction_summary = ["RPU   Under    Over   Total Qs"]
    for rpu, group in results_df.groupby("rpu", sort=True):
        underpredicted = (
            (group["actual_latency"] > group["predicted_latency"]).mean() * 100
        )
        overpredicted = (
            (group["actual_latency"] < group["predicted_latency"]).mean() * 100
        )
        direction_summary.append(
            f"{rpu:>3}  {underpredicted:>5.2f}%  {overpredicted:>5.2f}%  {len(group):>7}"
        )

    axes[0].text(
        0.97,
        0.04,
        "\n".join(direction_summary),
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.85,
        },
    )

    run_label = args.run_id or source
    fig.suptitle(f"Prediction Errors by RPU — Run {run_label}", fontsize=15)

    output_path = Path(args.output) if args.output else _default_output_path(log, run_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()