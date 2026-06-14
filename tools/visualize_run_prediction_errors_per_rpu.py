from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from autoslo.filesystem.structured_log import StructuredLog
from autoslo.models.iconq_model import IconqModel
from autoslo.visualizations.colors import Palette
from autoslo.visualizations.prediction_error_cdf import (
    add_monospace_summary_box,
    build_percentile_summary_lines,
    plot_grouped_cdf,
)
from autoslo.visualizations.prediction_error_scatter import (
    plot_factor_error_vs_effective_concurrency,
    plot_factor_error_vs_interarrival,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-RPU prediction errors from a structured log and "
            "save the figure to the run directory."
        )
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID (e.g. 1780875805584).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively.",
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        default=None,
        help=(
            "Optional IconQ model ID. When provided, recompute per-query "
            "predictions for this run using the model instead of log-derived "
            "predictions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log = StructuredLog.load(args.run_id)
    results_df = log.prediction_accuracy_df()
    if results_df.empty:
        raise ValueError(
            "No complete query records with predictions were found."
        )

    results_df = results_df.copy()
    if "query_id" not in results_df.columns:
        results_df.reset_index(names=["query_id"], inplace=True)

    source_label = "log-derived"
    if args.iconq_model_id is not None:
        model = IconqModel.load(model_id=args.iconq_model_id)
        dataset = model.build_dataset_from_run_id(args.run_id)
        predictions = model.predict_from_dataset(dataset)
        pred_by_query_id = {
            str(cluster_aware_qid.query_id): float(pred.overall_mean_s())
            for cluster_aware_qid, pred in predictions.items()
        }

        results_df["query_id"] = results_df["query_id"].astype(str)
        matched = results_df["query_id"].isin(pred_by_query_id)
        if not matched.any():
            raise ValueError(
                f"No query IDs from run {args.run_id} matched predictions "
                f"from model {args.iconq_model_id}."
            )

        # Keep only rows with counterfactual predictions to avoid mixing sources.
        results_df = results_df.loc[matched].copy()
        results_df["predicted_latency"] = results_df["query_id"].map(
            pred_by_query_id
        )

        # Recompute error metrics for the substituted predictions.
        results_df["actual_latency"] = pd.to_numeric(
            results_df["actual_latency"], errors="coerce"
        )
        results_df["predicted_latency"] = pd.to_numeric(
            results_df["predicted_latency"], errors="coerce"
        )
        results_df = results_df.dropna(subset=["actual_latency", "predicted_latency"])
        results_df = results_df[
            (results_df["actual_latency"] > 0)
            & (results_df["predicted_latency"] > 0)
        ]

        actual = results_df["actual_latency"].to_numpy(dtype=float)
        pred = results_df["predicted_latency"].to_numpy(dtype=float)
        factor = pred / actual
        is_censored = (
            results_df["is_censored_target"].fillna(False).astype(bool).to_numpy()
            if "is_censored_target" in results_df.columns
            else np.zeros(len(results_df), dtype=bool)
        )

        results_df["factor_error"] = factor
        symmetric_q = np.maximum(factor, 1.0 / factor)
        censored_q = np.maximum(actual / pred, 1.0)
        results_df["q_error"] = np.where(is_censored, censored_q, symmetric_q)

        source_label = f"iconq_{args.iconq_model_id}"
        print(
            f"Using {len(results_df):,} matched queries with predictions from "
            f"model {args.iconq_model_id}."
        )

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    plot_grouped_cdf(
        axes[0],
        results_df,
        value_col="factor_error",
        group_col="rpu",
        palette=dict(Palette.rpu_to_color()),
        title="Factor Error by RPU",
        xlabel="Factor error: predicted / actual",
        ylabel="Cumulative probability",
        log_x=True,
        legend_fontsize=8,
    )
    plot_grouped_cdf(
        axes[1],
        results_df,
        value_col="q_error",
        group_col="rpu",
        palette=dict(Palette.rpu_to_color()),
        title="Q-Error by RPU",
        xlabel="Q-error: max(actual / predicted, predicted / actual)",
        ylabel="Cumulative probability",
        log_x=True,
        legend_fontsize=8,
    )

    q_summary = build_percentile_summary_lines(
        results_df,
        group_col="rpu",
        value_col="q_error",
        quantiles=(0.50, 0.90, 0.95),
        group_header="RPU",
    )
    add_monospace_summary_box(axes[1], q_summary, fontsize=9)

    # direction_summary = build_direction_summary_lines(
    #     results_df,
    #     group_col="rpu",
    #     actual_col="actual_latency",
    #     predicted_col="predicted_latency",
    #     group_header="RPU",
    # )
    factor_summary = build_percentile_summary_lines(
        results_df,
        group_col="rpu",
        value_col="factor_error",
        quantiles=(0.25, 0.50, 0.75),
        group_header="RPU",
    )
    add_monospace_summary_box(axes[0], factor_summary, fontsize=9)

    fig.suptitle(
        f"Prediction Errors by RPU — Run {args.run_id} ({source_label})",
        fontsize=15,
    )

    out_name = "prediction_errors_per_rpu.png"
    if args.iconq_model_id is not None:
        out_name = f"prediction_errors_per_rpu_iconq_{args.iconq_model_id}.png"

    output_path = Path("data") / "runs" / args.run_id / out_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")

    latencies_df = log.query_latencies()[
        ["query_id", "arrival_s", "completion_s"]
    ].copy()
    cluster_events = log.df[
        log.df["event_type"].isin(["query_routed", "completion"])
    ][["query_id", "cluster_name", "rel_time_s"]].copy()
    cluster_events["query_id"] = cluster_events["query_id"].astype(str)
    cluster_events["cluster_name"] = (
        cluster_events["cluster_name"].fillna("").astype(str)
    )
    cluster_events = cluster_events[cluster_events["cluster_name"] != ""]
    cluster_by_query = (
        cluster_events.sort_values(["query_id", "rel_time_s"])
        .drop_duplicates(subset=["query_id"], keep="first")
        [["query_id", "cluster_name"]]
    )
    latencies_df["query_id"] = latencies_df["query_id"].astype(str)
    latencies_df = latencies_df.merge(cluster_by_query, on="query_id", how="left")
    arrivals_df = latencies_df[["query_id", "arrival_s", "cluster_name"]].copy()
    scatter_fig, scatter_ax = plt.subplots(
        1, 1, figsize=(8.5, 6.5), constrained_layout=True
    )
    plotted_df = plot_factor_error_vs_interarrival(
        scatter_ax,
        results_df=results_df,
        arrivals_df=arrivals_df,
        title=(
            "Factor Error vs Interarrival Time "
            f"— Run {args.run_id} ({source_label})"
        ),
        color_col="rpu",
    )

    scatter_out_name = "factor_error_vs_interarrival_scatter.png"
    if args.iconq_model_id is not None:
        scatter_out_name = (
            "factor_error_vs_interarrival_scatter_"
            f"iconq_{args.iconq_model_id}.png"
        )
    scatter_output_path = Path("data") / "runs" / args.run_id / scatter_out_name
    scatter_output_path.parent.mkdir(parents=True, exist_ok=True)
    scatter_fig.savefig(scatter_output_path, dpi=200)
    print(
        "Saved factor-error-vs-interarrival scatter to: "
        f"{scatter_output_path} (N={len(plotted_df):,})"
    )

    eff_fig, eff_ax = plt.subplots(1, 1, figsize=(8.5, 6.5), constrained_layout=True)
    eff_plot_df = plot_factor_error_vs_effective_concurrency(
        eff_ax,
        results_df=results_df,
        query_windows_df=latencies_df,
        title=(
            "Factor Error vs Effective Concurrency "
            f"— Run {args.run_id} ({source_label})"
        ),
        color_col="rpu",
    )

    eff_out_name = "factor_error_vs_effective_concurrency_scatter.png"
    if args.iconq_model_id is not None:
        eff_out_name = (
            "factor_error_vs_effective_concurrency_scatter_"
            f"iconq_{args.iconq_model_id}.png"
        )
    eff_output_path = Path("data") / "runs" / args.run_id / eff_out_name
    eff_output_path.parent.mkdir(parents=True, exist_ok=True)
    eff_fig.savefig(eff_output_path, dpi=200)
    print(
        "Saved factor-error-vs-effective-concurrency scatter to: "
        f"{eff_output_path} (N={len(eff_plot_df):,})"
    )

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(scatter_fig)
        plt.close(eff_fig)


if __name__ == "__main__":
    main()
