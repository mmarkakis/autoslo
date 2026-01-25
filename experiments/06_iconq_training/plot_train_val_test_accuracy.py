import argparse
import os
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.query_timeline import QueryTimeline
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.utils.colors import Palette
from autoslo.workload_execution.trace import Trace


def plot_true_predicted(
    ax: plt.Axes,
    title: str,
    true_y: pd.Series,
    predicted_y: dict[str, ModelPrediction],
):
    color_if_alone = Palette.dark_green
    color_if_overlapped = Palette.dark_blue

    points_x = []
    points_y = []
    colors = []
    for query_id, true_latency in true_y.items():
        predicted_latency = predicted_y[query_id]  # type: ignore
        points_x.append(true_latency)
        points_y.append(predicted_latency.overall_mean_s())
        if (
            predicted_latency.metadata
            and predicted_latency.metadata.get(
                "num_other_concurrent_queries", 0
            )
            > 0
        ):
            colors.append(color_if_overlapped)
        else:
            colors.append(color_if_alone)

    ax.scatter(points_x, points_y, c=colors, alpha=0.6)
    ax.set_xlabel("True Latency (s)")
    ax.set_ylabel("Predicted Latency (s)")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(
        [0, max(points_x)], [0, max(points_x)], color="red", linestyle="--"
    )  # Diagonal line


def plot_qerror(
    ax: plt.Axes,
    title: str,
    true_y: pd.Series,
    predicted_y: dict[str, ModelPrediction],
):
    qerrors = []
    for query_id, true_latency in true_y.items():
        predicted_latency = predicted_y[query_id]  # type: ignore
        pred_mean = predicted_latency.overall_mean_s()
        if true_latency == 0 and pred_mean == 0:
            qerror = 1.0
        elif true_latency == 0 or pred_mean == 0:
            qerror = float("inf")
        else:
            qerror = max(true_latency / pred_mean, pred_mean / true_latency)
        qerrors.append(qerror)

    # Plot a CDF of the Q-Errors using a lineplot
    cdf = pd.Series(qerrors).value_counts().sort_index().cumsum()
    cdf = cdf / cdf.iloc[-1]  # type: ignore
    ax.plot(cdf.index, cdf.values, color=Palette.dark_blue)  # type: ignore

    # At the bottom right of the plot, report the p50, p90 and p95 Q-Errors
    p50 = pd.Series(qerrors).quantile(0.5)
    p90 = pd.Series(qerrors).quantile(0.9)
    p95 = pd.Series(qerrors).quantile(0.95)
    ax.text(
        0.6,
        0.2,
        f"p50: {p50:.2f}\np90: {p90:.2f}\np95: {p95:.2f}",
        transform=ax.transAxes,
    )

    ax.set_xlabel("Q-Error")
    ax.set_ylabel("Frequency")
    ax.set_title(title)


def plot_qerror_single_barchart(
    ax: plt.Axes,
    title: str,
    split_true_y: dict[str, pd.Series],
    split_predicted_y: dict[str, dict[str, ModelPrediction]],
    include_censored: Optional[bool] = None,
):

    # Plot a grouped bar chart of p50, p90 and p95 Q-Errors for each split
    metrics = {"p50": 0.5, "p90": 0.9, "p95": 0.95}
    x = np.arange(len(metrics))  # the label locations
    width = 0.25  # the width of the bars

    fontsize = 20

    colors = {
        "train": Palette.light_green,
        "val": Palette.light_blue,
        "test": Palette.light_red,
    }

    for split_idx, split in enumerate(["train", "val", "test"]):
        qerrors = []
        true_y = split_true_y[split]
        predicted_y = split_predicted_y[split]
        for query_id, true_latency in true_y.items():
            predicted_latency = predicted_y[query_id]  # type: ignore
            pred_mean = predicted_latency.overall_mean_s()

            is_censored = predicted_latency.metadata.get(
                "target_is_lower_bound", False
            )

            if (include_censored is not None) and (
                is_censored != include_censored
            ):
                continue

            if true_latency == 0 and pred_mean == 0:
                qerror = 1.0
            elif true_latency == 0 or pred_mean == 0:
                qerror = float("inf")
            else:
                qerror = max(true_latency / pred_mean, pred_mean / true_latency)
            qerrors.append(qerror)

        p50 = pd.Series(qerrors).quantile(metrics["p50"])
        p90 = pd.Series(qerrors).quantile(metrics["p90"])
        p95 = pd.Series(qerrors).quantile(metrics["p95"])

        ax.bar(
            x + split_idx * width,
            [p50, p90, p95],
            width,
            label=split.title(),
            color=colors[split],
        )

        # Add a text label above each bar displaying its height
        for metric_idx, metric in enumerate(metrics.keys()):
            height = [p50, p90, p95][metric_idx]
            ax.text(
                x[metric_idx] + split_idx * width,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    ax.set_xticks(x + width)
    ax.set_ylim(bottom=1, top=ax.get_ylim()[1] * 1.1)
    ax.set_xticklabels(metrics.keys(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
    ax.set_ylabel("Q-Error", fontsize=fontsize)
    ax.set_title(title)
    ax.legend(fontsize=fontsize)


def plot_over_under_for_censored(
    ax: plt.Axes,
    title: str,
    split_true_y: dict[str, pd.Series],
    split_predicted_y: dict[str, dict[str, ModelPrediction]],
):

    # Plot a dashed vertical line at x=1.0
    ax.axvline(x=1.0, color=Palette.gray, linestyle="--")
    fontsize = 20

    colors = {
        "train": Palette.light_green,
        "val": Palette.light_blue,
        "test": Palette.light_red,
    }

    # Plot one line per split. The y axis should be a cdf, and the x axis should
    # be the relative value of predicted / true latency for censored points only.
    for split in ["train", "val", "test"]:
        relative_errors = []
        true_y = split_true_y[split]
        predicted_y = split_predicted_y[split]
        for query_id, true_latency in true_y.items():
            predicted_latency = predicted_y[query_id]  # type: ignore
            pred_mean = predicted_latency.overall_mean_s()

            is_censored = predicted_latency.metadata.get(
                "target_is_lower_bound", False
            )

            if not is_censored:
                continue

            if true_latency == 0 and pred_mean == 0:
                relative_error = 1.0
            elif true_latency == 0:
                relative_error = float("inf")
            else:
                relative_error = pred_mean / true_latency
                print(
                    f"Split: {split}, Query_id: {query_id}, Predicted: {pred_mean}, True: {true_latency}, Relative Error: {relative_error}"
                )

            relative_errors.append(relative_error)

        # Plot a CDF of the relative errors using a lineplot
        cdf = pd.Series(relative_errors).value_counts().sort_index().cumsum()
        cdf = cdf / cdf.iloc[-1]  # type: ignore
        ax.plot(cdf.index, cdf.values, label=split.title(), color=colors[split])  # type: ignore

    ax.set_xlabel("Predicted / True Latency", fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)
    ax.set_title(title)
    ax.legend(fontsize=fontsize)


def main(iconq_model_id: str, hide_plot_title: bool):
    model = IconqModel.load(
        model_id=iconq_model_id,
    )
    use_stage_for_isolated_queries = model._train_config_sequence[
        -1
    ].use_stage_for_isolated_queries

    trained_on_log_runtime = model.trained_on_log_runtime

    model_dir = os.path.join(
        pu.get_data_path(),
        "iconq_models",
        iconq_model_id,
    )

    overall_dataset_path = os.path.join(model_dir, "dataset.pkl")
    dataset = ConcurrentQueryDataset.load_from(overall_dataset_path)
    split_datasets = {}
    for split in ["train", "val", "test"]:
        with open(
            os.path.join(model_dir, f"{split}_indices.pkl"),
            "rb",
        ) as f:
            indices = pickle.load(f)
        split_datasets[split] = Subset(dataset=dataset, indices=indices)

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    qerror_fig, qerror_axs = plt.subplots(1, 3, figsize=(12, 6))

    split_true_y = {}
    split_predicted_y = {}

    for split_idx, split in enumerate(["train", "val", "test"]):
        split_dataset = split_datasets[split]
        print(f"Plotting for Split: {split}")

        true_y_d = {}
        for i in range(len(split_dataset)):
            _, _, y, query_id, _, _, _ = split_dataset[i]
            true_y_d[query_id] = (
                y.item() if not trained_on_log_runtime else np.expm1(y.item())
            )
        split_true_y[split] = pd.Series(true_y_d)

        split_predicted_y[split] = model.predict_from_dataset(
            dataset=split_dataset,
            use_stage_for_isolated_queries=use_stage_for_isolated_queries,
        )

        title = split.title() + " Set"

        plot_true_predicted(
            axs[split_idx], title, split_true_y[split], split_predicted_y[split]
        )
        plot_qerror(
            qerror_axs[split_idx],
            title,
            split_true_y[split],
            split_predicted_y[split],
        )

    # Save out the true and predicted series in text files
    for split in ["train", "val", "test"]:
        true_y_path = os.path.join(
            model_dir,
            f"iconq_model_{iconq_model_id}_{split}_true_y.csv",
        )
        predicted_y_path = os.path.join(
            model_dir,
            f"iconq_model_{iconq_model_id}_{split}_predicted_y.csv",
        )
        split_true_y[split].to_csv(true_y_path, header=["true_latency_s"])
        pd.Series(
            {
                query_id: pred.overall_mean_s()
                for query_id, pred in split_predicted_y[split].items()
            }
        ).to_csv(predicted_y_path, header=["predicted_latency_s"])
    print(f"Saved true and predicted latencies to {model_dir}")

    # Post-process true vs predicted figure
    suptitle = f"IconQ Model {iconq_model_id} Predictions vs True Latencies"
    if use_stage_for_isolated_queries:
        suptitle += " (Stage Fallback)"
    fig.suptitle(
        suptitle,
        fontsize=16,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(
        os.path.join(
            model_dir,
            f"iconq_model_{iconq_model_id}_pred_vs_true.svg",
        )
    )
    plt.close(fig)

    # Post-process Q-Error figure
    suptitle = f"IconQ Model {iconq_model_id} Q-Error CDFs"
    if use_stage_for_isolated_queries:
        suptitle += " (Stage Fallback)"
    qerror_fig.suptitle(
        suptitle,
        fontsize=16,
    )
    qerror_fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    qerror_fig.savefig(
        os.path.join(
            model_dir,
            f"iconq_model_{iconq_model_id}_qerror.svg",
        )
    )
    plt.close(qerror_fig)

    # Plot and post-process Q-Error barchart figure
    qerror_barchart_fig, qerror_barchart_ax = plt.subplots(
        1, 1, figsize=(10, 6)
    )
    title = "Q-Error Summary Across Splits"
    if use_stage_for_isolated_queries:
        title += " (Stage Fallback)"
    title += f" (Model {iconq_model_id})"
    plot_qerror_single_barchart(
        qerror_barchart_ax,
        title if not hide_plot_title else "",
        split_true_y,
        split_predicted_y,
        include_censored=None,
    )
    qerror_barchart_fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    qerror_barchart_fig.savefig(
        os.path.join(
            model_dir,
            f"iconq_model_{iconq_model_id}_qerror_barchart.svg",
        )
    )
    plt.close(qerror_barchart_fig)

    # Plot multi-panel Q-error barchart figure
    qerror_multi_barchart_fig, qerror_multi_barchart_axs = plt.subplots(
        1, 3, figsize=(27, 6), sharey=True
    )
    title = "Q-Error Summary Across Splits"
    if use_stage_for_isolated_queries:
        title += " (Stage Fallback)"
    title += f" (Model {iconq_model_id})"
    plot_qerror_single_barchart(
        qerror_multi_barchart_axs[0],
        title if not hide_plot_title else "",
        split_true_y,
        split_predicted_y,
        include_censored=None,
    )
    plot_qerror_single_barchart(
        qerror_multi_barchart_axs[1],
        title if not hide_plot_title else "",
        split_true_y,
        split_predicted_y,
        include_censored=False,
    )
    plot_qerror_single_barchart(
        qerror_multi_barchart_axs[2],
        title if not hide_plot_title else "",
        split_true_y,
        split_predicted_y,
        include_censored=True,
    )
    qerror_multi_barchart_fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    qerror_multi_barchart_fig.savefig(
        os.path.join(
            model_dir,
            f"iconq_model_{iconq_model_id}_qerror_multi_barchart.svg",
        )
    )
    plt.close(qerror_multi_barchart_fig)

    # Plot over/under estimation for censored points
    over_under_fig, over_under_ax = plt.subplots(1, 1, figsize=(10, 6))
    title = "Over/Under Estimation for Censored Points"
    if use_stage_for_isolated_queries:
        title += " (Stage Fallback)"
    title += f" (Model {iconq_model_id})"
    plot_over_under_for_censored(
        over_under_ax,
        title if not hide_plot_title else "",
        split_true_y,
        split_predicted_y,
    )
    over_under_fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    over_under_fig.savefig(
        os.path.join(
            model_dir,
            f"iconq_model_{iconq_model_id}_over_under_censored.svg",
        )
    )
    plt.close(over_under_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot true vs predicted query latencies for each split of an IconQ model."
        )
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        required=True,
        help="The IconQ model ID to use for predictions.",
    )
    parser.add_argument(
        "--hide_plot_title",
        action="store_true",
        help="Whether to hide the plot title.",
    )

    args = parser.parse_args()

    main(args.iconq_model_id, args.hide_plot_title)
