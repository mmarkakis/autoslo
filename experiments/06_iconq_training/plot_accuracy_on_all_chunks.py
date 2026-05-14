import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

import autoslo.filesystem.path_utils as pu
from autoslo.models.iconq_dataset_builder import build_dataset_from_trace
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_execution.trace import Trace


def plot_true_predicted(
    ax: plt.Axes,
    title: str,
    true_y: pd.Series,
    predicted_y: dict[str, ModelPrediction],
):
    color_if_alone = "green"
    color_if_overlapped = "blue"

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
    ax.plot(cdf.index, cdf.values)  # type: ignore

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


def main(iconq_model_id: str, use_stage_for_isolated_queries: bool):
    model = IconqModel.load(
        model_id=iconq_model_id,
    )

    pct_heavy_options = [0, 10, 25, 50]
    mean_interarrival_options = [10, 30, 60, 120]
    rpus = [4, 8, 16, 32]

    for rpu in rpus:
        print(f"Plotting for RPU: {rpu}")

        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        qerror_fig, qerror_axs = plt.subplots(4, 4, figsize=(20, 20))

        bar = tqdm(range(16))

        for i, pct_heavy in enumerate(pct_heavy_options):
            for j, mean_interarrival in enumerate(mean_interarrival_options):
                trace_run_id = max(
                    pu.RunLocator.get_run_ids(
                        schema_name="ext_tpcds1000",
                        workload_name="{}pctheavy_{}meaninterarrival".format(
                            pct_heavy, mean_interarrival
                        ),
                        blueprint_name=f"single_{rpu}",
                    )
                )

                trace = Trace(trace_run_id)
                dataset = build_dataset_from_trace(
                    trace=trace,
                    iconq_model=model,
                    run_id=trace_run_id,
                )
                raw = model.predict_from_dataset(dataset)
                predictions = {
                    caqid.query_id: pred.overall_mean_s()
                    for caqid, pred in raw.items()
                }

                true_y = trace.latencies_s
                predicted_y = predictions
                title = f"{pct_heavy}% Heavy, {mean_interarrival}s Mean Interarrival, {rpu} RPU"

                plot_true_predicted(axs[i, j], title, true_y, predicted_y)
                plot_qerror(qerror_axs[i, j], title, true_y, predicted_y)
                bar.update(1)
        bar.close()

        # Post-process true vs predicted figure
        suptitle = f"IconQ Model {iconq_model_id} Predictions vs True Latencies for {rpu} RPU"
        if use_stage_for_isolated_queries:
            suptitle += " (Stage Fallback)"
        fig.suptitle(
            suptitle,
            fontsize=16,
        )

        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(
            os.path.join(
                pu.AUTOSLO_ROOT,
                "experiments",
                "06_iconq_training",
                f"iconq_model_{iconq_model_id}_rpu_{rpu}.png",
            )
        )
        plt.close(fig)

        # Post-process Q-Error figure
        suptitle = f"IconQ Model {iconq_model_id} Q-Error CDFs for {rpu} RPU"
        if use_stage_for_isolated_queries:
            suptitle += " (Stage Fallback)"
        qerror_fig.suptitle(
            suptitle,
            fontsize=16,
        )
        qerror_fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        qerror_fig.savefig(
            os.path.join(
                pu.AUTOSLO_ROOT,
                "experiments",
                "06_iconq_training",
                f"iconq_model_{iconq_model_id}_rpu_{rpu}_qerror.png",
            )
        )
        plt.close(qerror_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot true vs predicted query latencies for each chunk using the provided Iconq model."
        )
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        required=True,
        help="The IconQ model ID to use for predictions.",
    )
    parser.add_argument(
        "--use_stage_for_isolated_queries",
        action="store_true",
        help="Whether to use the StageModel for isolated queries.",
    )

    args = parser.parse_args()

    main(args.iconq_model_id, args.use_stage_for_isolated_queries)
