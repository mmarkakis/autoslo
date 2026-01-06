import argparse

import matplotlib.pyplot as plt
import pandas as pd

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.query_timeline import QueryTimeline
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_execution.trace import Trace

from tqdm.auto import tqdm

import os


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
        predicted_latency = predicted_y[query_id]
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


def main(iconq_model_id: str):
    model = IconqModel.load(
        model_id=iconq_model_id,
    )

    pct_heavy_options = [0, 10, 25, 50]
    mean_interarrival_options = [10, 30, 60, 120]
    rpus = [4, 8, 16, 32]

    for rpu in rpus:
        print(f"Plotting for RPU: {rpu}")

        fig, ax = plt.subplots(4, 4, figsize=(20, 20))

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
                query_timeline = QueryTimeline(
                    model._iconq_query_featurizer,
                    model._iconq_interaction_featurizer,
                )
                query_timeline.initialize_from_trace(
                    trace, stage_model=model.stage_model
                )

                predictions = model.predict_from_query_timeline(query_timeline)

                true_y = trace.latencies_s
                predicted_y = predictions
                title = f"{pct_heavy}% Heavy, {mean_interarrival}s Mean Interarrival, {rpu} RPU"
                plot_true_predicted(ax[i, j], title, true_y, predicted_y)
                bar.update(1)

        plt.suptitle(
            f"IconQ Model {iconq_model_id} Predictions vs True Latencies for {rpu} RPU",
            fontsize=16,
        )
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(
            os.path.join(
                pu.AUTOSLO_ROOT,
                "experiments",
                "06_iconq_training",
                f"iconq_model_{iconq_model_id}_rpu_{rpu}.png",
            )
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot true vs predicted query latencies for each chunk using the provided Iconq model."
        )
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        default="1767629626",
        help="The IconQ model ID to use for predictions.",
    )

    args = parser.parse_args()

    main(args.iconq_model_id)
