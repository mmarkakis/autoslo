import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.billing import Billing
from autoslo.visualizations.colors import Palette

DATASETS = [
    {
        "name": "redset",
        "span": "Three Months",
        "base_color": Palette.light_gray,
        "extra_1s_color": Palette.dark_orange,
        "extra_60s_color": Palette.dark_red,
        "anno_color": Palette.dark_green,
        "fontsize": 14,
    },
]


def preprocess_redset():

    billed_seconds_per_cluster = []

    for cluster_id in tqdm(range(200)):
        df = pd.read_parquet(
            pu.get_redset_raw_data(
                cluster_type="serverless", cluster_id=cluster_id
            ),
            columns=[
                "instance_id",
                "query_id",
                "arrival_timestamp",
                "queue_duration_ms",
                "execution_duration_ms",
            ],
        )

        # Produce standardized column names.
        df.rename(columns={"instance_id": "cluster_id"}, inplace=True)
        df["start"] = df["arrival_timestamp"]
        df["end"] = (
            df["start"]
            + pd.to_timedelta(df["queue_duration_ms"], unit="ms")
            + pd.to_timedelta(df["execution_duration_ms"], unit="ms")
        )

        billed_s_0s = Billing.billed_s_from_df(
            df, threshold_s=0, granularity_s=0
        )
        billed_s_1s = Billing.billed_s_from_df(
            df, threshold_s=1, granularity_s=1
        )
        billed_s_60s = Billing.billed_s_from_df(
            df, threshold_s=60, granularity_s=1
        )
        billed_seconds_per_cluster.append(
            {
                "cluster_id": cluster_id,
                "billed_s_0s": billed_s_0s,
                "billed_s_1s": billed_s_1s,
                "billed_s_60s": billed_s_60s,
            }
        )

    billed_seconds_df = pd.DataFrame(billed_seconds_per_cluster)
    billed_seconds_df["blowup_0s_1s"] = (
        billed_seconds_df["billed_s_1s"] / billed_seconds_df["billed_s_0s"]
    )
    billed_seconds_df["blowup_0s_60s"] = (
        billed_seconds_df["billed_s_60s"] / billed_seconds_df["billed_s_0s"]
    )
    return billed_seconds_df


def plot(info):

    if info["name"] == "redset":
        path = os.path.join(
            pu.AUTOSLO_ROOT,
            "experiments",
            "22_redset_billing",
            "redset_billed_seconds.parquet",
        )
        if os.path.exists(path):
            billed_seconds_df = pd.read_parquet(path)

        else:
            billed_seconds_df = preprocess_redset()
            billed_seconds_df.to_parquet(path)
    else:
        raise ValueError(f"Unknown dataset name: {info['name']}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    num_workers = len(billed_seconds_df)

    ### Left subplot
    bottom = np.zeros(num_workers)
    alignment = "edge"
    ax[0].barh(
        billed_seconds_df.index,
        (billed_seconds_df["billed_s_0s"]),
        color=info["base_color"],
        left=bottom,
        align=alignment,
    )
    bottom = billed_seconds_df["billed_s_0s"]
    ax[0].barh(
        billed_seconds_df.index,
        billed_seconds_df["billed_s_1s"] - bottom,
        color=info["extra_1s_color"],
        left=bottom,
        align=alignment,
    )
    bottom = billed_seconds_df["billed_s_1s"]
    ax[0].barh(
        billed_seconds_df.index,
        billed_seconds_df["billed_s_60s"] - bottom,
        color=info["extra_60s_color"],
        left=bottom,
        align=alignment,
    )

    # Axes
    xmin = 1e-1
    xmax = 10 ** np.ceil(np.log10(billed_seconds_df["billed_s_60s"].max()))
    ymin = -num_workers / 40
    ymax = num_workers * 1.05

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(
        f"Cumulative Query Execution Time Over {info['span']} (s)",
        fontsize=info["fontsize"],
    )
    ax[0].tick_params(labelsize=info["fontsize"])
    ax[0].set_ylim(ymin, ymax)
    ax[0].set_yticks([num_workers * i / 4 for i in range(5)])
    ax[0].set_yticklabels(
        ["0", "25", "50", "75", "100"], fontsize=info["fontsize"]
    )
    ax[0].set_ylabel(
        "Worker Percentile by Billed Execution Time", fontsize=info["fontsize"]
    )

    # Legend
    ax[0].legend(
        [
            "As Measured",
            "Extra With 1s Minimum",
            "Extra With 60s Minimum",
        ],
        loc="lower right",
        framealpha=1,
        fontsize=info["fontsize"],
    )

    # Add a horizontal line for 1 minute, 1 hour, 1 day, 30 days, 1000 days
    text_y = num_workers * 1.02
    ax[0].axvline(60, color=info["anno_color"], linestyle="--", linewidth=1)
    ax[0].text(
        60 * 0.9,
        text_y,
        "1 minute",
        color=info["anno_color"],
        ha="right",
        va="center",
        fontsize=info["fontsize"],
    )
    ax[0].axvline(3600, color=info["anno_color"], linestyle="--", linewidth=1)
    ax[0].text(
        3600 * 0.9,
        text_y,
        "1 hour",
        color=info["anno_color"],
        ha="right",
        va="center",
        fontsize=info["fontsize"],
    )
    ax[0].axvline(86400, color=info["anno_color"], linestyle="--", linewidth=1)
    ax[0].text(
        86400 * 0.9,
        text_y,
        "1 day",
        color=info["anno_color"],
        ha="right",
        va="center",
        fontsize=info["fontsize"],
    )
    ax[0].axvline(
        86400 * 30, color=info["anno_color"], linestyle="--", linewidth=1
    )
    ax[0].text(
        86400 * 30 * 0.9,
        text_y,
        "1 month",
        color=info["anno_color"],
        ha="right",
        va="center",
        fontsize=info["fontsize"],
    )

    ### Right subplot
    bottom = np.ones(len(billed_seconds_df))
    alignment = "edge"
    ax[1].barh(
        billed_seconds_df.index,
        billed_seconds_df["blowup_0s_1s"] - bottom,
        color=info["extra_1s_color"],
        left=bottom,
        align=alignment,
    )
    bottom = billed_seconds_df["blowup_0s_1s"]
    ax[1].barh(
        billed_seconds_df.index,
        billed_seconds_df["blowup_0s_60s"] - bottom,
        color=info["extra_60s_color"],
        left=bottom,
        align=alignment,
    )

    # Axes
    xmin = 1
    xmax = 3000
    ymin = -num_workers / 40
    ymax = num_workers * 1.05

    ax[1].set_xlim(xmin, xmax)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(
        "Billed Execution Time Increase Factor Because of Billing Minima",
        fontsize=info["fontsize"],
    )
    ax[1].tick_params(labelsize=info["fontsize"])
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_yticks([num_workers * i / 4 for i in range(5)])
    ax[1].set_yticklabels(
        ["0", "25", "50", "75", "100"], fontsize=info["fontsize"]
    )

    # Annotation
    quartile_idx = int(0.75 * num_workers)
    ax[1].axhline(
        quartile_idx, color=info["anno_color"], linestyle="--", linewidth=1
    )
    topquartile = billed_seconds_df[
        billed_seconds_df.index >= int(0.75 * num_workers)
    ]
    ax[1].text(
        120,
        quartile_idx * 16 / 15,
        "Top Quartile Median Increase Factor\n"
        r"$\bullet$"
        f" With 1s Minimum: {topquartile['blowup_0s_1s'].median():.2f}"
        r"$\times$"
        "\n"
        r"$\bullet$"
        f" With 60s Minimum: {topquartile['blowup_0s_60s'].median():.2f}"
        r"$\times$",
        color=info["anno_color"],
        ha="left",
        va="bottom",
        fontsize=info["fontsize"],
    )
    ax[1].text(
        120,
        quartile_idx / 3,
        "Overall Median Increase Factor\n"
        r"$\bullet$"
        f" With 1s Minimum: {billed_seconds_df['blowup_0s_1s'].median():.2f}"
        r"$\times$"
        "\n"
        r"$\bullet$"
        f" With 60s Minimum: {billed_seconds_df['blowup_0s_60s'].median():.2f}"
        r"$\times$",
        color=info["anno_color"],
        ha="left",
        va="bottom",
        fontsize=info["fontsize"],
    )

    plt.tight_layout()
    figpath = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "22_redset_billing",
        f"{info['name']}_duration_increase.png",
    )
    plt.savefig(figpath, dpi=300)
    plt.clf()


def plot2(info):

    if info["name"] == "redset":
        path = os.path.join(
            pu.AUTOSLO_ROOT,
            "experiments",
            "22_redset_billing",
            "redset_billed_seconds.parquet",
        )
        if os.path.exists(path):
            billed_seconds_df = pd.read_parquet(path)

        else:
            billed_seconds_df = preprocess_redset()
            billed_seconds_df.to_parquet(path)
    else:
        raise ValueError(f"Unknown dataset name: {info['name']}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    num_workers = len(billed_seconds_df)
    print(f"Number of clusters: {num_workers}")

    # Plot the cdfs of the blowup factors for 1s and 60s minima.
    sorted_blowup_1s = np.sort(billed_seconds_df["blowup_0s_1s"])
    sorted_blowup_60s = np.sort(billed_seconds_df["blowup_0s_60s"])
    cdf = np.arange(1, num_workers + 1) / num_workers

    # Plot a vertical line at x=1 (no increase) and annotate it.
    ax.plot(
        sorted_blowup_1s,
        cdf,
        color=info["extra_1s_color"],
        label="1s minimum",
    )
    ax.plot(
        sorted_blowup_60s,
        cdf,
        color=info["extra_60s_color"],
        label="60s minimum",
    )

    ax.set_xlabel(
        "Billed Time / Active Time",
        fontsize=info["fontsize"],
    )
    ax.tick_params(labelsize=info["fontsize"])
    ax.set_ylabel(
        "Redset Serverless Cluster Fraction", fontsize=info["fontsize"]
    )
    ax.legend(fontsize=info["fontsize"], loc="lower right")
    ax.set_xscale("log")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    figpath = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "22_redset_billing",
        f"{info['name']}_duration_increase_cdf.png",
    )
    plt.savefig(figpath, dpi=300)
    plt.clf()


if __name__ == "__main__":
    for info in DATASETS:
        plot(info)
        plot2(info)
