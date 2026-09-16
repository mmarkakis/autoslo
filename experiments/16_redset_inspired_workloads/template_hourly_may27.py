"""
Compare the per-hour distribution of TPC-DS query templates on a given date
across the three selected redset clusters (157, 135, 105).

For each cluster's workload.parquet:
- Filter to queries with abs_start_time on the target date.
- Extract the TPC-DS template id from query_text_id (format schema#template#idx).
- Build an hour-of-day x template count matrix.
- Compute, per hour: number of unique templates and normalized entropy of the
  template distribution (0 = single template dominates, 1 = uniform mix).

Outputs (suffixed with the target date, e.g. _2024-04-15):
- template_hourly_<date>.csv: long-form counts (cluster, hour, template, count)
- template_hourly_summary_<date>.csv: per-cluster/hour unique-template-count and entropy
- template_hourly_heatmap_<date>.png: 3-panel heatmap (hour x template) per cluster
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autoslo.workload_definition.query import QueryTextId

CLUSTERS = [157, 135, 105]
OUT_DIR = os.path.dirname(__file__)


def load_day(cluster_id: int, target_date: str) -> pd.DataFrame:
    path = f"/home/markakis/chunkbench/data/workloads/redbench_provisioned_{cluster_id}_0.parquet"
    df = pd.read_parquet(path, columns=["query_text_id", "abs_start_time"])
    df = df[df["abs_start_time"].dt.date == pd.Timestamp(target_date).date()].copy()
    df["hour"] = df["abs_start_time"].dt.hour
    df["template"] = df["query_text_id"].apply(lambda x: QueryTextId(x).template_id)
    df["cluster_id"] = cluster_id
    return df


def normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    probs = counts[counts > 0] / total
    entropy = -(probs * np.log(probs)).sum()
    n_possible = len(counts)
    if n_possible <= 1:
        return 0.0
    return entropy / np.log(n_possible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_date", default="2024-05-27", help="YYYY-MM-DD")
    target_date = parser.parse_args().target_date

    all_days = pd.concat([load_day(c, target_date) for c in CLUSTERS], ignore_index=True)

    long_counts = (
        all_days.groupby(["cluster_id", "hour", "template"]).size()
        .rename("count").reset_index()
    )
    long_counts.to_csv(os.path.join(OUT_DIR, f"template_hourly_{target_date}.csv"), index=False)

    summary_rows = []
    fig, axs = plt.subplots(1, len(CLUSTERS), figsize=(18, 6), sharey=True)
    for ax, cluster_id in zip(axs, CLUSTERS):
        cluster_df = all_days[all_days["cluster_id"] == cluster_id]
        pivot = cluster_df.pivot_table(
            index="template", columns="hour", values="query_text_id",
            aggfunc="count", fill_value=0,
        )
        # Reindex hours 0-23 for a consistent x-axis even if some hours are missing.
        pivot = pivot.reindex(columns=range(24), fill_value=0)

        for hour in range(24):
            counts = pivot[hour].to_numpy()
            n_unique = int((counts > 0).sum())
            entropy = normalized_entropy(counts)
            summary_rows.append({
                "cluster_id": cluster_id,
                "hour": hour,
                "total_queries": int(counts.sum()),
                "n_unique_templates": n_unique,
                "template_entropy": entropy,
            })

        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24), fontsize=7)
        ax.set_xlabel("Hour of Day")
        ax.set_title(f"Cluster {cluster_id}")
        fig.colorbar(im, ax=ax, label="Query Count")
    axs[0].set_ylabel("Template Rank (sorted)")
    plt.suptitle(f"TPC-DS Template Usage by Hour on {target_date}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"template_hourly_heatmap_{target_date}.png"), dpi=150)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, f"template_hourly_summary_{target_date}.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\nPer-cluster daily totals on", target_date)
    print(summary_df.groupby("cluster_id")["total_queries"].sum())
    print("\nPer-cluster mean unique templates/hour and mean entropy/hour (hours with >0 queries):")
    active = summary_df[summary_df["total_queries"] > 0]
    print(active.groupby("cluster_id")[["n_unique_templates", "template_entropy"]].mean())


if __name__ == "__main__":
    main()
