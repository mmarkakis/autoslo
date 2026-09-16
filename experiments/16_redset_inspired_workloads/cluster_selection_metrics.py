"""Compute quantitative seasonality/latency-mix/volume metrics for every
redset provisioned cluster, then rank clusters by similarity to cluster 157
(our existing pick) to shortlist additional candidates.

Reuses the same per-cluster fields as class_analysis.ipynb's check(), but
emits numeric scores instead of plots so the full sweep can be filtered and
sorted programmatically. Run with `python cluster_selection_metrics.py`.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm.auto import tqdm

import autoslo.filesystem.path_utils as pu

REFERENCE_CLUSTER_ID = 157
LATENCY_BUCKET_EDGES = [0, 1, 10, 60]
N_TOP_CANDIDATES = 15
OUTPUT_CSV = "cluster_metrics.csv"


def compute_cluster_metrics(cluster_id: int) -> dict | None:
    """Compute seasonality/latency-mix/volume metrics for one cluster."""
    columns = [
        "arrival_timestamp",
        "queue_duration_ms",
        "execution_duration_ms",
    ]
    try:
        df = pd.read_parquet(
            pu.get_redset_raw_data(cluster_id=cluster_id), columns=columns
        )
    except IndexError:
        return None
    if df.empty:
        return None

    df["day"] = df["arrival_timestamp"].dt.date
    df["day_of_week"] = df["arrival_timestamp"].dt.dayofweek
    df["hour_of_day"] = df["arrival_timestamp"].dt.hour
    df["week"] = df["arrival_timestamp"].dt.isocalendar().week

    # Volume: mean number of queries per week.
    weekly_counts = df.groupby("week").size()
    mean_queries_per_week = weekly_counts.mean()

    # Weekly seasonality: coefficient of variation of mean daily volume
    # across day-of-week (0 = perfectly uniform, higher = more seasonal).
    daily_counts = df.groupby("day").agg(
        num_queries=("day_of_week", "size"),
        day_of_week=("day_of_week", "first"),
    )
    mean_by_dow = daily_counts.groupby("day_of_week")["num_queries"].mean()
    weekly_seasonality = mean_by_dow.std() / mean_by_dow.mean()

    # Daily seasonality: coefficient of variation of query volume across
    # hour-of-day (0 = uniform, higher = more seasonal).
    hourly_counts = df.groupby("hour_of_day").size()
    daily_seasonality = hourly_counts.std() / hourly_counts.mean()

    # Latency mix: entropy of the fraction of queries in each latency
    # bucket (0 = all queries in one bucket, higher = more evenly mixed).
    latency_s = (
        df["execution_duration_ms"] + df["queue_duration_ms"]
    ) / 1000
    bucket_idx = np.searchsorted(
        LATENCY_BUCKET_EDGES, latency_s, side="right"
    ) - 1
    bucket_fracs = (
        pd.Series(bucket_idx)
        .value_counts(normalize=True)
        .reindex(range(len(LATENCY_BUCKET_EDGES)), fill_value=0)
    )
    latency_mix_entropy = entropy(bucket_fracs, base=2)

    return {
        "cluster_id": cluster_id,
        "mean_queries_per_week": mean_queries_per_week,
        "weekly_seasonality": weekly_seasonality,
        "daily_seasonality": daily_seasonality,
        "latency_mix_entropy": latency_mix_entropy,
    }


def main() -> None:
    rows = []
    for cluster_id in tqdm(range(200)):
        metrics = compute_cluster_metrics(cluster_id)
        if metrics is not None:
            rows.append(metrics)
    df = pd.DataFrame(rows).set_index("cluster_id")

    # Z-score each metric so they're comparable, using the reference
    # cluster's values as the target profile to match against.
    metric_cols = [
        "mean_queries_per_week",
        "weekly_seasonality",
        "daily_seasonality",
        "latency_mix_entropy",
    ]
    z = (df[metric_cols] - df[metric_cols].mean()) / df[metric_cols].std()
    reference = z.loc[REFERENCE_CLUSTER_ID]
    df["distance_to_157"] = np.sqrt(
        ((z - reference) ** 2).sum(axis=1)
    )

    # Exclude extreme-volume clusters (outside the 10th-90th percentile of
    # weekly query counts), matching the "non-extreme volume" criterion.
    low, high = df["mean_queries_per_week"].quantile([0.1, 0.9])
    df["non_extreme_volume"] = df["mean_queries_per_week"].between(
        low, high
    )

    df = df.sort_values("distance_to_157")
    df.to_csv(OUTPUT_CSV)

    print(f"Reference cluster {REFERENCE_CLUSTER_ID}:")
    print(df.loc[[REFERENCE_CLUSTER_ID]].to_string())
    print(f"\nTop {N_TOP_CANDIDATES} candidates closest to cluster "
          f"{REFERENCE_CLUSTER_ID} (excluding it), non-extreme volume only:")
    candidates = df[
        (df.index != REFERENCE_CLUSTER_ID) & df["non_extreme_volume"]
    ].head(N_TOP_CANDIDATES)
    print(candidates.to_string())
    print(f"\nFull metrics written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
