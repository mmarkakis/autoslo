"""
Compute quantitative seasonality / latency-mix / volume metrics for every
provisioned redset cluster, so we can shortlist additional clusters similar
to cluster 157 without eyeballing 200 report images.

Metrics per cluster (same underlying data as class_analysis.ipynb's check()):
- weekly_cv: coefficient of variation of query counts across ISO weeks
    (volume stability across the observed period)
- dow_cv: coefficient of variation of mean queries per day-of-week
    (weekly seasonality strength)
- hour_cv: coefficient of variation of queries per hour-of-day
    (daily seasonality strength)
- latency_entropy: normalized entropy of the query-latency-bucket
    distribution (0 = all queries in one bucket, 1 = uniform mix)
- median_weekly_queries: median number of queries per ISO week (volume)

Output: cluster_metrics.csv in this directory.
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import autoslo.filesystem.path_utils as pu

CLUSTER_TYPE = "provisioned"
NUM_CLUSTERS = 200
LATENCY_BUCKET_EDGES = [0, 1, 10, 60]
COLUMNS = [
    "cluster_size",
    "arrival_timestamp",
    "feature_fingerprint",
    "query_type",
    "queue_duration_ms",
    "execution_duration_ms",
]


def cv(series: pd.Series) -> float:
    """Coefficient of variation (std / mean); NaN-safe for degenerate series."""
    mean = series.mean()
    if mean == 0 or pd.isna(mean):
        return float("nan")
    return series.std() / mean


def bucketize(latencies_s: np.ndarray) -> np.ndarray:
    bucket_idx = np.searchsorted(LATENCY_BUCKET_EDGES, latencies_s, side="right") - 1
    counts = np.bincount(bucket_idx, minlength=len(LATENCY_BUCKET_EDGES))
    return counts


def normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    probs = counts[counts > 0] / total
    entropy = -(probs * np.log(probs)).sum()
    max_entropy = np.log(len(counts))
    return entropy / max_entropy


def compute_metrics(cluster_id: int) -> dict | None:
    path = pu.get_redset_raw_data(cluster_type=CLUSTER_TYPE, cluster_id=cluster_id)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=COLUMNS)
    if df.empty:
        return None

    df["day"] = df["arrival_timestamp"].dt.date
    df["day_of_week"] = df["arrival_timestamp"].dt.dayofweek
    df["hour_of_day"] = df["arrival_timestamp"].dt.hour
    df["week"] = df["arrival_timestamp"].dt.isocalendar().week

    weekly_counts = df.groupby("week").size()

    daily_grouped = (
        df.groupby("day")
        .agg(num_queries=("arrival_timestamp", "size"), day_of_week=("day_of_week", "first"))
    )
    mean_by_dow = daily_grouped.groupby("day_of_week")["num_queries"].mean()

    hourly_counts = df.groupby("hour_of_day").size()

    latency_s = ((df["execution_duration_ms"] + df["queue_duration_ms"]) / 1000).to_numpy()
    latency_bucket_counts = bucketize(latency_s)

    return {
        "cluster_id": cluster_id,
        "total_queries": len(df),
        "num_weeks": weekly_counts.shape[0],
        "median_weekly_queries": weekly_counts.median(),
        "weekly_cv": cv(weekly_counts),
        "dow_cv": cv(mean_by_dow),
        "hour_cv": cv(hourly_counts),
        "latency_entropy": normalized_entropy(latency_bucket_counts),
    }


def main() -> None:
    rows = []
    for cluster_id in tqdm(range(NUM_CLUSTERS)):
        try:
            metrics = compute_metrics(cluster_id)
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")
            continue
        if metrics is not None:
            rows.append(metrics)

    df = pd.DataFrame(rows).set_index("cluster_id").sort_index()
    out_path = os.path.join(os.path.dirname(__file__), "cluster_metrics.csv")
    df.to_csv(out_path)
    print(f"Wrote {len(df)} rows to {out_path}")

    if 157 in df.index:
        print("\nCluster 157 (our current reference pick):")
        print(df.loc[157])
    else:
        print("\nWARNING: cluster 157 not found in results")


if __name__ == "__main__":
    main()
