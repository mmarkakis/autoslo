"""
Compute a "peak burst concurrency" screening metric for all 200 provisioned
redset clusters on 2024-05-27.

Output: burst_metrics_2024-05-27.csv in this directory, with one row per
cluster (id, day_total_queries, peak_5min_bin_count, peak_compressed_qps,
concurrency_estimate).
"""
import os

import pandas as pd
import yaml
from tqdm import tqdm

import autoslo.filesystem.path_utils as pu

NUM_CLUSTERS = 200
TARGET_DATE = "2024-05-27"
RESCALE_FACTOR = 1.0 / 6.0
BIN_WIDTH_S = 300.0  # 5-minute real-time bins
OUT_DIR = os.path.dirname(__file__)

# Mean isolated p50 latency across all 99 TPC-DS templates (cluster-agnostic
# baseline from data/slos/1778469201294_p50_k1.yml), used as an approximate
# average per-query service time for the Little's-Law concurrency estimate.
with open(pu.get_data_dir() / "slos" / "1778469201294_p50_k1.yml") as f:
    _slo_table = yaml.safe_load(f)
MEAN_SERVICE_TIME_S = sum(_slo_table["slo_dict"].values()) / len(
    _slo_table["slo_dict"]
)


def compute_burst_metrics(cluster_id: int) -> dict | None:
    path = pu.get_redset_raw_data(cluster_type="provisioned", cluster_id=cluster_id)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=["arrival_timestamp"])
    day = df[df["arrival_timestamp"].dt.date == pd.Timestamp(TARGET_DATE).date()]
    if day.empty:
        return {
            "cluster_id": cluster_id,
            "day_total_queries": 0,
            "peak_5min_bin_count": 0,
            "peak_compressed_qps": 0.0,
            "concurrency_estimate": 0.0,
        }

    bin_counts = day.groupby(day["arrival_timestamp"].dt.floor("5min")).size()
    peak_bin_count = int(bin_counts.max())
    # Compressed-time equivalent: BIN_WIDTH_S of real time becomes
    # BIN_WIDTH_S * RESCALE_FACTOR of live/simulated time.
    peak_compressed_qps = peak_bin_count / (BIN_WIDTH_S * RESCALE_FACTOR)
    concurrency_estimate = peak_compressed_qps * MEAN_SERVICE_TIME_S

    return {
        "cluster_id": cluster_id,
        "day_total_queries": len(day),
        "peak_5min_bin_count": peak_bin_count,
        "peak_compressed_qps": peak_compressed_qps,
        "concurrency_estimate": concurrency_estimate,
    }


def main() -> None:
    rows = []
    for cluster_id in tqdm(range(NUM_CLUSTERS)):
        try:
            metrics = compute_burst_metrics(cluster_id)
        except Exception as e:
            print(f"Error processing cluster {cluster_id}: {e}")
            continue
        if metrics is not None:
            rows.append(metrics)

    df = pd.DataFrame(rows).set_index("cluster_id").sort_index()
    out_path = os.path.join(OUT_DIR, "burst_metrics_2024-05-27.csv")
    df.to_csv(out_path)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"\nMean isolated service time used: {MEAN_SERVICE_TIME_S:.2f}s")

    for ref_id in (157, 135, 105):
        if ref_id in df.index:
            print(f"\nCluster {ref_id}:")
            print(df.loc[ref_id])


if __name__ == "__main__":
    main()
