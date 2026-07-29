import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import autoslo.filesystem.path_utils as pu


def _template_id(temp_and_q_idx: str) -> int:
    """Extract template ID from 'template_querynum' string."""
    return int(temp_and_q_idx.split("_")[0])


def _idx_in_template(temp_and_q_idx: str) -> int:
    """Extract query index from 'template_querynum' string."""
    return int(temp_and_q_idx.split("_")[1])


def read_in_query_texts(
    temp_and_q_idxs: list[str],
) -> dict[str, str]:

    query_texts_dict: dict[str, str] = {}

    for temp_and_q_idx in temp_and_q_idxs:
        template_id = _template_id(temp_and_q_idx)
        query_num = _idx_in_template(temp_and_q_idx)
        template_str = f"query{template_id:03d}"
        query_path = (
            pu.QUERIES_PATH
            / template_str
            / f"{template_str}_{query_num:03d}.sql"
        )
        if not query_path.exists():
            raise FileNotFoundError(f"Query file {query_path} does not exist.")
        with open(query_path, "r") as f:
            query_texts_dict[temp_and_q_idx] = f.read()
    return query_texts_dict


def main(args) -> None:
    # Create the output directory and dump the parameters.
    output_dir = (
        pu.get_data_path()
        / "redset_workloads"
        / f"redset_{args.cluster_type}_cluster{args.cluster_id}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "parameters.yml", "w") as f:
        yaml.dump(vars(args), f)

    # Read in and process the Redset trace.
    columns = [
        "query_id",
        "arrival_timestamp",
        "queue_duration_ms",
        "execution_duration_ms",
    ]
    df = pd.read_parquet(
        pu.get_redset_raw_data(cluster_id=args.cluster_id), columns=columns
    )
    N = len(df)
    df["arrival_hour"] = df["arrival_timestamp"].dt.floor("h")
    df["arrival_day"] = df["arrival_timestamp"].dt.floor("D")

    latency_bin_left_edges_s = [0, 1, 10, 60]
    df["latency_s"] = (
        df["execution_duration_ms"] + df["queue_duration_ms"]
    ) / 1000
    df["latency_bin_left_edge_s"] = pd.cut(
        df["latency_s"],
        bins=latency_bin_left_edges_s + [float("inf")],
        labels=latency_bin_left_edges_s,
        right=False,
    ).astype(float)
    df["latency_bin_idx"] = df["latency_bin_left_edge_s"].apply(
        lambda x: latency_bin_left_edges_s.index(x)
    )

    # Read in the TPC-DS probability distribution and mapping.
    tpcds_prob_distribution_dir = Path(args.tpcds_prob_distribution_dir)
    dist_path = tpcds_prob_distribution_dir / "array.npy"
    with open(dist_path, "rb") as f:
        dist = np.load(f)
    index_dict_path = tpcds_prob_distribution_dir / "index_dict.yml"
    with open(index_dict_path, "r") as f:
        index_dict = yaml.safe_load(f)
    column_dict_path = tpcds_prob_distribution_dir / "column_dict.yml"
    with open(column_dict_path, "r") as f:
        column_dict = yaml.safe_load(f)

    assert isinstance(dist, np.ndarray)
    assert dist.shape[0] == len(latency_bin_left_edges_s)
    assert dist.shape[0] == len(index_dict)
    for bin_idx, bin_left_edge in index_dict.items():
        assert latency_bin_left_edges_s[int(bin_idx)] == bin_left_edge
    assert dist.shape[1] == len(column_dict)

    # Precompute CDFs
    dist /= dist.sum(axis=1, keepdims=True)
    C = np.cumsum(dist, axis=1)
    C[:, -1] = 1.0  # guard against tiny floating error

    # Sample.
    rng = np.random.default_rng(seed=args.seed)
    u = rng.random(N, dtype=np.float32)
    A_codes = np.empty(N, dtype=np.int16)  # 297 fits in int16
    for b in range(4):
        idx = np.flatnonzero(df["latency_bin_idx"] == b)
        A_codes[idx] = np.searchsorted(C[b], u[idx], side="right")

    df["temp_option_idx"] = A_codes

    # Set columns.
    df["workload_id"] = (
        f"redset_{args.cluster_type}_cluster{args.cluster_id}_seed{args.seed}"
    )
    df["rel_start_time_s"] = (
        df["arrival_timestamp"] - df["arrival_timestamp"].min()
    ).dt.total_seconds()
    df["tpcds_temp_and_q_idx"] = df["temp_option_idx"].map(column_dict.get)
    df["query_template"] = df["tpcds_temp_and_q_idx"].apply(_template_id)
    df["query_num_within_template"] = df["tpcds_temp_and_q_idx"].apply(
        _idx_in_template
    )
    query_texts: dict[str, str] = read_in_query_texts(
        list(df["tpcds_temp_and_q_idx"].unique())
    )
    df["query_text"] = df["tpcds_temp_and_q_idx"].map(query_texts)

    # Write out the full workload.
    full_out_path = output_dir / "full_workload.parquet"
    df.to_parquet(full_out_path, index=False)

    # Write out per day workloads.
    day_dir = output_dir / "days"
    day_dir.mkdir(parents=True, exist_ok=True)
    for day, day_df in df.groupby("arrival_day"):
        day_str = day.strftime("%Y-%m-%d")
        day_out_path = day_dir / f"{day_str}.parquet"
        day_df["rel_start_time_s"] = (
            day_df["arrival_timestamp"] - day_df["arrival_timestamp"].min()
        ).dt.total_seconds()
        day_df.to_parquet(day_out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Redset-inspired workload"
    )
    parser.add_argument(
        "--cluster_type",
        type=str,
        choices=["provisioned", "serverless"],
        default="provisioned",
        help="Type of cluster to generate workload for.",
    )
    parser.add_argument(
        "--cluster_id",
        type=int,
        help="ID of the cluster to generate workload for (e.g., 12, 13, 14, 15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation.",
    )
    parser.add_argument(
        "--tpcds_prob_distribution_dir",
        type=str,
        default=str(
            pu.get_data_path() / "generation_parameters" / "dist_16_rpu"
        ),
        help="Path to directory containing TPC-DS probability distributions.",
    )

    args = parser.parse_args()

    main(args)
