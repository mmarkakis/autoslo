import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

import autoslo.utils.paths as pu
from autoslo.workload_definition.tpcds_sampler import TPCDSSampler
from autoslo.workload_definition.workload import Query, Workload


@dataclass
class RedsetWorkloadSamplingSpec:
    tpcds_prob_distribution_dir: str
    seed: int
    abs_start_time: Optional[datetime] = None
    abs_end_time: Optional[datetime] = None
    real_queries_per_output_queries: float = 1.0
    real_s_per_output_s: float = 1.0


class RedsetWorkload(Workload):
    """
    A specific Redset-inspired workload.
    """

    def __init__(
        self,
        cluster_type: str,
        cluster_id: int,
    ):
        self.cluster_type = cluster_type
        self.cluster_id = cluster_id

        columns = [
            "query_id",
            "arrival_timestamp",
            "queue_duration_ms",
            "execution_duration_ms",
        ]
        df = pd.read_parquet(
            pu.get_redset_raw_data(cluster_id=cluster_id), columns=columns
        )

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

        self._full_workload_df = df

    @property
    def name(self) -> str:
        """Returns the name of the workload."""
        return f"redset_{self.cluster_type}_cluster{self.cluster_id}"

    def save_dir(self) -> str:
        """Get the directory path where the redset workload is saved."""
        return os.path.join(
            pu.get_data_path(),
            "redset_workloads",
            self.name,
        )

    @staticmethod
    def load(workload_name: str) -> "RedsetWorkload":
        """Load the redset workload definition from a YAML file."""
        parts = workload_name.split("_")
        if len(parts) != 3 or parts[0] != "redset":
            raise ValueError(
                f"Invalid workload name {workload_name}. Expected format: "
                "redset_{cluster_type}_cluster{cluster_id}"
            )
        cluster_type = parts[1]
        cluster_id_str = parts[2]
        if not cluster_id_str.startswith("cluster"):
            raise ValueError(
                f"Invalid workload name {workload_name}. Expected format: "
                "redset_{cluster_type}_cluster{cluster_id}"
            )
        cluster_id = int(cluster_id_str[len("cluster") :])
        return RedsetWorkload(
            cluster_type=cluster_type,
            cluster_id=cluster_id,
        )

    def queries(self, sampling_spec: RedsetWorkloadSamplingSpec) -> list[Query]:
        """
        Returns a list of sampled queries in the workload, transformed according
        to the given spec.
        """

        df = self._full_workload_df

        if sampling_spec.abs_start_time is not None:
            df = df[df["arrival_timestamp"] >= sampling_spec.abs_start_time]
        if sampling_spec.abs_end_time is not None:
            df = df[df["arrival_timestamp"] <= sampling_spec.abs_end_time]

        if sampling_spec.real_queries_per_output_queries > 1.0:
            frac = 1 / sampling_spec.real_queries_per_output_queries
            df = df.sample(
                frac=frac,
                random_state=sampling_spec.seed,
            )
            df = df.sort_values(
                "arrival_timestamp", ascending=True
            ).reset_index(drop=True)

        if sampling_spec.real_s_per_output_s != 1.0:
            df["arrival_timestamp"] = pd.to_datetime(
                df["arrival_timestamp"]
            )  # ensure it's datetime
            min_time = df["arrival_timestamp"].min()
            df["arrival_timestamp"] = min_time + pd.to_timedelta(
                (df["arrival_timestamp"] - min_time).dt.total_seconds()
                / sampling_spec.real_s_per_output_s,
                unit="s",
            )

        # Cache TPCDSSampler by directory to avoid disk I/O on every sample.
        if not hasattr(self, "_sampler_cache"):
            self._sampler_cache = {}  # type: ignore
        if sampling_spec.tpcds_prob_distribution_dir not in self._sampler_cache:
            self._sampler_cache[sampling_spec.tpcds_prob_distribution_dir] = (
                TPCDSSampler.from_dir(
                    sampling_spec.tpcds_prob_distribution_dir
                )
            )
        sampler = self._sampler_cache[sampling_spec.tpcds_prob_distribution_dir]
        df["tpcds_temp_and_q_idx"] = sampler.sample(
            latencies_s=df["latency_s"], seed=sampling_spec.seed
        )
        df["rel_start_time_s"] = (
            df["arrival_timestamp"] - df["arrival_timestamp"].min()
        ).dt.total_seconds()

        # Use vectorized column extraction + zip instead of iterrows() for O(N) instead of O(N²) performance.
        query_ids = df["query_id"].tolist()
        tpcds_indices = df["tpcds_temp_and_q_idx"].tolist()
        abs_times = df["arrival_timestamp"].tolist()
        rel_times = df["rel_start_time_s"].tolist()

        queries = [
            Query(
                query_id=qid,
                tpcds_temp_and_q_idx=tpcds_idx,
                abs_start_time=abs_time,
                rel_start_time_s=rel_time,
            )
            for qid, tpcds_idx, abs_time, rel_time in zip(query_ids, tpcds_indices, abs_times, rel_times)
        ]
        return queries
