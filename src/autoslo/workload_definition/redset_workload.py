import os
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import autoslo.utils.colors as cu
import autoslo.utils.paths as pu
from autoslo.workload_definition.workload import Query, Workload
from autoslo.workload_execution.trace import Trace


class RedsetWorkload(Workload):
    """
    A specific Redset-inspired workload.
    """

    def __init__(
        self,
        cluster_type: str,
        cluster_id: int,
        seed: int,
    ):
        self.cluster_type = cluster_type
        self.cluster_id = cluster_id
        self.seed = seed

        self._queries: list[Query] = []

    @property
    def name(self) -> str:
        """Returns the name of the workload."""
        return f"redset_{self.cluster_type}_cluster{self.cluster_id}_seed{self.seed}"

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
        load_dir = os.path.join(
            pu.get_data_path(),
            "redset_workloads",
            workload_name,
        )
        in_path = os.path.join(load_dir, "parameters.yml")
        with open(in_path, "r") as f:
            params = yaml.safe_load(f)
        return RedsetWorkload(
            cluster_type=params["cluster_type"],
            cluster_id=params["cluster_id"],
            seed=params["seed"],
        )

    @property
    def queries(self) -> list[Query]:
        """
        Get the list of queries in the redset workload.

        Returns:
            A list of Query instances.
        """
        if len(self._queries) > 0:
            return self._queries

        workload_path = os.path.join(
            self.save_dir(),
            "days",
            f"2024-04-15.parquet",
        )
        if not os.path.exists(workload_path):
            raise FileNotFoundError(
                f"Workload file {workload_path} does not exist."
            )
        df = pd.read_parquet(workload_path)

        self._queries = [
            Query(
                query_id=row["query_id"],
                start_time_s=row["rel_start_time_s"],
                tpcds_temp_and_q_idx=(
                    f"{row['query_template']:03d}_"
                    f"{row['query_num_within_template']:03d}"
                ),
            )
            for _, row in df.iterrows()
        ]
        return self._queries
