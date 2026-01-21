from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_execution.trace import Trace


@dataclass
class QueryInfo:
    """
    Lightweight container for query information needed for dataset construction.

    Attributes:
        query_id: The unique identifier for the query.
        tpcds_temp_and_q_idx: The TPC-DS template and query index.
        start_time_s: The start time of the query (in seconds, relative to some reference).
        cluster_name: The name of the cluster where the query executes.
        query_featurization: Optional pre-computed query features.
        stage_latency_prediction: Optional stage model latency prediction for this query.
        latency_s: The actual or estimated latency of the query (in seconds).
    """

    query_id: str
    tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx
    start_time_s: float
    cluster_name: str
    query_featurization: IconqQueryFeaturizer.IconqQueryFeaturization
    stage_latency_prediction: float
    latency_s: Optional[float] = None



class ConcurrentQueryDataset(Dataset):
    """
    A PyTorch Dataset for concurrent query data.

    Each item in the dataset is a tuple of the form
    (x, pinch_point, y, query_ids), where:
        - x is a list of the the input tensor, of shape (seq_len, input_size).
            The list length is batch_size.
        - pinch_point is the pinch points tensor, of shape (1,).
        - y is the target tensor, of shape (1,).
        - query_ids is the list of query IDs.
        - tpcds_temp_and_q_idx is the TPC-DS template and query index.
    """

    def __init__(
        self,
        x: list[torch.Tensor],
        pinch_points: torch.Tensor,
        y: torch.Tensor,
        query_ids: list[str],
        tpcds_temp_and_q_idx: list[Trace.TPCDSTempAndQIdx],
        run_ids: list[str],
    ):
        self.x = x
        self.pinch_points = pinch_points
        self.y = y
        self.query_ids = query_ids
        self.tpcds_temp_and_q_idx = tpcds_temp_and_q_idx
        self.run_ids = run_ids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        Trace.TPCDSTempAndQIdx,
        str,
    ]:
        return (
            self.x[idx],
            self.pinch_points[idx],
            self.y[idx],
            self.query_ids[idx],
            self.tpcds_temp_and_q_idx[idx],
            self.run_ids[idx],
        )

    @staticmethod
    def concatenate(
        datasets: list["ConcurrentQueryDataset"],
    ) -> "ConcurrentQueryDataset":
        if not all(isinstance(d, ConcurrentQueryDataset) for d in datasets):
            raise ValueError(
                "Can only add ConcurrentQueryDataset to another "
                "ConcurrentQueryDataset"
            )

        new_x = []
        for dataset in datasets:
            new_x.extend(dataset.x)
        new_pinch_points = torch.cat(
            [dataset.pinch_points for dataset in datasets], dim=0
        )
        new_y = torch.cat([dataset.y for dataset in datasets], dim=0)
        new_query_ids = []
        for dataset in datasets:
            new_query_ids.extend(dataset.query_ids)
        new_tpcds_temp_and_q_idx = []
        for dataset in datasets:
            new_tpcds_temp_and_q_idx.extend(dataset.tpcds_temp_and_q_idx)
        new_run_ids = []
        for dataset in datasets:
            new_run_ids.extend(dataset.run_ids)

        return ConcurrentQueryDataset(
            x=new_x,
            pinch_points=new_pinch_points,
            y=new_y,
            query_ids=new_query_ids,
            tpcds_temp_and_q_idx=new_tpcds_temp_and_q_idx,
            run_ids=new_run_ids,
        )

    @staticmethod
    def collate_and_pad(
        batch: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                str,
                Trace.TPCDSTempAndQIdx,
                str,
            ]
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[str],
        list[Trace.TPCDSTempAndQIdx],
        list[str],
    ]:
        """
        Custom collation function for DataLoader of ConcurrentQueryDataset.
        It pads the sequences in the batch to the maximum length per batch.

        Parameters:
            batch: The batch of data to collate. Each element in the batch is a
                tuple produced by ConcurrentQueryDataset, so its elements are
                (x, pinch_point, y, query_id, tpcds_temp_and_q_idx).

        Returns:
            x: The input tensor (for the model), of shape
                (batch_size, max_seq_len, input_size).
            x_len: The actual lengths of the sequences in the batch, even though
                they are 0-padded to max_seq_len in `x`.
            pinch_points: The indices of the pinch points in each of the
                sequences in the batch. This tensor has shape (batch_size,).
            y: The target tensor, of shape (batch_size,).
            query_ids: The list of query IDs, of shape (batch_size,).
            tpcds_temp_and_q_idx: The list of TPC-DS template and query indices,
                of shape (batch_size,).
            run_ids: The list of run IDs, of shape (batch_size,).
        """
        (x, pinch_points, y, query_ids, tpcds_temp_and_q_idx, run_ids) = zip(
            *batch
        )
        x_len = [len(i) for i in x]
        sort_idx = torch.tensor(
            np.argsort(x_len)[::-1].copy(), dtype=torch.long
        )

        # Sort batch by sequence length (optional but good οfor efficiency)
        x_out = [x[i] for i in sort_idx]
        x_len_out = torch.tensor(x_len, dtype=torch.long)[sort_idx]
        pinch_points_out = torch.tensor(pinch_points, dtype=torch.long)[
            sort_idx
        ]
        y_out = torch.tensor(y, dtype=torch.float)[sort_idx]
        query_ids_out: list[str] = [query_ids[i] for i in sort_idx]
        tpcds_temp_and_q_idx_out: list[Trace.TPCDSTempAndQIdx] = [
            tpcds_temp_and_q_idx[i] for i in sort_idx
        ]
        run_ids_out: list[str] = [run_ids[i] for i in sort_idx]

        # Pad sequences to the maximum length per batch
        padded_x_out = pad_sequence(x_out, batch_first=True, padding_value=0)
        return (
            padded_x_out,
            x_len_out,
            pinch_points_out,
            y_out,
            query_ids_out,
            tpcds_temp_and_q_idx_out,
            run_ids_out,
        )

    def save_to(self, path: str) -> None:
        """
        Saves the dataset to disk at the specified path.

        Parameters:
            path: The file path where the dataset will be saved.
        """
        torch.save(
            {
                "x": self.x,
                "pinch_points": self.pinch_points,
                "y": self.y,
                "query_ids": self.query_ids,
                "tpcds_temp_and_q_idx": self.tpcds_temp_and_q_idx,
                "run_ids": self.run_ids,
            },
            path,
        )

    @classmethod
    def load_from(cls, path: str) -> "ConcurrentQueryDataset":
        """
        Loads a dataset from disk at the specified path.

        Parameters:
            path: The file path from where the dataset will be loaded.

        Returns:
            An instance of ConcurrentQueryDataset loaded from the specified path.
        """
        data = torch.load(path)
        return cls(
            x=data["x"],
            pinch_points=data["pinch_points"],
            y=data["y"],
            query_ids=data["query_ids"],
            tpcds_temp_and_q_idx=data["tpcds_temp_and_q_idx"],
            run_ids=data["run_ids"],
        )

    @staticmethod
    def build_from_query_groups(
        iconq_interaction_featurizer: IconqInteractionFeaturizer,
        base_queries: list[QueryInfo],
        query_neighbors: dict[str, list[QueryInfo]],
        use_log_runtime: bool = True,
        run_id: Optional[str] = None,
    ) -> "ConcurrentQueryDataset":
        """
        Build a dataset from base queries and their pre-computed neighbors.

        This method constructs interaction features between a base query and its neighbors,
        all assumed to be on the same cluster. 

        Parameters:
            base_queries: List of QueryInfo objects for which to build dataset rows.
            query_neighbors: Dict mapping query_id to list of neighboring QueryInfo objects.
                Each query in base_queries should have an entry in this dict. If
                the neighbors include the base query itself, it will be ignored.
            use_log_runtime: Whether to use log(runtime) as the target variable.
            run_id: Optional run ID to associate with all queries.

        Returns:
            A ConcurrentQueryDataset ready for training or inference.
        """

        # If empty, return empty dataset
        if not base_queries or len(base_queries) == 0:
            return ConcurrentQueryDataset(
                x=[],
                pinch_points=torch.tensor([], dtype=torch.int8),
                y=torch.tensor([], dtype=torch.float32),
                query_ids=[],
                tpcds_temp_and_q_idx=[],
                run_ids=[],
            )

        # Build dataset components
        x = []
        y = []
        pinch_points = []
        query_ids_out = []
        tpcds_temp_and_q_idx_out = []
        run_ids_out = []

        for base_query in base_queries:
            neighbors = query_neighbors.get(base_query.query_id, [])

            # Build interaction features
            interaction_featurizations: dict[float, list[float]] = {}

            # Add self-interaction (helps with isolated queries)
            interaction_featurizations[base_query.start_time_s] = (
                iconq_interaction_featurizer.featurize_from_vectors(
                    cluster_name=base_query.cluster_name,
                    qa_features=base_query.query_featurization,
                    qa_start_time_s=base_query.start_time_s,
                    qa_latency_prediction=base_query.stage_latency_prediction,
                    qb_features=base_query.query_featurization,
                    qb_start_time_s=base_query.start_time_s,
                    qb_latency_prediction=base_query.stage_latency_prediction,
                )
            )

            # Add neighbor interactions, sorted by start time
            for neighbor in sorted(neighbors, key=lambda q: q.start_time_s):
                if neighbor.query_id == base_query.query_id:
                    continue  # Skip self-interaction; already added
                interaction_featurizations[neighbor.start_time_s] = (
                    iconq_interaction_featurizer.featurize_from_vectors(
                        cluster_name=base_query.cluster_name,
                        qa_features=base_query.query_featurization,
                        qa_start_time_s=base_query.start_time_s,
                        qa_latency_prediction=base_query.stage_latency_prediction,
                        qb_features=neighbor.query_featurization,
                        qb_start_time_s=neighbor.start_time_s,
                        qb_latency_prediction=neighbor.stage_latency_prediction,
                    )
                )

            # Sort interactions by start time to create the sequence
            neighbor_sort_order = sorted(interaction_featurizations.keys())

            # Add to dataset
            x.append(
                torch.stack(
                    [
                        torch.tensor(
                            interaction_featurizations[start_time],
                            dtype=torch.float32,
                        )
                        for start_time in neighbor_sort_order
                    ]
                )
            )
            if base_query.latency_s is not None:
                latency = base_query.latency_s
                y.append(latency if not use_log_runtime else np.log1p(latency))
            else:
                y.append(0.0)  # Placeholder if latency is not available
            pinch_points.append(
                neighbor_sort_order.index(base_query.start_time_s)
            )
            query_ids_out.append(base_query.query_id)
            tpcds_temp_and_q_idx_out.append(base_query.tpcds_temp_and_q_idx)
            run_ids_out.append(run_id if run_id is not None else "unknown")

        # Convert to tensors
        x_tensorized = x
        pinch_points_tensorized = torch.tensor(pinch_points, dtype=torch.int8)
        y_tensorized = torch.tensor(y, dtype=torch.float32)

        return ConcurrentQueryDataset(
            x=x_tensorized,
            pinch_points=pinch_points_tensorized,
            y=y_tensorized,
            query_ids=query_ids_out,
            tpcds_temp_and_q_idx=tpcds_temp_and_q_idx_out,
            run_ids=run_ids_out,
        )
