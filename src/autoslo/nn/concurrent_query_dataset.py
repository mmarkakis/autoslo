from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.workload_definition.query import Query, QueryTextId


class ConcurrentQueryDataset(Dataset):
    """
    A PyTorch Dataset for concurrent query data.

    Each item in the dataset is a tuple, where:
        - x is a list of the the input tensor, of shape (seq_len, input_size).
            The list length is batch_size.
        - pinch_point is the pinch points tensor, of shape (1,).
        - y is the target tensor, of shape (1,).
        - query_ids is the list of query IDs.
        - query_text_id is the query text ID.
        - run_ids is the run ID associated with the query.
        - y_is_lower_bound is a tensor indicating if the target is a lower
            bound, of shape (1,).
    """

    def __init__(
        self,
        x: list[torch.Tensor],
        pinch_points: torch.Tensor,
        y: torch.Tensor,
        query_ids: list[str],
        query_text_id: list[QueryTextId],
        run_ids: list[str],
        y_is_lower_bound: torch.Tensor,
    ):
        self.x = x
        self.pinch_points = pinch_points
        self.y = y
        self.query_ids = query_ids
        self.query_text_id = query_text_id
        self.run_ids = run_ids
        self.y_is_lower_bound = y_is_lower_bound

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        QueryTextId,
        str,
        torch.Tensor,
    ]:
        return (
            self.x[idx],
            self.pinch_points[idx],
            self.y[idx],
            self.query_ids[idx],
            self.query_text_id[idx],
            self.run_ids[idx],
            self.y_is_lower_bound[idx],
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
        new_query_text_id = []
        for dataset in datasets:
            new_query_text_id.extend(dataset.query_text_id)
        new_run_ids = []
        for dataset in datasets:
            new_run_ids.extend(dataset.run_ids)
        new_y_is_lower_bound = torch.cat(
            [dataset.y_is_lower_bound for dataset in datasets], dim=0
        )

        return ConcurrentQueryDataset(
            x=new_x,
            pinch_points=new_pinch_points,
            y=new_y,
            query_ids=new_query_ids,
            query_text_id=new_query_text_id,
            run_ids=new_run_ids,
            y_is_lower_bound=new_y_is_lower_bound,
        )

    @staticmethod
    def collate_and_pad(
        batch: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                str,
                QueryTextId,
                str,
                torch.Tensor,
            ]
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[str],
        list[QueryTextId],
        list[str],
        torch.Tensor,
    ]:
        """
        Custom collation function for DataLoader of ConcurrentQueryDataset.
        It pads the sequences in the batch to the maximum length per batch.

        Parameters:
            batch: The batch of data to collate. Each element in the batch is a
                tuple produced by ConcurrentQueryDataset.

        Returns:
            x: The input tensor (for the model), of shape
                (batch_size, max_seq_len, input_size).
            x_len: The actual lengths of the sequences in the batch, even though
                they are 0-padded to max_seq_len in `x`.
            pinch_points: The indices of the pinch points in each of the
                sequences in the batch. This tensor has shape (batch_size,).
            y: The target tensor, of shape (batch_size,).
            query_ids: The list of query IDs, of shape (batch_size,).
            query_text_id: The list of QueryTextId objects,
                of shape (batch_size,).
            run_ids: The list of run IDs, of shape (batch_size,).
            y_is_lower_bound: The tensor indicating if the target is a lower
                bound, of shape (batch_size,).
        """
        (
            x,
            pinch_points,
            y,
            query_ids,
            query_text_id,
            run_ids,
            y_is_lower_bound,
        ) = zip(*batch)
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
        query_text_id_out: list[QueryTextId] = [
            query_text_id[i] for i in sort_idx
        ]
        run_ids_out: list[str] = [run_ids[i] for i in sort_idx]
        y_is_lower_bound_out = torch.tensor(y_is_lower_bound, dtype=torch.bool)[
            sort_idx
        ]

        # Pad sequences to the maximum length per batch
        padded_x_out = pad_sequence(x_out, batch_first=True, padding_value=0)
        return (
            padded_x_out,
            x_len_out,
            pinch_points_out,
            y_out,
            query_ids_out,
            query_text_id_out,
            run_ids_out,
            y_is_lower_bound_out,
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
                "query_text_id": self.query_text_id,
                "run_ids": self.run_ids,
                "y_is_lower_bound": self.y_is_lower_bound,
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
            query_text_id=data["query_text_id"],
            run_ids=data["run_ids"],
            y_is_lower_bound=data["y_is_lower_bound"],
        )

    @staticmethod
    def build_from_query_groups(
        iconq_interaction_featurizer: IconqInteractionFeaturizer,
        cluster_to_base_to_neighbors: dict[str, dict[Query, list[Query]]],
        targets: dict[str, float] | None = None,
        is_lower_bound: dict[str, bool] | None = None,
        use_log_runtime: bool = True,
    ) -> "ConcurrentQueryDataset":
        """
        Build a dataset from base queries and their pre-computed neighbors.

        This method constructs interaction features between a base query and its neighbors,
        all assumed to be on the same cluster.

        Parameters:
            iconq_interaction_featurizer: The featurizer to use for constructing
                interaction features between queries.
            cluster_to_base_to_neighbors: A dictionary mapping cluster names to
                dictionaries that map base queries to their list of neighboring
                queries.
            targets: Optional mapping of query_id → actual latency (the *y*
                value).  When *None* (inference), *y* defaults to 0.0.
            is_lower_bound: Optional mapping of query_id → whether the target
                is a censored (lower-bound) observation.  When *None*, defaults
                to False.
            use_log_runtime: Whether to use log(runtime) as the target variable.
           
        Returns:
            A ConcurrentQueryDataset ready for training or inference.
        """

        # If empty, return empty dataset
        if len(cluster_to_base_to_neighbors) == 0:
            return ConcurrentQueryDataset(
                x=[],
                pinch_points=torch.tensor([], dtype=torch.int8),
                y=torch.tensor([], dtype=torch.float32),
                query_ids=[],
                query_text_id=[],
                run_ids=[],
                y_is_lower_bound=torch.tensor([], dtype=torch.bool),
            )

        # Build dataset components
        x = []
        y = []
        pinch_points = []
        query_ids_out = []
        query_text_id_out = []
        run_ids_out = []
        y_is_lower_bound_out = []

        for cluster_name, base_to_neighbors in cluster_to_base_to_neighbors.items():
            # Derive RPU once per cluster for stage-prediction lookup.
            rpu = iconq_interaction_featurizer._get_rpu(cluster_name)

            def _stage_pred(q: Query) -> float:
                return q.stage_predictions_per_rpu.get(rpu, -1.0)

          

            # General path: distinct neighbor lists per base query.
            for base_query in base_to_neighbors.keys():

                # Build qb_entries: self-interaction first, then neighbors.
                qb_entries: list[
                    tuple[float, QueryTextId, float, bool]
                ] = [
                    (
                        base_query.rel_start_time_s,
                        base_query.query_text_id,
                        _stage_pred(base_query),
                        True,  # is_self
                    )
                ]
                for neighbor in base_to_neighbors[base_query]:
                    if neighbor.query_id != base_query.query_id:
                        qb_entries.append((
                            neighbor.rel_start_time_s,
                            neighbor.query_text_id,
                            _stage_pred(neighbor),
                            False,  # is_self
                        ))

                arr, pinch_idx = (
                    iconq_interaction_featurizer.featurize_one_vs_many_to_numpy(
                        cluster_name=cluster_name,
                        qa_query_text_id=base_query.query_text_id,
                        qa_start_time_s=base_query.rel_start_time_s,
                        qa_latency_prediction=_stage_pred(base_query),
                        qb_entries=qb_entries,
                    )
                )
                x.append(torch.from_numpy(arr))
                if targets is not None and base_query.query_id in targets:
                    latency = targets[base_query.query_id]
                    floored_latency = max(latency, 0.001)
                    y.append(
                        floored_latency
                        if not use_log_runtime
                        else np.log(floored_latency)
                    )
                else:
                    y.append(0.0)  # Placeholder if latency is not available
                pinch_points.append(pinch_idx)
                query_ids_out.append(base_query.query_id)
                query_text_id_out.append(base_query.query_text_id)
                run_ids_out.append(cluster_name)
                lb = (
                    is_lower_bound.get(base_query.query_id, False)
                    if is_lower_bound is not None
                    else False
                )
                y_is_lower_bound_out.append(lb)

        # Convert to tensors
        x_tensorized = x
        pinch_points_tensorized = torch.tensor(pinch_points, dtype=torch.int16)
        y_tensorized = torch.tensor(y, dtype=torch.float32)
        y_is_lower_bound_tensorized = torch.tensor(
            y_is_lower_bound_out, dtype=torch.bool
        )

        return ConcurrentQueryDataset(
            x=x_tensorized,
            pinch_points=pinch_points_tensorized,
            y=y_tensorized,
            query_ids=query_ids_out,
            query_text_id=query_text_id_out,
            run_ids=run_ids_out,
            y_is_lower_bound=y_is_lower_bound_tensorized,
        )
