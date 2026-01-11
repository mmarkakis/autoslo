import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from autoslo.workload_execution.trace import Trace


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
    ):
        self.x = x
        self.pinch_points = pinch_points
        self.y = y
        self.query_ids = query_ids
        self.tpcds_temp_and_q_idx = tpcds_temp_and_q_idx

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        str,
        Trace.TPCDSTempAndQIdx,
    ]:
        return (
            self.x[idx],
            self.pinch_points[idx],
            self.y[idx],
            self.query_ids[idx],
            self.tpcds_temp_and_q_idx[idx],
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

        return ConcurrentQueryDataset(
            x=new_x,
            pinch_points=new_pinch_points,
            y=new_y,
            query_ids=new_query_ids,
            tpcds_temp_and_q_idx=new_tpcds_temp_and_q_idx,
        )

    @staticmethod
    def collate_and_pad(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, Trace.TPCDSTempAndQIdx]],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[Trace.TPCDSTempAndQIdx]
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
        """
        (x, pinch_points, y, query_ids, tpcds_temp_and_q_idx) = zip(*batch)
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

        # Pad sequences to the maximum length per batch
        padded_x_out = pad_sequence(x_out, batch_first=True, padding_value=0)
        return (padded_x_out, x_len_out, pinch_points_out, y_out, query_ids_out, tpcds_temp_and_q_idx_out)
    
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
        )
