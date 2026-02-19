import os

import numpy as np
import pandas as pd
import yaml

from autoslo.workload_definition.query import TPCDSTempAndQIdx


class TPCDSSampler:
    """
    Sampler for TPC-DS queries based on a precomputed probability distribution
    of templates and query indices (TPCDSTempAndQIdx) conditioned on query
    latency bins.
    """

    def __init__(
        self,
        dist: np.ndarray,
        index_dict: dict[int, float],
        column_dict: dict[int, TPCDSTempAndQIdx],
    ):
        self.index_dict = index_dict
        self.column_dict = column_dict
        self.latency_bin_left_edges_s = list(index_dict.values())

        # Perform checks.
        assert isinstance(dist, np.ndarray)
        assert dist.shape[0] == len(index_dict)
        assert dist.shape[1] == len(column_dict)
        prev_bin_edge = -float("inf")
        for bin_left_edge in self.latency_bin_left_edges_s:
            assert bin_left_edge > prev_bin_edge
            prev_bin_edge = bin_left_edge

        # Precompute CDFs
        dist = dist.astype(np.float64)
        dist /= dist.sum(axis=1, keepdims=True)
        self.C = np.cumsum(dist, axis=1)
        self.C[:, -1] = 1.0  # guard against tiny floating error

    @staticmethod
    def from_dir(tpcds_prob_distribution_dir: str) -> "TPCDSSampler":
        # Read in the TPC-DS probability distribution and mapping.
        dist_path = os.path.join(tpcds_prob_distribution_dir, "array.npy")
        with open(dist_path, "rb") as f:
            dist = np.load(f)
        index_dict_path = os.path.join(
            tpcds_prob_distribution_dir, "index_dict.yml"
        )
        with open(index_dict_path, "r") as f:
            # This is a dictionary from row index to the left edge of
            # the corresponding latency bin, in seconds.
            index_dict = yaml.safe_load(f)
        column_dict_path = os.path.join(
            tpcds_prob_distribution_dir, "column_dict.yml"
        )
        with open(column_dict_path, "r") as f:
            # This is a dictionary from column index to the corresponding
            # TPCDSTempAndQIdx.
            column_dict = yaml.safe_load(f)
        return TPCDSSampler(dist, index_dict, column_dict)

    def sample(self, latencies_s: pd.Series, seed: int) -> pd.Series:
        """
        Sample TPC-DS queries for the given query latencies (in seconds) using
        the TPC-DS probability distribution.

        Parameters:
            latencies_s: A pandas Series of query latencies in seconds, indexed
                by query ID.
            seed: The random seed to use for sampling.

        Returns:
            A pandas Series of sampled TPCDSTempAndQIdx values, indexed by
                query ID.
        """

        # Detemrine the latency bin for each query.
        latency_bin_idxs = (
            np.searchsorted(
                self.latency_bin_left_edges_s, latencies_s, side="right"
            )
            - 1
        )

        # Sample.
        rng = np.random.default_rng(seed=seed)
        u = rng.random(len(latencies_s))
        A_codes = np.empty(len(latencies_s))
        for b in range(len(self.index_dict)):
            idxs = np.flatnonzero(latency_bin_idxs == b)
            A_codes[idxs] = np.searchsorted(self.C[b], u[idxs], side="right")

        temp_and_q_idxs = [ 
            self.column_dict[i] for i in A_codes.astype(int)
        ]

        return pd.Series(temp_and_q_idxs, index=latencies_s.index)
