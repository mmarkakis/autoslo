"""
TPC-DS Matching Featurizer.

A standalone featurizer that extracts a 5-dimensional feature vector from both
TPC-DS Trace objects and Redset DataFrames, for the purpose of mapping Redset
fingerprints to TPC-DS (template, query_index) pairs based on nearest-neighbour
distance in CORAL-aligned feature space.

The 5 features are:
    - num_joins
    - num_scans
    - num_aggregations
    - num_permanent_tables_accessed
    - log_mbytes_scanned  (i.e. log1p of megabytes scanned)
"""

import os

import numpy as np
import pandas as pd

import autoslo.utils.paths as pu
from autoslo.workload_execution.trace import Trace


class TpcdsMatchingFeaturizer:
    """
    Extracts the 5-dimensional matching feature vector used to map Redset
    fingerprints to TPC-DS (template, query_index) pairs.
    """

    FEATURE_NAMES: list[str] = [
        "num_joins",
        "num_scans",
        "num_aggregations",
        "num_permanent_tables_accessed",
        "log_mbytes_scanned",
    ]
    """The ordered feature column names produced by this featurizer."""

    _DATA_SUBDIR = "tpcds_matching_features"

    def __init__(self, reference_trace_run_id: str) -> None:
        """
        Initializes this matching featurizer based on the trace of the
        given TPC-DS run. The queries in the trace are featurized and then
        we compute reference statistics, so that we can later apply CORAL
        transformation to the features of the queries of the incoming
        Redset traces, to better align the feature distributions of the TPC-DS.

        Parameters:
            reference_trace_run_id: The run_id of the TPC-DS trace that will be
                used as reference for computing the normalization and whitening
                statistics.
        """

        self.reference_trace_run_id = reference_trace_run_id

        # Check if we already have computed and saved the normalization stats
        # for this reference trace.
        save_dir = os.path.join(
            pu.get_data_path(),
            "tpcds_normalization_stats",
            f"{reference_trace_run_id}",
        )
        os.makedirs(save_dir, exist_ok=True)
        stats_path = os.path.join(save_dir, "stats.parquet")
        covs_path = os.path.join(save_dir, "covs.parquet")

        if os.path.exists(stats_path) and os.path.exists(covs_path):
            # If the stats already exist, load them.
            self.stats_df = pd.read_parquet(stats_path)
            self.covs_df = pd.read_parquet(covs_path)
            return

        # Read in and mask valid queries from the reference trace.
        trace = Trace(reference_trace_run_id)
        aborted = trace.was_aborted()
        cached = trace.was_cached()
        valid = ~aborted & ~cached

        # Extract per-query features (Series indexed by query_id).
        num_joins = trace.num_joins()
        num_scans = trace.num_scans()
        num_aggs = trace.num_aggregates()
        num_perm_tables = trace.num_permanent_tables()
        log_mbytes = np.log1p(trace.mbytes_scanned())
        tpcds_ids = trace.tpcds_temp_and_q_idxs

        # Construct aggregated dataframe.
        df = pd.DataFrame(
            {
                "tpcds_temp_and_q_idx": tpcds_ids,
                "num_joins": num_joins,
                "num_scans": num_scans,
                "num_aggregations": num_aggs,
                "num_permanent_tables_accessed": num_perm_tables,
                "log_mbytes_scanned": log_mbytes,
            }
        )
        df = df.loc[valid]
        df = df.dropna(subset=["tpcds_temp_and_q_idx"])
        agg_df = df.groupby("tpcds_temp_and_q_idx")[self.FEATURE_NAMES].median()

        # Compute and save normalization stats.
        feature_df = agg_df[TpcdsMatchingFeaturizer.FEATURE_NAMES]
        means = feature_df.mean()
        stds = feature_df.std(ddof=1).clip(lower=1e-6)
        covs = feature_df.cov()

        self.stats_df = pd.DataFrame(
            [means, stds], index=["mean", "std"], columns=self.FEATURE_NAMES
        )
        self.stats_df.to_parquet(stats_path)

        covs_path = os.path.join(save_dir, "covs.parquet")
        self.covs_df = pd.DataFrame(
            covs.values, columns=self.FEATURE_NAMES, index=self.FEATURE_NAMES
        )
        self.covs_df.to_parquet(covs_path)

    def featurize_trace(self, run_id: str, align: bool = False) -> pd.DataFrame:
        """
        Extract the feature vectors for all queries in the given trace. 
        Optionally, also apply CORAL transformation to better align the feature 
        distributions to the reference trace.

        Parameters:
            run_id: The run_id of the trace to featurize.
            align: If True, apply CORAL transformation.

        Returns:
            A DataFrame indexed by ``tpcds_temp_and_q_idx`` with columns
            matching :pyattr:`FEATURE_NAMES`.
        """
        trace = Trace(run_id)
        df = pd.DataFrame(
            {
                "tpcds_temp_and_q_idx": trace.tpcds_temp_and_q_idxs,
                "num_joins": trace.num_joins(),
                "num_scans": trace.num_scans(),
                "num_aggregations": trace.num_aggregates(),
                "num_permanent_tables_accessed": trace.num_permanent_tables(),
                "log_mbytes_scanned": np.log1p(trace.mbytes_scanned()),
            }
        )
        df = df.dropna(subset=["tpcds_temp_and_q_idx"])
        df = df.set_index("tpcds_temp_and_q_idx")
        if align:
            df = self.align(df)
        return df[self.FEATURE_NAMES]
    

    def featurize_redset_rows(self, df: pd.DataFrame, align: bool = False) -> pd.DataFrame:
        """
        Convert Redset columns into the 5-dimensional feature space. Optionally, 
        also apply CORAL transformation to better align the feature

        Parameters:
            df: A DataFrame with at least the columns ``num_joins``,
                ``num_scans``, ``num_aggregations``,
                ``num_permanent_tables_accessed``, and ``mbytes_scanned``.
            align: If True, apply CORAL transformation.

        Returns:
            A DataFrame with exactly the 5 feature columns defined by
            :pyattr:`FEATURE_NAMES`.  Row count is preserved (one row in →
            one row out).
        """
        out = df[
            [
                "num_joins",
                "num_scans",
                "num_aggregations",
                "num_permanent_tables_accessed",
            ]
        ].copy()
        out["log_mbytes_scanned"] = np.log1p(df["mbytes_scanned"])
        if align:
            out = self.align(out)
        return out[TpcdsMatchingFeaturizer.FEATURE_NAMES]

    def align(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply CORAL transformation to better align the feature distributions to
        the TPC-DS reference trace.

        Parameters:
            df: A DataFrame with at least the columns ``tpcds_temp_and_q_idx``,
                ``num_joins``, ``num_scans``, ``num_aggregations``,
                ``num_permanent_tables_accessed``, and ``mbytes_scanned``.

        Returns:
            A DataFrame indexed by ``tpcds_temp_and_q_idx`` with columns
            matching :pyattr:`FEATURE_NAMES`.
        """
        feature_df = df[self.FEATURE_NAMES]
        normalized = (
            feature_df - self.stats_df.loc["mean"]
        ) / self.stats_df.loc["std"]
        cov_source = feature_df.cov()
        cov_target = self.covs_df
        whitening = np.linalg.inv(np.linalg.cholesky(cov_source))
        coloring = np.linalg.cholesky(cov_target)
        aligned_values = normalized.values @ whitening @ coloring
        aligned_df = pd.DataFrame(
            aligned_values, columns=self.FEATURE_NAMES, index=df.index
        )

        return aligned_df
