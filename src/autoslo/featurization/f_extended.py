from typing import Any, Callable

import numpy as np
import pandas as pd

from autoslo.featurization.featurizer import Featurizer
from autoslo.workload_execution.trace import Trace


class FExtended(Featurizer):
    """
    An extended featurizer that extracts more statistics from a trace.
    """

    def __init__(
        self,
        label_summary_metric: str = "p95",
        label_in_log_space: bool = True,
        use_num_queries: bool = True,
        use_interarrival_time: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize a FExtended instance.

        Parameters:
            label_summary_metric: The summary metric to use for aggregating the
                label (default is "p95", other options are "mean" and "p99").
            label_in_log_space: Whether the label should be in log space
                (default is True).
            use_num_queries: Whether to use the number of queries as a feature
                (default is True).
            use_interarrival_time: Whether to use interarrival time as a feature
                (default is False).
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Raises:
            ValueError: If an unsupported summary metric is provided.
        """
        super().__init__(*args, **kwargs)

        if label_summary_metric not in {"mean", "p95", "p99"}:
            raise ValueError(
                f"Unsupported label summary metric: {label_summary_metric}"
            )

        self.label_summary_metric: str = label_summary_metric
        self.summary_funcs: dict[str, Callable] = {
            "mean": lambda s: s.mean(),
            "p95": lambda s: s.quantile(0.95),
            "p99": lambda s: s.quantile(0.99),
            "p5": lambda s: s.quantile(0.05),
            "p1": lambda s: s.quantile(0.01),
        }
        self.label_summary_func = self.summary_funcs[self.label_summary_metric]
        self.label_in_log_space = label_in_log_space
        self.use_num_queries = use_num_queries
        self.use_interarrival_time = use_interarrival_time

    @property
    def name(self) -> str:
        """
        Get the name of the FExtended instance.
        """
        return (
            f"FExtended(label_summary_metric={self.label_summary_metric},"
            f"label_in_log_space={self.label_in_log_space}),"
            f"use_num_queries={self.use_num_queries}),"
            f"use_interarrival_time={self.use_interarrival_time})"
        )

    @property
    def feature_names(self) -> list[str]:
        """
        Get the ordered names of the features produced by this featurizer, which
        are intended as input to the models.

        Returns:
            A list of feature names.
        """
        l = []
        if self.use_num_queries:
            l.append("num_queries")
            l.append("nan_cluster_size_num_queries")
        if self.use_interarrival_time:
            l.append("interarrival_time_s_mean")
            l.append("interarrival_time_s_p5")
            l.append("interarrival_time_s_p1")
        return l + [
            "was_aborted_mean",
            "was_cached_mean",
            "num_permanent_tables_accessed_mean",
            "num_external_tables_accessed_mean",
            "num_system_tables_accessed_mean",
            "mbytes_scanned_mean",
            "mbytes_scanned_p95",
            "mbytes_scanned_p99",
            "num_joins_mean",
            "num_joins_p95",
            "num_joins_p99",
            "num_scans_mean",
            "num_scans_p95",
            "num_scans_p99",
            "num_aggregations_mean",
            "num_aggregations_p95",
            "num_aggregations_p99",
            "query_type_analyze_mean",
            "query_type_copy_mean",
            "query_type_ctas_mean",
            "query_type_delete_mean",
            "query_type_insert_mean",
            "query_type_other_mean",
            "query_type_select_mean",
            "query_type_unload_mean",
            "query_type_update_mean",
            "query_type_vacuum_mean",
            "rpu",
        ]

    @property
    def label_name(self) -> str:
        """
        Get the name of the label produced by this featurizer, which is intended
        as output from the models.

        Returns:
            The name of the output feature.
        """
        prefix = "log1p_" if self.is_label_in_log_space else ""
        return f"{prefix}latency_s_{self.label_summary_metric}"

    @property
    def is_label_in_log_space(self) -> bool:
        """
        Indicates whether the label produced by this featurizer is in log space.
        "Log space" means that the label has been transformed using `np.log1p`.

        Returns:
            True if the label is in log space, False otherwise.
        """
        return self.label_in_log_space

    def _featurize_trace_impl(
        self,
        trace: Trace,
    ) -> tuple[Featurizer.WorkloadFeaturization, Any]:
        """
        Featurize a given trace into a vector of basic statistics.

        Parameters:
            trace: A Trace instance to featurize.
            rpu: Redshift processing units for the cluster.

        Returns:
            features: The feature values.
            label: The label value.
        """

        l = []

        # Optional number of queries and interarrival time features
        if self.use_num_queries:
            l.append(float(trace.num_queries))
            l.append(0)  # We don't have nan_cluster_size_num_queries in traces
        if self.use_interarrival_time:
            interarrival_times_s = (
                trace.arrival_times()
                .diff()
                .dropna()
                .apply(lambda x: x.total_seconds())
            )
            for metric in ["mean", "p5", "p1"]:
                l.append(self.summary_funcs[metric](interarrival_times_s))

        # Was aborted and was cached
        l.append(float(trace.was_aborted().mean()))
        l.append(float(trace.was_cached().mean()))

        # Number of tables accessed
        l.append(float(trace.num_permanent_tables().mean()))
        l.append(float(trace.num_external_tables().mean()))
        l.append(float(trace.num_system_tables().mean()))

        # Mbytes scanned, number of joins, scans, aggregations
        bases = [
            trace.mbytes_scanned(),
            trace.num_joins(),
            trace.num_scans(),
            trace.num_aggregates(),
        ]
        for base in bases:
            for metric in ["mean", "p95", "p99"]:
                l.append(float(self.summary_funcs[metric](base)))

        # Query types (one-hot encoded)
        query_types = [
            "analyze",
            "copy",
            "ctas",
            "delete",
            "insert",
            "other",
            "select",
            "unload",
            "update",
            "vacuum",
        ]
        query_type_counts = trace.query_type().value_counts(normalize=True)
        query_type_counts.index = query_type_counts.index.str.lower()
        for qt in query_types:
            l.append(float(query_type_counts.get(qt, 0.0)))

        # RPU
        rpu_per_cluster = trace.rpu_per_cluster()
        # For simplicity, assume a single cluster and take its RPU.
        # FIXME: Extend to multi-cluster traces in the future.
        rpu = next(iter(rpu_per_cluster.values()))
        l.append(float(rpu))

        # Label
        duration_s_stat = self.label_summary_func(trace.latencies_s)
        if self.is_label_in_log_space:
            duration_s_stat = np.log1p(duration_s_stat)

        return l, float(duration_s_stat)

    @property
    def _required_redset_summary_columns(self) -> list[str]:
        """
        Get the list of Redset summary DataFrame columns required by this
        featurizer in order to compute the featurization.

        Returns:
            A list of required column names.
        """
        l = []
        if self.use_num_queries:
            l.append("num_queries")
            l.append("nan_cluster_size_num_queries")
        if self.use_interarrival_time:
            l.append(f"interarrival_time_s_mean")
            l.append(f"interarrival_time_s_p5")
            l.append(f"interarrival_time_s_p1")

        return l + [
            "was_aborted_mean",
            "was_cached_mean",
            "num_permanent_tables_accessed_mean",
            "num_external_tables_accessed_mean",
            "num_system_tables_accessed_mean",
            "mbytes_scanned_mean",
            "mbytes_scanned_p95",
            "mbytes_scanned_p99",
            "num_joins_mean",
            "num_joins_p95",
            "num_joins_p99",
            "num_scans_mean",
            "num_scans_p95",
            "num_scans_p99",
            "num_aggregations_mean",
            "num_aggregations_p95",
            "num_aggregations_p99",
            "query_type_analyze",
            "query_type_copy",
            "query_type_ctas",
            "query_type_delete",
            "query_type_insert",
            "query_type_other",
            "query_type_select",
            "query_type_unload",
            "query_type_update",
            "query_type_vacuum",
            "unique_cluster_size_count",
            "unique_cluster_sizes",
            f"duration_s_{self.label_summary_metric}",
        ]

    def _featurize_redset_impl(
        self,
        redset_summary_df: pd.DataFrame,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Featurize a Redset summary DataFrame.

        Parameters:
            redset_summary_df: A DataFrame containing the Redset summary to
                featurize.
            args: Positional arguments (as needed by specific featurizers).
            kwargs: Keyword arguments (as needed by specific featurizers).

        Returns:
            A dataframe where each row is the featurization of a distinct row in
            the input DataFrame.
        """

        df = redset_summary_df[
            redset_summary_df["unique_cluster_size_count"] == 1
        ].copy()
        df["cluster_size"] = df["unique_cluster_sizes"].apply(lambda x: x[0])
        df["rpu"] = df["cluster_size"] * 8
        df[self.label_name] = df[f"duration_s_{self.label_summary_metric}"]

        if self.is_label_in_log_space:
            df[self.label_name] = np.log1p(df[self.label_name])

        # Append "_mean" to query type columns
        for col in df.columns:
            if col.startswith("query_type_"):
                new_col = f"{col}_mean"
                df[new_col] = df[col]

        return df[self.feature_names + [self.label_name]]
