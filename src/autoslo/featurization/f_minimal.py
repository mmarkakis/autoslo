from typing import Any, Callable

import numpy as np
import pandas as pd

from autoslo.featurization.featurizer import Featurizer
from autoslo.workload_execution.trace import Trace


class FMinimal(Featurizer):
    """
    A minimal featurizer that extracts basic statistics from a trace.
    """

    def __init__(
        self,
        features_summary_metric: str = "mean",
        label_summary_metric: str = "p95",
        interarrival_summary_metric: str = "mean",
        label_in_log_space: bool = True,
        use_num_queries: bool = True,
        use_interarrival_time: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize a FMinimal instance.

        Parameters:
            features_summary_metric: The summary metric to use for aggregating
                features
                (default is "mean", other options are "p95" and "p99").
            label_summary_metric: The summary metric to use for aggregating
                the label
                (default is "p95", other options are "mean" and "p99").
            interarrival_summary_metric: The summary metric to use for
                aggregating interarrival times
                (default is "mean", other options are "p5" and "p1").
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
        if features_summary_metric not in {"mean", "p95", "p99"}:
            raise ValueError(
                "Unsupported features summary metric: "
                f"{features_summary_metric}"
            )
        if label_summary_metric not in {"mean", "p95", "p99"}:
            raise ValueError(
                f"Unsupported label summary metric: {label_summary_metric}"
            )
        if interarrival_summary_metric not in {"mean", "p5", "p1"}:
            raise ValueError(
                f"Unsupported interarrival summary metric: "
                f"{interarrival_summary_metric}"
            )

        self.features_summary_metric: str = features_summary_metric
        self.label_summary_metric: str = label_summary_metric
        self.interarrival_summary_metric: str = interarrival_summary_metric
        summary_funcs: dict[str, Callable] = {
            "mean": lambda s: s.mean(),
            "p95": lambda s: s.quantile(0.95),
            "p99": lambda s: s.quantile(0.99),
            "p5": lambda s: s.quantile(0.05),
            "p1": lambda s: s.quantile(0.01),
        }
        self.features_summary_func = summary_funcs[self.features_summary_metric]
        self.label_summary_func = summary_funcs[self.label_summary_metric]
        self.interarrival_summary_func = summary_funcs[
            self.interarrival_summary_metric
        ]
        self.label_in_log_space = label_in_log_space
        self.use_num_queries = use_num_queries
        self.use_interarrival_time = use_interarrival_time

    @property
    def name(self) -> str:
        """
        Get the name of the FMinimal instance.
        """
        return (
            f"FMinimal(features_summary_metric={self.features_summary_metric},"
            f"label_summary_metric={self.label_summary_metric},"
            f"interarrival_summary_metric={self.interarrival_summary_metric},"
            f"label_in_log_space={self.label_in_log_space},"
            f"use_num_queries={self.use_num_queries},"
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
        if self.use_interarrival_time:
            l.append(f"interarrival_time_s_{self.interarrival_summary_metric}")
        return l + [
            f"mbytes_scanned_{self.features_summary_metric}",
            f"num_joins_{self.features_summary_metric}",
            f"num_scans_{self.features_summary_metric}",
            f"num_aggregations_{self.features_summary_metric}",
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
        total_queries = trace.num_queries
        interarrival_times_s = (
            trace.arrival_times()
            .diff()
            .dropna()
            .apply(lambda x: x.total_seconds())
        )
        interarrival_time_stat = self.interarrival_summary_func(
            interarrival_times_s
        )
        mbytes_scanned_stat = self.features_summary_func(trace.mbytes_scanned())
        num_joins_stat = self.features_summary_func(trace.num_joins())
        num_scans_stat = self.features_summary_func(trace.num_scans())
        num_aggregates_stat = self.features_summary_func(trace.num_aggregates())
        duration_s_stat = self.label_summary_func(trace.latencies_s)
        if self.is_label_in_log_space:
            duration_s_stat = np.log1p(duration_s_stat)

        rpu_per_cluster = trace.rpu_per_cluster()
        # For simplicity, assume a single cluster and take its RPU.
        # FIXME: Extend to multi-cluster traces in the future.
        rpu = next(iter(rpu_per_cluster.values()))

        # Construct the feature vector based on selected features
        l = []
        if self.use_num_queries:
            l.append(float(total_queries))
        if self.use_interarrival_time:
            l.append(float(interarrival_time_stat))
        l.append(float(mbytes_scanned_stat))
        l.append(float(num_joins_stat))
        l.append(float(num_scans_stat))
        l.append(float(num_aggregates_stat))
        l.append(float(rpu))

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
        if self.use_interarrival_time:
            l.append(f"interarrival_time_s_{self.interarrival_summary_metric}")

        return l + [
            f"mbytes_scanned_{self.features_summary_metric}",
            f"num_joins_{self.features_summary_metric}",
            f"num_scans_{self.features_summary_metric}",
            f"num_aggregations_{self.features_summary_metric}",
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

        return df[self.feature_names + [self.label_name]]
