from typing import Callable

from autoslo.featurization.featurizer import Featurizer
from autoslo.workload_execution.trace import Trace


class FMinimal(Featurizer):
    """
    A minimal featurizer that extracts basic statistics from a trace.
    """

    def __init__(self, summary_metric: str = "mean", *args, **kwargs) -> None:
        """
        Initialize a FMinimal instance.

        Parameters:
            summary_metric: The summary metric to use for aggregating statistics
                (default is "mean", other options are "p95" and "p99").
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Raises:
            ValueError: If an unsupported summary metric is provided.
        """

        if summary_metric not in {"mean", "p95", "p99"}:
            raise ValueError(f"Unsupported summary metric: {summary_metric}")

        self.summary_metric: str = summary_metric
        self.summary_func: Callable = {
            "mean": lambda s: s.mean(),
            "p95": lambda s: s.quantile(0.95),
            "p99": lambda s: s.quantile(0.99),
        }[self.summary_metric]
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        """
        Get the name of the FMinimal instance.
        """
        return f"FMinimal(summary_metric={self.summary_metric})"

    @property
    def feature_names(self) -> list[str]:
        """
        Get the ordered names of the features produced by this featurizer.
        """
        return [
            "num_queries",
            f"mbytes_scanned_{self.summary_metric}",
            f"num_joins_{self.summary_metric}",
            f"num_scans_{self.summary_metric}",
            f"num_aggregations_{self.summary_metric}",
            "rpu",
        ]

    def _featurize_trace_impl(
        self, trace: Trace,
    ) -> Featurizer.WorkloadFeaturization:
        """
        Featurize a given trace into a vector of basic statistics.

        Parameters:
            trace: A Trace instance to featurize.
            rpu: Redshift processing units for the cluster.

        Returns:
            A featurization vector representing the trace.
        """
        total_queries = trace.num_queries
        mbytes_scanned_stat = self.summary_func(trace.mbytes_scanned())
        num_joins_stat = self.summary_func(trace.num_joins())
        num_scans_stat = self.summary_func(trace.num_scans())
        num_aggregates_stat = self.summary_func(trace.num_aggregates())

        rpu_per_cluster = trace.rpu_per_cluster()
        # For simplicity, assume a single cluster and take its RPU.
        # FIXME: Extend to multi-cluster traces in the future.
        rpu = next(iter(rpu_per_cluster.values()))

        return [
            float(total_queries),
            float(mbytes_scanned_stat),
            float(num_joins_stat),
            float(num_scans_stat),
            float(num_aggregates_stat),
            float(rpu),
        ]
