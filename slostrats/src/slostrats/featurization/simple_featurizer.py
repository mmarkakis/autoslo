from chunkload.building_blocks.trace import Trace
from slostrats.featurization.featurizer import Featurizer


class SimpleFeaturizer(Featurizer):
    """
    A simple featurizer that extracts basic statistics from a trace.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a SimpleFeaturizer instance.

        Parameters:
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).
        """
        super().__init__(*args, **kwargs)

    def featurize(
        self, trace: Trace, rpu: int
    ) -> Featurizer.WorkloadFeaturization:
        """
        Featurize a given trace into a vector of basic statistics.

        Parameters:
            trace: A Trace instance to featurize.
            rpu: Redshift processing units for the cluster.

        Returns:
            A featurization vector representing the trace.
        """
        total_queries = trace.num_queries()
        mbytes_scanned_mean = trace.mbytes_scanned_mean()
        num_joins_mean = trace.num_joins_mean()
        num_scans_mean = trace.num_scans_mean()
        num_aggregations_mean = trace.num_aggregations_mean()

        return [
            float(total_queries),
            float(mbytes_scanned_mean),
            float(num_joins_mean),
            float(num_scans_mean),
            float(num_aggregations_mean),
            float(rpu),
        ]