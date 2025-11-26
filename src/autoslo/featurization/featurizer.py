from abc import ABC, abstractmethod

import autoslo.utils.class_with_factory as au
from autoslo.workload_execution.trace import Trace


class Featurizer(ABC):
    """
    Abstract base class for featurizers that convert traces into vectors that
    the model-based strategies can use for predictions.

    TODO: The same featurizer must also provide ways to recover the same
    featurization from Redset.
    """

    WorkloadFeaturization = list[float]

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Featurizer instance.

        Parameters:
            args: Positional arguments (as needed by specific featurizers).
            kwargs: Keyword arguments (as needed by specific featurizers).
        """
        pass

    @abstractmethod
    def featurize_trace(
        self, trace: Trace, *args, **kwargs
    ) -> WorkloadFeaturization:
        """
        Featurize a given trace into a vector.

        Parameters:
            trace: A Trace instance to featurize.
            args: Positional arguments (as needed by specific featurizers).
            kwargs: Keyword arguments (as needed by specific featurizers).

        Returns:
            A featurization vector representing the trace.
        """
        raise NotImplementedError("Subclasses should implement this method.")
