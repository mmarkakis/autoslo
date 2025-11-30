import os
import pickle
from abc import abstractmethod

import autoslo.utils.paths as pu
from autoslo.utils.class_with_factory import ClassWithFactory
from autoslo.workload_execution.trace import Trace


class Featurizer(ClassWithFactory):
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

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """
        Get the ordered names of the features produced by this featurizer.

        Returns:
            A list of feature names.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def featurize_trace(
        self, trace: Trace, force: bool = False, *args, **kwargs
    ) -> WorkloadFeaturization:
        """
        Featurize a given trace into a vector.

        Parameters:
            trace: A Trace instance to featurize.
            force: If True, forces re-computation of the featurization even if
                a cached version exists.
            args: Positional arguments (as needed by specific featurizers).
            kwargs: Keyword arguments (as needed by specific featurizers).

        Returns:
            A featurization vector representing the trace.
        """
        # Check for cached featurization and load it if available.
        run_dir = os.path.join(pu.get_runs_path(), trace.run_id)
        file_name = f"featurization_{self.name}.pkl"
        featurization_path = os.path.join(run_dir, file_name)

        # Check for cached featurization and load it if available.
        if os.path.exists(featurization_path) and not force:
            with open(featurization_path, "rb") as f:
                featurization = pickle.load(f)
            return featurization

        # Compute the featurization and cache it.
        featurization = self._featurize_trace_impl(trace, *args, **kwargs)
        with open(featurization_path, "wb") as f:
            pickle.dump(featurization, f)

        return featurization

    @abstractmethod
    def _featurize_trace_impl(
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
