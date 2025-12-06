import os
import pickle
from abc import abstractmethod

import pandas as pd
import pyarrow as pa

import autoslo.utils.paralellism as plu
import autoslo.utils.paths as pu
from autoslo.utils.class_with_factory import ClassWithFactory
from autoslo.workload_execution.trace import Trace


class Featurizer(ClassWithFactory):
    """
    Abstract base class for featurizers that convert traces into vectors that
    the model-based strategies can use for predictions. The same featurizers
    can also operate on split summaries of the Redset datasets.

    In the featurization, the input features are contained first, in the order
    specified by `input_feature_names`, followed by the output feature specified
    by `output_feature_name`.
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
        Get the ordered names of the features produced by this featurizer, which
        are intended as input to the models.

        Returns:
            A list of feature names.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    @abstractmethod
    def label_name(self) -> str:
        """
        Get the name of the label produced by this featurizer, which is intended
        as output from the models.

        Returns:
            The name of the output feature.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    @abstractmethod
    def is_label_in_log_space(self) -> bool:
        """
        Indicates whether the label produced by this featurizer is in log space.
        "Log space" means that the label has been transformed using `np.log1p`.

        Returns:
            True if the label is in log space, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    @abstractmethod
    def _required_redset_summary_columns(self) -> list[str]:
        """
        Get the list of Redset summary DataFrame columns required by this
        featurizer in order to compute the featurization.

        Returns:
            A list of required column names.
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

    def featurize_redset(
        self, redset_summary_name: str, force: bool = False, *args, **kwargs
    ) -> pd.DataFrame:
        """
        Featurize a Redset summary DataFrame.

        Parameters:
            redset_summary_name: Name of the Redset summary to featurize.
            force: If True, forces re-computation of the featurization even if
                a cached version exists.
            args: Positional arguments (as needed by specific featurizers).
            kwargs: Keyword arguments (as needed by specific featurizers).

        Returns:
            A list of featurization vectors representing the Redset summary.
        """
        pa.set_cpu_count(plu.inner_level_num_cpus())

        # Check that redset_summary_name is of the correct form and exists.
        splits = redset_summary_name.split("_")
        if len(splits) != 2:
            raise ValueError(
                f"redset_summary_name must be of the form "
                f"<split_name>_<granularity>, got '{redset_summary_name}'."
            )
        split_name, granularity = splits
        if split_name not in ["train", "validation", "test"]:
            raise ValueError(
                f"Split name must be one of 'train', 'validation', or 'test', "
                f"got '{split_name}'."
            )
        if granularity not in ["hourly", "daily"]:
            raise ValueError(
                f"Granularity must be either 'hourly' or 'daily', got "
                f"'{granularity}'."
            )

        # Check for cached featurization and load it if available.
        featurization_dir = os.path.join(
            pu.get_data_path(),
            "redset_byproducts",
            "provisioned",
            f"{granularity}_featurizations",
        )
        os.makedirs(featurization_dir, exist_ok=True)
        featurization_path = os.path.join(
            featurization_dir,
            f"{redset_summary_name}_{self.name}.parquet",
        )
        if os.path.exists(featurization_path) and not force:
            featurization_df = pd.read_parquet(
                featurization_path, engine="pyarrow"
            )
            return featurization_df

        # Compute the featurization and cache it.
        input_df_path = os.path.join(
            pu.get_data_path(),
            "redset_byproducts",
            "provisioned",
            f"{granularity}_splits",
            f"{redset_summary_name}.parquet",
        )
        if not os.path.exists(input_df_path):
            raise ValueError(
                f"Redset summary file '{input_df_path}' does not exist."
            )

        input_df = pd.read_parquet(
            input_df_path,
            columns=self._required_redset_summary_columns,
            engine="pyarrow",
        )
        featurization_df = self._featurize_redset_impl(
            input_df, *args, **kwargs
        )
        featurization_df.to_parquet(featurization_path, index=False)

        return featurization_df

    @abstractmethod
    def _featurize_redset_impl(
        self, redset_summary_df: pd.DataFrame, *args, **kwargs
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
        raise NotImplementedError("Subclasses should implement this method.")
