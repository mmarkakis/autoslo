import os

import pandas as pd
import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.routing.query_router import QueryRouter


class RModelBased(QueryRouter):
    """
    A QueryRouter implementation that routes queries based on a model-based
    approach, trained from a specific selector run.
    """

    def __init__(
        self,
        selector_run_id: str,
        iconq_query_featurizer_id: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize an RModelBased instance.

        Parameters:
            selector_run_id: The identifier for the selector run used to
                determine the sequence number to cluster mapping.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Raises:
            FileNotFoundError: If the mapping file for the given selector_run_id
                does not exist.
        """
        # Retrieve the exact mapping from the selector run, as well as the
        # workload to which it refers.
        self._selector_run_id = selector_run_id
        mapping_path = os.path.join(
            pu.get_data_path(), "selector_runs", selector_run_id, "mapping.yml"
        )
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"Mapping file not found for selector_run_id "
                f"{selector_run_id} at path {mapping_path}."
            )
        with open(mapping_path, "r") as f:
            self._mapping: dict[int, str] = yaml.safe_load(f)
        config_path = os.path.join(
            pu.get_data_path(), "selector_runs", selector_run_id, "config.yml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self._workload_name: str = config["workload_name"]

        # Create a blueprint that includes all clusters in the mapping
        cluster_names = sorted(list(set(self._mapping.values())))
        self._blueprint = Blueprint(
            clusters=[Cluster.from_config(name) for name in cluster_names]
        )

        # Initialize the featurizer and then get the featurization of each query
        # from the workload.
        self._iconq_query_featurizer_id = iconq_query_featurizer_id
        featurizer = IconqQueryFeaturizer.load(self._iconq_query_featurizer_id)
        workload_path = os.path.join(
            pu.get_data_path(),
            "chunks",
            self._workload_name,
            "chunk_workload.parquet",
        )
        workload_df = pd.read_parquet(workload_path)
        workload_df["tpcds_temp_and_q_idx"] = workload_df.apply(
            lambda row: f"{row['query_template']:03d}_{row['query_num_within_template']:03d}",
            axis=1,
        )
        workload_df["featurization"] = workload_df[
            "tpcds_temp_and_q_idx"
        ].apply(featurizer.featurize_from_tpcds_temp_and_q_idx)
        self._featurizations = {
            row["query_id"]: row["featurization"]
            for _, row in workload_df.iterrows()
        }

        # TODO: Continue here; implement a decision tree classifier that uses
        # the featurizations to predict the cluster name based on the mapping.

    @property
    def name(self) -> str:
        """
        Get the name of the RModelBased instance.
        """
        return (
            f"RModelBased(selector_run_id={repr(self._selector_run_id)}, "
            f"iconq_query_featurizer_id="
            f"{repr(self._iconq_query_featurizer_id)})"
        )

    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this RModelBased router.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint  
    
    def route_query(self,  *args, **kwargs) -> str:
        """
        Route the query based on its featurization.

        Parameters:
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The cluster name to which the query should be routed.
        """
        raise NotImplementedError(
            "RModelBased routing not yet implemented."
        )
