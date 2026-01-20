import os

import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.routing.query_router import QueryRouter


class RSeqNum(QueryRouter):
    """
    A QueryRouter implementation that routes queries based on their sequence
    number.
    """

    def __init__(self, selector_run_id: str, *args, **kwargs) -> None:
        """
        Initialize an RSeqNum instance.

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

        # TODO: have a check somewhere that we then use the router only for the
        # appropriate workload?

    @property
    def name(self) -> str:
        """
        Get the name of the RSeqNum instance.
        """
        return f"RSeqNum(selector_run_id={repr(self._selector_run_id)})"
    
    @property
    def workload_name(self) -> str:
        """
        Get the workload name associated with this RSeqNum router.

        Returns:
            The workload name.
        """
        return self._workload_name

    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this RSeqNum router.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint

    def route_query(
        self, workload_name: str, seq_num: int, *args, **kwargs
    ) -> str:
        """
        Route the query based on its sequence number.

        Parameters:
            workload_name: The name of the workload from which the query
                originates.
            seq_num: The sequence number of the query within the workload.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The name of the cluster to which the query with the given sequence
            number should be routed.

        Raises:
            ValueError: If the workload_name does not match the workload for
                which this router was configured, or if the sequence number is
                not found in the mapping.
        """
        if workload_name != self._workload_name:
            raise ValueError(
                f"Workload name {workload_name} does not match the workload "
                f"for which this router was configured: "
                f"{self._workload_name}."
            )
        cluster_name = self._mapping.get(seq_num, None)
        if cluster_name is None:
            raise ValueError(
                f"Sequence number {seq_num} not found in mapping for "
                f"selector_run_id {self._selector_run_id}."
            )
        return cluster_name
