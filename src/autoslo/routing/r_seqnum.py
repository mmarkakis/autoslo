from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.routing.query_router import QueryRouter
from autoslo.workload_execution.trace import Trace
import os
import yaml
import autoslo.utils.paths as pu


class RSeqNum(QueryRouter):
    """
    A QueryRouter implementation that routes queries based on their sequence
    number.
    """

    def __init__(self, selector_run_id: str, *args, **kwargs) -> None:
        """
        Initialize an RSeqNum instance.

        Parameters:
            fixed_cluster_name: The name of the cluster to which all queries
                will be routed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Raises:
            ValueError: If the fixed_cluster_name is not the first-ordered
                cluster for its RPUs, or if the cluster is not found in the
                corresponding blueprint.
        """
        self._selector_run_id = selector_run_id
        mapping_path = os.path.join(
            pu.get_data_path(), "selector_runs", selector_run_id, "mapping.yml"
        )
        with open(mapping_path, "r") as f:
            self._mapping: dict[int, str] = yaml.safe_load(f)

        # Create a blueprint that includes all clusters in the mapping
        cluster_names = sorted(list(set(self._mapping.values())))
        self._blueprint = Blueprint(
            clusters=[Cluster.from_config(name) for name in cluster_names]
        )

    @property
    def name(self) -> str:
        """
        Get the name of the RSeqNum instance.
        """
        return f"RSeqNum(selector_run_id={repr(self._selector_run_id)})"

    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this RSeqNum router.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint

    def route_query(self, seq_num: int, *args, **kwargs) -> str:
        """
        Route the query based on its sequence number.

        Parameters:
            query: The SQL query string to be routed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The name of the fixed cluster.
        """
        return self._mapping[seq_num]
