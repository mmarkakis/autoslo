from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.routing.query_router import QueryRouter


class RFixed(QueryRouter):
    """
    A QueryRouter implementation that always routes queries to a fixed cluster.
    """

    def __init__(self, fixed_cluster_name: str, *args, **kwargs) -> None:
        """
        Initialize an RFixed instance.

        Parameters:
            fixed_cluster_name: The name of the cluster to which all queries
                will be routed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """
        cluster = Cluster.from_config(cluster_name=fixed_cluster_name)
        self._blueprint = Blueprint(clusters=[cluster])
        if fixed_cluster_name not in self._blueprint.cluster_names:
            raise ValueError(
                f"Cluster name {fixed_cluster_name} not found in blueprint."
            )

        self._fixed_cluster_name = fixed_cluster_name

    @property
    def name(self) -> str:
        """
        Get the name of the RFixed instance.
        """
        return f"RFixed(fixed_cluster_name={repr(self._fixed_cluster_name)})"
    
    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this RFixed router.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint

    def route_query(self, *args, **kwargs) -> str:
        """
        Route the query to the fixed cluster.

        Parameters:
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The name of the fixed cluster.
        """
        return self._fixed_cluster_name
