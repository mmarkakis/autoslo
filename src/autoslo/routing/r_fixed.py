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

        Raises:
            ValueError: If the fixed_cluster_name is not the first-ordered
                cluster for its RPUs, or if the cluster is not found in the
                corresponding blueprint.
        """
        # Assert that the cluster name is the first-ordered cluster for its RPUs.
        # not strictly necessary but simplifies the naming of the blueprint.
        cluster = Cluster.from_config(cluster_name=fixed_cluster_name)
        rpu = cluster.rpu
        if (
            Cluster.ordered_cluster_names_per_rpu()[rpu][0]
            != fixed_cluster_name
        ):
            raise ValueError(
                f"Cluster name {fixed_cluster_name} is not the first-ordered "
                f"cluster for RPU {rpu}."
            )

        self._blueprint = Blueprint.from_config(f"single_({rpu})")
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
