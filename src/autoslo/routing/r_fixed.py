from autoslo.blueprints.blueprint import Blueprint
from autoslo.routing.query_router import QueryRouter
from typing import Optional


class RFixed(QueryRouter):
    """
    A QueryRouter implementation that always routes queries to a fixed cluster.
    """

    def __init__(self, blueprint: Blueprint, fixed_cluster_name: str) -> None:
        """
        Initialize an RFixed instance.

        Parameters:
            blueprint: The Blueprint instance containing the clusters to route
                queries to.
            fixed_cluster_name: The name of the cluster to which all queries
                will be routed.

        Raises:
            ValueError: If the fixed_cluster_name is not found in the blueprint.
        """
        super().__init__(blueprint)
        if fixed_cluster_name not in blueprint.cluster_names:
            raise ValueError(
                f"Cluster name {fixed_cluster_name} not found in blueprint."
            )
        self._fixed_cluster_name = fixed_cluster_name

    @property
    def name(self) -> str:
        """
        Get the name of the RFixed instance in a reproducible format that can be
        parsed by QueryRouter.from_name.
        """
        return f"RFixed(fixed_cluster_name={repr(self._fixed_cluster_name)})"

    def route_query(self, query: str, *args, **kwargs) -> str:
        """
        Route the query to the fixed cluster.

        Parameters:
            query: The SQL query string to be routed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The name of the fixed cluster.
        """
        return self._fixed_cluster_name
