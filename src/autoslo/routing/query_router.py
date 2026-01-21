from abc import abstractmethod
from typing import Any

from autoslo.blueprints.blueprint import Blueprint
from autoslo.utils.class_with_factory import ClassWithFactory


class QueryRouter(ClassWithFactory):
    """
    An abstract base class for routing queries to clusters within the context
    of a blueprint. Can be instantiated via the from_name factory method.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a QueryRouter instance.

        Parameters:
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the QueryRouter instance.
        """
        raise NotImplementedError("Subclasses must implement name property.")

    @property
    @abstractmethod
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this QueryRouter.

        Returns:
            The Blueprint instance.
        """
        raise NotImplementedError(
            "Subclasses must implement blueprint property."
        )

    @abstractmethod
    def route_query(self, *args, **kwargs) -> str:
        """
        Given appropriate information, determine the appropriate cluster to
        route a query to.

        Parameters:
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The cluster name to which the query should be routed.
        """
        raise NotImplementedError(
            "Subclasses must implement route_query method."
        )

    def on_query_start(
        self, query_id: Any, cluster_name: str, *args, **kwargs
    ) -> None:
        """
        Called when a query starts executing on a cluster.

        Parameters:
            query_id: The ID of the query.
            cluster_name: The name of the cluster where the query is executed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """
        pass

    def on_query_finish(
        self, query_id: Any, cluster_name: str, *args, **kwargs
    ) -> None:
        """
        Called when a query finishes executing.

        Parameters:
            query_id: The ID of the query.
            cluster_name: The name of the cluster where the query is executed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """
        pass
