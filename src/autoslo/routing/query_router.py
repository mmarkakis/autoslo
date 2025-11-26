from abc import abstractmethod

from autoslo.blueprints.blueprint import Blueprint
from autoslo.utils.class_with_factory import ClassWithFactory


class QueryRouter(ClassWithFactory):
    """
    An abstract base class for routing queries to clusters within the context
    of a blueprint. Can be instantiated via the from_name factory method.
    """

    def __init__(self, blueprint: Blueprint, *args, **kwargs) -> None:
        """
        Initialize a QueryRouter instance.

        Parameters:
            blueprint: The Blueprint instance containing the clusters to route
                queries to.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.
        """
        self._blueprint = blueprint

    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this QueryRouter.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint

    @abstractmethod
    def route_query(self, query: str, *args, **kwargs) -> str:
        """
        Given a query string, determine the appropriate cluster to route it to.

        Parameters:
            query: The SQL query string to be routed.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The cluster name to which the query should be routed.
        """
        pass
