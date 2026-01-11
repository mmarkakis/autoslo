from abc import abstractmethod

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
