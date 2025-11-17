from abc import ABC, abstractmethod

from autoslo.blueprints.blueprint import Blueprint


class EnumerationStrategy(ABC):
    """
    Base class for enumeration strategies.

    An enumeration strategy defines how to enumerate possible cluster
    configurations (blueprints) for the next period in the future.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize an EnumerationStrategy instance.

        Parameters:
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        pass

    @abstractmethod
    def enumerate(self, *args, **kwargs) -> list[Blueprint]:
        """
        Enumerate possible cluster configurations (blueprints) for the next
        period.

        Parameters:
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            A list of Blueprint instances representing possible cluster
                configurations.
        """
        raise NotImplementedError("Subclasses should implement this method.")
