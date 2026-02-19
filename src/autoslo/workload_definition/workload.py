from abc import ABC, abstractmethod
from autoslo.workload_definition.query import Query


class Workload(ABC):
    """Abstract base class for workload definitions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the workload."""
        pass

    @abstractmethod
    def queries(self, *args, **kwargs) -> list[Query]:
        """Returns the list of queries in the workload."""
        pass