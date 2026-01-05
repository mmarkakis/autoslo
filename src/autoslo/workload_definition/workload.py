from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    """Class representing a single query in the workload."""

    query_id: int
    start_time_s: float
    tpcds_temp_and_q_idx: str



class Workload(ABC):
    """Abstract base class for workload definitions."""

    @property
    @abstractmethod
    def queries(self) -> list[Query]:
        """Returns the list of queries in the workload."""
        pass