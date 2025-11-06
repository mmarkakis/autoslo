from slostrats.enumeration.blueprint import Blueprint
from slostrats.enumeration.cluster import Cluster
from slostrats.strategies_enumeration.enumeration_strategy import (
    EnumerationStrategy,
)


class ESFixed(EnumerationStrategy):
    """
    Enumeration strategy that always enumerates a single Blueprint for the next
    period. The blueprint contains a single cluster with the specified fixed RPU
    capacity.
    """

    def __init__(self, rpu: int, *args, **kwargs) -> None:
        """
        Initialize an ESFixed instance.

        Parameters:
            rpu: The fixed RPU capacity for the cluster in the blueprint.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).
        """
        super().__init__(*args, **kwargs)
        self._rpu = rpu

    def enumerate(self, *args, **kwargs) -> list[Blueprint]:
        """
        Enumerate a blueprint with a single cluster of the specified fixed RPU
        capacity.

        Parameters:
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Returns:
            A list of Blueprint instances, each containing a single cluster
                with the specified fixed RPU capacity.
        """
        cluster = Cluster(rpu=self._rpu)
        blueprint = Blueprint(clusters=[cluster])
        return [blueprint]
