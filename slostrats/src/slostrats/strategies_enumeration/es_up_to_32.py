from slostrats.enumeration.blueprint import Blueprint
from slostrats.strategies_enumeration.enumeration_strategy import EnumerationStrategy


class ESUpTo32(EnumerationStrategy):
    """
    Enumeration strategy that always enumerates the same set of Blueprints for
    the next period. Each blueprint contains a single cluster with RPU
    capacities ranging from 4 to 32, in powers of two.
    """

    def enumerate(self, *args, **kwargs) -> list[Blueprint]:
        """
        Enumerate blueprints with clusters of RPU capacities from 4 to 32,
        in powers of two.

        Parameters:
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Returns:
            A list of Blueprint instances, each containing a single cluster
                with RPU capacities ranging from 4 to 32, in powers of two.
        """
        return Blueprint.simple_blueprints_up_to_32_rpu()
