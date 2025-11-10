from typing import Optional


class Cluster:
    """
    Represents a computing cluster with a specified number of RPU (Redshift
    Processing Units).
    """

    US_EAST_1_COST_PER_RPU_HOUR = 0.375
    ONE_HOUR_S = 3600

    UP_TO_32_RPU_SIZES = [4, 8, 16, 32]
    ALL_ALLOWED_RPU_SIZES = UP_TO_32_RPU_SIZES

    def __init__(
        self,
        rpu: int,
        name: Optional[str] = None,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
    ) -> None:
        """
        Initialize a Cluster instance.

        Parameters:
            rpu: The number of RPU (Redshift Processing Units) for the cluster.
            name: Optional name for the cluster. If not provided, a default name
                will be generated.
            cost_per_rpu_hour: The cost per RPU per hour. Defaults to the
                US_EAST_1_COST_PER_RPU_HOUR constant.
        """
        self.rpu = rpu
        self.name = name if name is not None else f"cluster_{rpu}rpu"
        self.cost_per_rpu_hour = cost_per_rpu_hour

    def cost(self, duration_s: float = ONE_HOUR_S) -> float:
        """
        Calculate the cost of running the cluster for a given duration.

        Parameters:
            duration_s: The duration in seconds for which the cluster is run. If
                not provided, defaults to one hour.

        Returns:
            The total cost of running the cluster for the specified duration, in
                dollars.
        """
        hours = duration_s / self.ONE_HOUR_S
        total_cost = self.rpu * self.cost_per_rpu_hour * hours
        return total_cost
