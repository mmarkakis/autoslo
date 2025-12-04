from autoslo.blueprints.blueprint import Blueprint


class BlueprintTimeseries:
    """
    Represents a sequence of blueprint states over a set of time periods.
    """

    def __init__(self, period_idx_to_blueprint: dict[int, Blueprint]) -> None:
        """
        Initialize a BlueprintTimeseries instance.

        Parameters:
            period_idx_to_blueprint: A mapping from period indices to their
                corresponding Blueprint instances.
        """
        self._period_idx_to_blueprint = period_idx_to_blueprint

    @staticmethod
    def empty() -> "BlueprintTimeseries":
        """
        Create an empty BlueprintTimeseries instance.

        Returns:
            An empty BlueprintTimeseries instance.
        """
        return BlueprintTimeseries({})

    @staticmethod
    def one_cluster_fixed_size(
        cluster_rpu: float,
        total_periods: int,
    ) -> "BlueprintTimeseries":
        """
        Create a BlueprintTimeseries instance with a single cluster of fixed
        RPU for all periods.

        Parameters:
            cluster_rpu: The RPU of the single cluster.
            total_periods: The total number of periods.

        Returns:
            A BlueprintTimeseries instance with the specified configuration.
        """
        period_idx_to_blueprint = {
            period_idx: Blueprint.one_cluster_with(cluster_rpu)
            for period_idx in range(total_periods)
        }
        return BlueprintTimeseries(period_idx_to_blueprint)

    def blueprint_for_period(self, period_idx: int) -> Blueprint:
        """
        Get the blueprint for a specific period.

        Parameters:
            period_idx: The index of the period.

        Returns:
            The Blueprint instance for the specified period.

        Raises:
            KeyError: If the period index does not exist in the timeseries.
        """
        if period_idx not in self._period_idx_to_blueprint:
            raise KeyError(
                f"Period index '{period_idx}' not found in timeseries."
            )
        return self._period_idx_to_blueprint[period_idx]

    def set_blueprint_for_period(
        self,
        period_idx: int,
        blueprint: Blueprint,
        error_if_exists: bool = True,
    ) -> None:
        """
        Set a blueprint for a specific period.

        Parameters:
            period_idx: The index of the period.
            blueprint: The Blueprint instance to add.
            error_if_exists: If True, raise an error if a blueprint already
                exists for the specified period.

        Raises:
            ValueError: If a blueprint already exists for the specified period
                and error_if_exists is True.
        """
        if error_if_exists and (period_idx in self._period_idx_to_blueprint):
            raise ValueError(
                f"Period index '{period_idx}' already has a blueprint "
                f"assigned."
            )
        self._period_idx_to_blueprint[period_idx] = blueprint
