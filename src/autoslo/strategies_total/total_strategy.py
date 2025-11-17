from abc import ABC, abstractmethod

from autoslo.blueprints.blueprint import Blueprint
from autoslo.prediction.p_exact import PExact
from autoslo.strategies_enumeration.enumeration_strategy import (
    EnumerationStrategy,
)
from autoslo.strategies_prediction.prediction_strategy import (
    PredictionStrategy,
)
from autoslo.strategies_selection.selection_strategy import SelectionStrategy


class TotalStrategy(ABC):
    """
    Interface for a total strategy, which encompasses:
    - An enumeration strategy to enumerate blueprints for each period.
    - A prediction strategy to predict the performance of each blueprint
      with respect to a given SLO, for each period.
    - A selection strategy to select the best blueprint from the enumerated
      blueprints based on their predictions, for each period.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a TotalStrategy instance.

        Parameters:
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        self.es: EnumerationStrategy
        self.ps: PredictionStrategy
        self.ss: SelectionStrategy
        pass

    def es_class(self) -> type[EnumerationStrategy]:
        """
        Get the class of the enumeration strategy used.

        Returns:
            The class (type) of the enumeration strategy.
        """
        return self.es.__class__

    def es_name(self) -> str:
        """
        Get the name of the enumeration strategy used.

        Returns:
            The name (str) of the enumeration strategy.
        """

        return self.es_class().__name__

    def ps_class(self) -> type[PredictionStrategy]:
        """
        Get the class of the prediction strategy used.

        Returns:
            The class (type) of the prediction strategy.
        """
        return self.ps.__class__

    def ps_name(self) -> str:
        """
        Get the name of the prediction strategy used.

        Returns:
            The name (str) of the prediction strategy.
        """

        return self.ps_class().__name__

    def ss_class(self) -> type[SelectionStrategy]:
        """
        Get the class of the selection strategy used.

        Returns:
            The class (type) of the selection strategy.
        """
        return self.ss.__class__

    def ss_name(self) -> str:
        """
        Get the name of the selection strategy used.

        Returns:
            The name (str) of the selection strategy.
        """
        return self.ss_class().__name__

    @abstractmethod
    def suggest_blueprint(
        self,
        workload_name: str,
        day_idx: int,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> Blueprint:
        """
        Suggest the best blueprint for the specified workload and day based on
        the provided latency SLO.

        Parameters:
            workload_name: The name of the workload to suggest for.
            day_idx: The index of the day for which the suggestion is made.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            The suggested blueprint for the specified workload and day.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def perf_of_suggested_blueprint(
        self,
        workload_name: str,
        day_idx: int,
        latency_slo_s: float,
    ) -> tuple[Blueprint, PExact]:
        """
        Get the actual performance of the suggested blueprint for the specified
        workload and day based on the provided latency SLO.

        Parameters:
            workload_name: The name of the workload to suggest for.
            day_idx: The index of the day for which the suggestion is made.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            The actual performance of the suggested blueprint.
        """
        suggested_blueprint = self.suggest_blueprint(
            workload_name=workload_name,
            day_idx=day_idx,
            latency_slo_s=latency_slo_s,
        )
        _perf_of_suggested_blueprint = self.ps.actual(
            workload_name=workload_name,
            day_idx=day_idx,
            blueprint=suggested_blueprint,
            latency_slo_s=latency_slo_s,
        )
        return suggested_blueprint, _perf_of_suggested_blueprint
