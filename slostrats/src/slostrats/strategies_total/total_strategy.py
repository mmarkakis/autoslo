from abc import ABC, abstractmethod


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
        pass

    @abstractmethod
    def suggest_blueprint(self, latency_slo_s: float, *args, **kwargs):
        """
        Suggest the best blueprint for the next period based on the provided
        latency SLO.

        Parameters:
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            The suggested blueprint for the next period.
        """
        raise NotImplementedError("Subclasses should implement this method.")
