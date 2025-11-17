from abc import ABC, abstractmethod

from autoslo.blueprints.blueprint import Blueprint
from autoslo.prediction.prediction import Prediction


class SelectionStrategy(ABC):
    """
    Strategy interface for selecting the best blueprint from a set of
    enumerated blueprints based on their predictions.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a SelectionStrategy instance.

        Parameters:
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        pass

    @abstractmethod
    def select(
        self,
        bp_to_pred: dict[Blueprint, Prediction],
        *args,
        **kwargs,
    ) -> Blueprint:
        """
        Select the best blueprint based on the provided predictions.

        Parameters:
            bp_to_pred: A dictionary mapping Blueprint instances to their 
                corresponding Prediction instances.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            The selected Blueprint instance.
        """
        raise NotImplementedError("Subclasses should implement this method.")
