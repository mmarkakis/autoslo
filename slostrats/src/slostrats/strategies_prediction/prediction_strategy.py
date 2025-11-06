from abc import ABC, abstractmethod

from slostrats.enumeration.blueprint import Blueprint
from slostrats.prediction.prediction import Prediction

class PredictionStrategy(ABC):
    """
    Strategy interface for predicting the performance of an unknown future 
    workload on a blueprint, with respect to a given SLO.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a PredictionStrategy instance.

        Parameters:
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        pass

    @abstractmethod
    def predict(
        self,
        blueprint: Blueprint,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> Prediction:
        """
        Predict the performance of an unknown future workload on a blueprint.

        Parameters:
            blueprint: A Blueprint instance to evaluate.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            A Prediction instance corresponding to the evaluated blueprint.
        """
        raise NotImplementedError("Subclasses should implement this method.")