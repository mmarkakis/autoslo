from abc import ABC, abstractmethod
from typing import Self


class Prediction(ABC):
    """
    Represents a prediction of the performance of a given workload on a given
    blueprint. The main focus is that predictions are meant to be compared.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize a Prediction instance.

        Parameters:
            args: Positional arguments (as needed by specific prediction types).
            kwargs: Keyword arguments (as needed by specific prediction types).
        """
        pass

    @abstractmethod
    def has_lower_predicted_slo_violation_rate(self, other: Self) -> bool:
        """
        Compare this prediction with another prediction based on their predicted
        SLO violation rates.

        Parameters:
            other: The other Prediction instance to compare.
        Returns:
            True if this prediction has a lower predicted SLO violation rate
                than the other prediction, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def has_lower_predicted_cost(self, other: Self) -> bool:
        """
        Compare this prediction with another prediction based on their predicted
        costs.

        Parameters:
            other: The other Prediction instance to compare.

        Returns:
            True if this prediction has a lower predicted cost than the other
                prediction, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def _has_predicted_slo_violation_rate_under(self, threshold: float) -> bool:
        """
        Check if a prediction's predicted SLO violation rate is under a given
        threshold.

        Parameters:
            threshold: The SLO violation rate threshold.

        Returns:
            True if the prediction's predicted SLO violation rate is under the
                threshold, False otherwise.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _validate_threshold(self, threshold: float) -> None:
        """
        Validate that the SLO violation rate threshold is between 0 and 1.

        Parameters:
            threshold: The SLO violation rate threshold.

        Raises:
            ValueError: If the threshold is not between 0 and 1.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                "The SLO violation rate threshold must be between 0 and 1."
            )

    def has_predicted_slo_violation_rate_under(self, threshold: float) -> bool:
        """
        Check if a prediction's predicted SLO violation rate is under a given
        threshold, after validating the threshold.

        Parameters:
            threshold: The SLO violation rate threshold.

        Returns:
            True if the prediction's predicted SLO violation rate is under the
                threshold, False otherwise.
        """
        self._validate_threshold(threshold)
        return self._has_predicted_slo_violation_rate_under(threshold)
