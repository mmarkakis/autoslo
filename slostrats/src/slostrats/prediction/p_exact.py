from typing import Self

from slostrats.prediction.prediction import Prediction


class PExact(Prediction):
    """
    Prediction class that represents exact predictions for a blueprint. These
    predictions explicitly store the predicted SLO violation rate and cost of
    operating the blueprint on the target period.
    """

    def __init__(
        self, slo_violation_rate: float, cost: float, *args, **kwargs
    ) -> None:
        """
        Initialize a PExact instance.

        Parameters:
            slo_violation_rate: The predicted SLO violation rate.
            cost: The predicted cost for operating the blueprint.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).
        """
        super().__init__(*args, **kwargs)
        self._slo_violation_rate = slo_violation_rate
        self._cost = cost

    @property
    def slo_violation_rate(self) -> float:
        """
        Get the predicted SLO violation rate.

        Returns:
            The predicted SLO violation rate.
        """
        return self._slo_violation_rate

    @property
    def cost(self) -> float:
        """
        Get the predicted cost for operating the blueprint.

        Returns:
            The predicted cost.
        """
        return self._cost

    def has_lower_predicted_slo_violation_rate(self, other: Self) -> bool:
        """
        Compare this prediction with another prediction based on their predicted
        SLO violation rates.

        Parameters:
            other: The other PExact instance to compare.

        Returns:
            True if this prediction has a lower predicted SLO violation rate
                than the other prediction, False otherwise.
        """
        return self.slo_violation_rate < other.slo_violation_rate

    def has_lower_predicted_cost(self, other: Self) -> bool:
        """
        Compare this prediction with another prediction based on their predicted
        costs.

        Parameters:
            other: The other PExact instance to compare.

        Returns:
            True if this prediction has a lower predicted cost than the other
                prediction, False otherwise.
        """
        return self.cost < other.cost

    def _has_predicted_slo_violation_rate_under(self, threshold: float) -> bool:
        """
        Check if this prediction's predicted SLO violation rate is under a given
        threshold.

        Parameters:
            threshold: The SLO violation rate threshold.

        Returns:
            True if the prediction's predicted SLO violation rate is under the
                threshold, False otherwise.
        """
        return self.slo_violation_rate < threshold
