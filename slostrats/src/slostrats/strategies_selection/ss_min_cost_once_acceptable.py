from slostrats.building_blocks.blueprint import Blueprint
from slostrats.prediction.prediction import Prediction
from slostrats.strategies_selection.selection_strategy import SelectionStrategy
from slostrats.strategies_selection.ss_min_slo_violation_rate import (
    SSMinSLOViolationRate,
)


class SSMinCostOnceAcceptable(SelectionStrategy):
    """
    Selection strategy that selects the blueprint with the minimum predicted
    cost among those that are predicted to have acceptable SLO violation rate.
    """

    def __init__(
        self, slo_violation_rate_threshold: float, *args, **kwargs
    ) -> None:
        """
        Initialize the SSMinCostOnceAcceptable strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).
        """
        super().__init__(*args, **kwargs)
        self.slo_violation_rate_threshold = slo_violation_rate_threshold

    def select(
        self,
        bp_to_pred: dict[Blueprint, Prediction],
        *args,
        **kwargs,
    ) -> Blueprint:
        """
        Select the blueprint with the minimum predicted cost among those
        with an acceptable predicted SLO violation rate. If no blueprint has an
        acceptable predicted SLO violation rate, select the blueprint with the
        minimum predicted SLO violation rate.

        Parameters:
            bp_to_pred: A dictionary mapping Blueprint instances to their
                corresponding Prediction instances.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Returns:
            The Blueprint instance with the minimum predicted cost among those
                confident to meet the SLO.
        """

        if not bp_to_pred:
            raise ValueError("The bp_to_pred dictionary is empty.")

        selected_blueprint = None

        for blueprint, prediction in bp_to_pred.items():
            if prediction.has_predicted_slo_violation_rate_under(
                self.slo_violation_rate_threshold
            ):
                if (
                    selected_blueprint is None
                    or prediction.has_lower_predicted_cost(
                        bp_to_pred[selected_blueprint]
                    )
                ):
                    selected_blueprint = blueprint

        # Did we find at least one acceptable blueprint? If not, fall back to
        # selecting the one with the minimum predicted SLO violation rate.
        if selected_blueprint is None:
            return SSMinSLOViolationRate().select(bp_to_pred, *args, **kwargs)

        return selected_blueprint
