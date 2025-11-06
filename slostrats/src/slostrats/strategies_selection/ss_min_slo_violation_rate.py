from slostrats.building_blocks.blueprint import Blueprint
from slostrats.prediction.prediction import Prediction
from slostrats.strategies_selection.selection_strategy import SelectionStrategy


class SSMinSLOViolationRate(SelectionStrategy):
    """
    Selection strategy that selects the blueprint with the minimum predicted SLO
    violation rate.
    """

    def select(
        self,
        bp_to_pred: dict[Blueprint, Prediction],
        *args,
        **kwargs,
    ) -> Blueprint:
        """
        Select the blueprint with the minimum number of predicted SLO
        violations. Break ties arbitrarily.
        
        Parameters:
            bp_to_pred: A dictionary mapping Blueprint instances to their
                corresponding Prediction instances.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Returns:
            The Blueprint instance with the minimum number of predicted SLO
                violations.

        Raises:
            ValueError: If the bp_to_pred dictionary is empty.
        """
        if not bp_to_pred:
            raise ValueError("The bp_to_pred dictionary is empty.")

        selected_blueprint = bp_to_pred.keys().__iter__().__next__()

        for blueprint, prediction in bp_to_pred.items():
            if prediction.has_lower_predicted_slo_violation_rate(
                bp_to_pred[selected_blueprint]
            ):
                selected_blueprint = blueprint

        return selected_blueprint
