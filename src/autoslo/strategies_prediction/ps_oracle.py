from autoslo.blueprints.blueprint import Blueprint
from autoslo.prediction.prediction import Prediction
from autoslo.strategies_prediction.prediction_strategy import (
    PredictionStrategy,
)


class PSOracle(PredictionStrategy):
    """
    Prediction strategy that predicts the performance of blueprints based on
    full performance information of the actual performance of the workload,
    across different blueprints.
    """

    def predict(
        self,
        workload_name: str,
        day_idx: int,
        blueprint: Blueprint,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> Prediction:
        """
        Predict the performance of the future workload, by actually using its
        true performance on the blueprint.

        Parameters:
            workload_name: The name of the workload to predict for.
            day_idx: The index of the day for which the prediction is made.
            blueprint: A Blueprint instance to evaluate.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Returns:
            A Prediction instance corresponding to the evaluated blueprint.
        """
        return self.actual(
            workload_name,
            day_idx,
            blueprint,
            latency_slo_s,
        )
