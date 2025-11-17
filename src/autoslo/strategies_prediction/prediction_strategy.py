import os
from abc import ABC, abstractmethod

import pandas as pd

from autoslo.workload_definition.composite import Composite
from autoslo.workload_execution.trace import Trace
from autoslo.blueprints.blueprint import Blueprint
from autoslo.prediction.p_exact import PExact
from autoslo.prediction.prediction import Prediction


class PredictionStrategy(ABC):
    """
    Strategy interface for predicting the performance of a workload day on a
    blueprint, with respect to a given SLO.
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
        workload_name: str,
        day_idx: int,
        blueprint: Blueprint,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> Prediction:
        """
        Predict the performance of a workload day on a blueprint.

        Parameters:
            workload_name: The name of the workload to predict for.
            day_idx: The index of the day for which the prediction is made.
            blueprint: A Blueprint instance to evaluate.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            A Prediction instance corresponding to the evaluated blueprint.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def actual(
        self,
        workload_name: str,
        day_idx: int,
        blueprint: Blueprint,
        latency_slo_s: float,
    ) -> PExact:
        """
        Retrieve the actual performance of a workload day on a blueprint.

        Parameters:
            workload_name: The name of the workload.
            day_idx: The index of the day.
            blueprint: A Blueprint instance to evaluate.
            latency_slo_s: The latency SLO in seconds to evaluate against.

        Returns:
            A Prediction instance corresponding to the evaluated blueprint.
        """
        # Find the correct trace for the given workload day and blueprint
        workload_dir = Composite.dir_for_composite_workload(
            workload_name=workload_name
        )
        rpu = blueprint.clusters[0].rpu  # TODO: Handle multi-cluster blueprints
        trace = Trace.from_path(
            os.path.join(
                workload_dir,
                "day_traces",
                f"day_{day_idx}",
                f"{workload_name}_day{day_idx}_{rpu}.parquet",
            )
        )

        slo_violation_rate = (
            trace.num_queries_with_latency_over(latency_slo_s)
            / trace.num_queries()
        )
        total_cost = blueprint.total_cost(
            # TODO: Handle multi-cluster blueprints
            {blueprint.cluster_names[0]: trace.billed_s()}
        )

        return PExact(slo_violation_rate=slo_violation_rate, cost=total_cost)
