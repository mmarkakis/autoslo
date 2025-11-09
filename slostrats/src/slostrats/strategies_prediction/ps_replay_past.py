import os

import pandas as pd

from chunkload.building_blocks.composite import Composite
from chunkload.building_blocks.trace import Trace
from slostrats.building_blocks.blueprint import Blueprint
from slostrats.prediction.p_exact import PExact
from slostrats.strategies_prediction.prediction_strategy import (
    PredictionStrategy,
)


class PSReplayPast(PredictionStrategy):
    """
    Prediction strategy that predicts the performance of blueprints based on
    full performance information of past workloads, across different bluepints.
    """

    def __init__(
        self, window_size: int, per_period_average: bool, *args, **kwargs
    ) -> None:
        """
        Initialize the PSReplayPast strategy.

        Parameters:
            window_size: The size of the past time window to consider for
                predictions, in units of time periods. The prediction will be
                for one time period into the future.
            per_period_average: Whether to average metrics per time period,
                instead of pooling data across the entire window.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).
        """
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.per_period_average = per_period_average

    def predict(
        self,
        workload_name: str,
        day_idx: int,
        blueprint: Blueprint,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> PExact:
        """
        Predict the performance of an unknown future workload on a blueprint
        using historical data from a past time window.

        Parameters:
            workload_name: The name of the workload to predict for.
            day_idx: The index of the day for which the prediction is made.
            blueprint: A Blueprint instance to evaluate.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Returns:
            A Prediction instance corresponding to the evaluated blueprint.

        Raises:
            ValueError: If there are no past traces available for prediction.
        """

        # Find the most recent `window_size` past traces
        workload_dir = Composite.dir_for_composite_workload(
            workload_name=workload_name
        )
        earliest_day_idx = max(0, day_idx - self.window_size)
        rpu = blueprint.clusters[0].rpu  # TODO: Handle multi-cluster blueprints
        past_traces = [
            Trace.from_path(
                os.path.join(
                    workload_dir,
                    "day_traces",
                    f"day_{idx}",
                    f"{workload_name}_day{idx}_{rpu}.parquet",
                )
            )
            for idx in range(earliest_day_idx, day_idx)
        ]

        # Enusre there are nonzero past traces to go off of.
        if not past_traces:
            raise ValueError(
                "No past traces available for prediction; cannot proceed."
            )

        # Calculate the number of SLO-violating requests and cost
        # in the recent traces for the given blueprint
        slo_violating_queries = []
        total_queries = []
        total_cost = 0.0
        for trace in past_traces:
            slo_violating_queries.append(
                trace.num_queries_with_latency_over(latency_slo_s)
            )
            total_queries.append(trace.num_queries())
            total_billed_s = trace.billed_s()
            total_cost += blueprint.total_cost(
                # TODO: Handle multi-cluster blueprints
                {blueprint.cluster_names[0]: total_billed_s}
            )

        # Compute the SLO violation rate, either pooled or per-period average.
        slo_violation_rate = (
            sum(slo_violating_queries) / sum(total_queries)
            if not self.per_period_average
            else (
                sum(
                    slo_violating / total
                    for slo_violating, total in zip(
                        slo_violating_queries, total_queries
                    )
                )
                / len(past_traces)
            )
        )
        return PExact(slo_violation_rate=slo_violation_rate, cost=total_cost)
