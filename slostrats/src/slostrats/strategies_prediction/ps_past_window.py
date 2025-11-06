from slostrats.building_blocks.blueprint import Blueprint
from slostrats.prediction.prediction import Prediction
from slostrats.prediction.p_exact import PExact
from slostrats.strategies_prediction.prediction_strategy import (
    PredictionStrategy,
)

from chunkload.building_blocks.trace import Trace
from datetime import datetime, timedelta


class PSPastWindow(PredictionStrategy):
    """
    Prediction strategy that predicts the performance of blueprints based on
    historical data from a past time window.
    """

    def __init__(
        self, window_size: int, per_period_average: bool, *args, **kwargs
    ) -> None:
        """
        Initialize the PSPastWindow strategy.

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
        blueprint: Blueprint,
        latency_slo_s: float,
        past_traces: dict[datetime, Trace],
        *args,
        **kwargs,
    ) -> PExact:
        """
        Predict the performance of an unknown future workload on a blueprint
        using historical data from a past time window.

        Parameters:
            blueprint: A Blueprint instance to evaluate.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            past_traces: A dictionary of past workloads and their performance,
             mapping datetime instances to Trace objects.
            args: Positional arguments (not used).
            kwargs: Keyword arguments (not used).

        Returns:
            A Prediction instance corresponding to the evaluated blueprint.
        """

        # Find the most recent `window_size` past_traces
        if not past_traces:
            raise ValueError("The past_traces dictionary is empty.")
        sorted_timestamps = sorted(past_traces.keys(), reverse=True)
        recent_timestamps = sorted_timestamps[: self.window_size]

        # Calculate the number of SLO-violating requests and cost
        # in the recent traces for the given blueprint
        slo_violating_queries = []
        total_queries = []
        total_cost = 0.0
        for ts in recent_timestamps:
            trace = past_traces[ts]
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
                ) / len(recent_timestamps)
            )
        )
        return PExact(slo_violation_rate=slo_violation_rate, cost=total_cost)
