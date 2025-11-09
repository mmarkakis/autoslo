from datetime import datetime
from typing import Type

from chunkload.building_blocks.trace import Trace
from slostrats.building_blocks.blueprint import Blueprint
from slostrats.prediction.prediction import Prediction
from slostrats.strategies_enumeration.es_up_to_32 import ESUpTo32
from slostrats.strategies_prediction.ps_replay_past import PSReplayPast
from slostrats.strategies_selection.ss_min_cost_once_acceptable import (
    SSMinCostOnceAcceptable,
)
from slostrats.strategies_selection.ss_min_slo_violation_rate import (
    SSMinSLOViolationRate,
)
from slostrats.strategies_total.total_strategy import TotalStrategy
from slostrats.strategies_selection.selection_strategy import SelectionStrategy


class TSReplayPast(TotalStrategy):
    """
    Total strategy that uses past data from the most recent periods to
    predict and select the best blueprint for the next period.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        window_size: int,
        selection_strategy: Type[SelectionStrategy],
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the TSReplayPast strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
            window_size: The number of past periods to consider for prediction.
            selection_strategy: The strategy to use for selecting the best
                blueprint.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        super().__init__(*args, **kwargs)

        self.violation_rate_threshold = slo_violation_rate_threshold

        self.es = ESUpTo32()
        self.ps = PSReplayPast(window_size=window_size, per_period_average=True)
        self.ss = selection_strategy(
            slo_violation_rate_threshold=self.violation_rate_threshold,
        )

    def suggest_blueprint(
        self,
        workload_name: str,
        day_idx: int,
        latency_slo_s: float,
        *args,
        **kwargs,
    ):
        """
        Suggest the best blueprint for the specified workload and day based on
        the provided latency SLO, using data from past periods.

        Parameters:
            workload_name: The name of the workload to suggest for.
            day_idx: The index of the day for which the suggestion is made.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            The suggested blueprint for the next period.
        """
        candidate_blueprints = self.es.enumerate()
        bp_to_prediction: dict[Blueprint, Prediction] = {}
        for blueprint in candidate_blueprints:
            prediction = self.ps.predict(
                workload_name,
                day_idx,
                blueprint,
                latency_slo_s,
            )
            bp_to_prediction[blueprint] = prediction
        selected_blueprint = self.ss.select(bp_to_prediction)
        return selected_blueprint


class TSReplayPast1Cost(TSReplayPast):
    """
    A TSReplayPast strategy that uses a past window size of 1 period and the
    SSMinCostOnceAcceptable selection strategy.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the TSReplayPast1Cost strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
        """
        super().__init__(
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            window_size=1,
            selection_strategy=SSMinCostOnceAcceptable,
        )


class TSReplayPast7Cost(TSReplayPast):
    """
    A TSReplayPast strategy that uses a past window size of 7 periods and the
    SSMinCostOnceAcceptable selection strategy.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the TSReplayPast7Cost strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
        """
        super().__init__(
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            window_size=7,
            selection_strategy=SSMinCostOnceAcceptable,
        )


class TSReplayPast14Cost(TSReplayPast):
    """
    A TSReplayPast strategy that uses a past window size of 14 periods and the
    SSMinCostOnceAcceptable selection strategy.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the TSReplayPast14Cost strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
        """
        super().__init__(
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            window_size=14,
            selection_strategy=SSMinCostOnceAcceptable,
        )


class TSReplayPast1Perf(TSReplayPast):
    """
    A TSReplayPast strategy that uses a past window size of 1 period and the
    SSMinSLOViolationRate selection strategy.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the TSReplayPast1Perf strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
        """
        super().__init__(
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            window_size=1,
            selection_strategy=SSMinSLOViolationRate,
        )


class TSReplayPast7Perf(TSReplayPast):
    """
    A TSReplayPast strategy that uses a past window size of 7 periods and the
    SSMinSLOViolationRate selection strategy.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the TSReplayPast7Perf strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
        """
        super().__init__(
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            window_size=7,
            selection_strategy=SSMinSLOViolationRate,
        )


class TSReplayPast14Perf(TSReplayPast):
    """
    A TSReplayPast strategy that uses a past window size of 14 periods and the
    SSMinSLOViolationRate selection strategy.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the TSReplayPast14Perf strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
        """
        super().__init__(
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            window_size=14,
            selection_strategy=SSMinSLOViolationRate,
        )
