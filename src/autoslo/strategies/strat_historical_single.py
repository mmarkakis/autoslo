from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.routing.query_router import QueryRouter
from autoslo.routing.r_fixed import RFixed
from autoslo.strategies.slo_strategy import SLOStrategy
from autoslo.strategies.slo_strategy_performance import (
    E2ESLOMetrics,
    SLOStrategyPerformance,
)
from autoslo.workload_definition.composite import Composite


class StratHistoricalSingle(SLOStrategy):
    """
    SLO strategy that uses past data from the most recent periods to
    predict and select the best single-cluster blueprint for the next period.

    # FIXME: We should accept the historical bluepirnts over the training window
    # as input, and only access the ground truth on those blueprints. For the
    # rest, we should be using a model. Or we can keep this one as-is and create
    # a new strategy that uses the model this way. 
    """

    def __init__(
        self,
        window_size: int,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the StratHistoricalSingle strategy.

        Parameters:
            window_size: The number of past periods to consider for prediction.
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.violation_rate_threshold = slo_violation_rate_threshold

    def suggest(
        self,
        workload: Composite,
        day_idx: int,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> tuple[Blueprint, QueryRouter]:
        """
        Base suggestions on how each single-cluster blueprint would have
        performed over the past `window_size` periods.
        """
        options: list[tuple[Blueprint, QueryRouter]] = []
        option_perfs: list[E2ESLOMetrics] = []

        # For each candidate single-cluster blueprint, evaluate its past
        # performance over the specified window size.
        for rpu in Cluster.all_allowed_rpu_sizes():
            blueprint = Blueprint.one_cluster_with(rpu)
            query_router = RFixed(blueprint, blueprint.cluster_names[0])
            options.append((blueprint, query_router))
            day_perfs = []

            for past_day_idx in range(
                max(0, day_idx - self.window_size), day_idx
            ):
                # Evaluate how this blueprint would have performed in the past
                # period by co-opting evaluate_suggestion.
                past_perf = self.evaluate_suggestion(
                    workload,
                    past_day_idx,
                    latency_slo_s,
                    blueprint,
                    query_router,
                )
                day_perfs.append(past_perf)

            # Aggregate the performance over the past periods.
            option_perfs.append(SLOStrategyPerformance.aggregate(day_perfs))

        # Select the best-performing blueprint and its associated query router.
        best_perf = E2ESLOMetrics.best_among(
            option_perfs,
            self.violation_rate_threshold,
        )
        best_idx = option_perfs.index(best_perf)
        return options[best_idx]


class StratHistoricalSingle1(StratHistoricalSingle):
    """
    A StratHistoricalSingle strategy that uses a past window size of 1.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            window_size=1,
            slo_violation_rate_threshold=slo_violation_rate_threshold,
        )


class StratHistoricalSingle7(StratHistoricalSingle):
    """
    A StratHistoricalSingle strategy that uses a past window size of 7.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            window_size=7,
            slo_violation_rate_threshold=slo_violation_rate_threshold,
        )


class StratHistoricalSingle14(StratHistoricalSingle):
    """
    A StratHistoricalSingle strategy that uses a past window size of 14.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            window_size=14,
            slo_violation_rate_threshold=slo_violation_rate_threshold,
        )
