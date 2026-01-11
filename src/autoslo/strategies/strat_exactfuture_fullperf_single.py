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


class StratExactFutureFullPerfSingle(SLOStrategy):
    """
    SLO strategy that has oracle knowledge of the best single-cluster
    blueprint for the next period.
    """

    def __init__(
        self, slo_violation_rate_threshold: float, *args, **kwargs
    ) -> None:
        """
        Initialize the StratOracle strategy.

        Parameters:
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        super().__init__(*args, **kwargs)
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
        Suggest the best single-cluster blueprint based on oracle knowledge
        of the next period.

        Parameters:
            workload: The Composite workload to suggest for.
            day_idx: The index of the day for which the suggestion is made.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            blueprint: The suggested Blueprint instance.
            query_router: The associated QueryRouter instance.
        """
        options: list[tuple[Blueprint, QueryRouter]] = []
        option_perfs: list[E2ESLOMetrics] = []

        # For each candidate single-cluster blueprint, evaluate its performance
        # on the specified day.
        for rpu in Cluster.all_allowed_rpu_sizes():
            blueprint = Blueprint.one_cluster_with(rpu)
            query_router = RFixed(fixed_cluster_name=blueprint.cluster_names[0])
            options.append((blueprint, query_router))

            perf = SLOStrategy.evaluate_suggestion(
                workload,
                day_idx,
                latency_slo_s,
                blueprint,
                query_router,
            )

            option_perfs.append(SLOStrategyPerformance.aggregate([perf]))

        # Select the best-performing blueprint and its associated query router.
        best_perf = E2ESLOMetrics.best_among(
            option_perfs,
            self.violation_rate_threshold,
        )
        best_idx = option_perfs.index(best_perf)
        return options[best_idx]
