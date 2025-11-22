from abc import ABC, abstractmethod

from autoslo.blueprints.blueprint import Blueprint
from autoslo.routing.query_router import QueryRouter
from autoslo.strategies.slo_strategy_performance import SLOStrategyPerformance
from autoslo.workload_definition.composite import Composite


class SLOStrategy(ABC):
    """
    Interface for a strategy that determines the best blueprint for a given SLO,
    and the associated query router to operate the blueprint.
    Optionally, a strategy may accept workload representations or performance
    data from previous periods to inform its decisions.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize an SLOStrategy instance.

        Parameters:
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        pass

    @abstractmethod
    def suggest(
        self,
        workload: Composite,
        day_idx: int,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> tuple[Blueprint, QueryRouter]:
        """
        For the specified workload and day, suggest the best blueprint and
        associated query router to meet the given latency SLO.

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

        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def evaluate_suggestion(
        workload: Composite,
        day_idx: int,
        latency_slo_s: float,
        blueprint: Blueprint,
        query_router: QueryRouter,
    ) -> SLOStrategyPerformance:
        """
        Get the actual performance of the suggested blueprint for the specified
        workload and day based on the provided latency SLO.

        Parameters:
            workload: The Composite workload to suggest for.
            day_idx: The index of the day for which the suggestion is made.
            latency_slo_s: The latency SLO in seconds to evaluate against.
            blueprint: The suggested Blueprint instance.
            query_router: The associated QueryRouter instance.

        Returns:
            A SLOStrategyPerformance instance containing performance metrics.

        Raises:
            IndexError: If the specified day index is out of range for the
                workload.
        """

        if day_idx >= len(workload.days):
            raise IndexError(
                f"Day index {day_idx} is out of range for workload "
                f"'{workload.name}' with {len(workload.days)} days."
            )
        day = workload.days[day_idx]

        trace = day.get_most_recent_trace_on(
            blueprint_name=blueprint.name, query_router_name=query_router.name
        )

        perf = SLOStrategyPerformance(
            latencies_s=trace.latencies_s,
            costs=trace.costs,
            routing_times_s=trace.routing_times_s,
            latency_slo_s=latency_slo_s,
        )

        return perf
