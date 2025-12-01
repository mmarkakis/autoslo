from dataclasses import dataclass
import pandas as pd


@dataclass
class E2ESLOMetrics:
    msvr: float  # Mean SLO Violation Rate
    tc: float  # Total Cost
    moo: float  # Mean Online Overhead

    @staticmethod
    def best_among(
        metrics_list: list["E2ESLOMetrics"], slo_violation_rate_threshold: float
    ) -> "E2ESLOMetrics":
        """
        From a list of E2ESLOMetrics, select the best one based on the
        specified SLO violation rate threshold. The best metric is defined as
        the one with the lowest total cost among those that meet the SLO
        violation rate threshold. If there are ties, choose the one with the
        lowest mean online overhead. If none meet the threshold, return the one
        with the lowest SLO violation rate.

        Parameters:
            metrics_list: A list of E2ESLOMetrics instances to evaluate.
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold.

        Returns:
            The best E2ESLOMetrics instance based on the criteria.
        """
        # Filter metrics that meet the SLO violation rate threshold
        acceptable_metrics = [
            m for m in metrics_list if m.msvr <= slo_violation_rate_threshold
        ]

        if acceptable_metrics:
            # Choose the one with the lowest total cost, breaking ties with
            # lowest mean online overhead
            best_metric = min(acceptable_metrics, key=lambda m: (m.tc, m.moo))
        else:
            # If none are acceptable, choose the one with the lowest SLO
            # violation rate
            best_metric = min(metrics_list, key=lambda m: m.msvr)
        return best_metric


class SLOStrategyPerformance:
    """
    A class to help quantify the performance of an SLO strategy.
    """

    def __init__(
        self,
        latencies_s: list[float],
        costs: list[float],
        routing_times_s: list[float],
        latency_slo_s: float,
    ) -> None:
        """
        Initialize a SLOStrategyPerformance instance.

        Parameters:
            latencies_s: List of latency values in seconds.
            costs: List of costs associated with each cluster.
            routing_times_s: List of the time taken to route each query.
            latency_slo_s: The latency SLO in seconds that was targeted.

        Raises:
            ValueError: If the lengths of latencies_s and routing_times_s do not
                match.
        """
        if not len(latencies_s) == len(routing_times_s):
            raise ValueError(
                "latencies_s and routing_times_s must have the same length, "
                "representing the same number of queries. Instead, we got "
                f"{len(latencies_s)} latencies and {len(routing_times_s)} "
                "routing times."
            )

        self._latencies_s = latencies_s
        self._costs = costs
        self._routing_times_s = routing_times_s
        self._latency_slo_s = latency_slo_s

    @property
    def latencies_s(self) -> list[float]:
        """
        Get the list of latency values in seconds.

        Returns:
            A list of latency values in seconds.
        """
        return self._latencies_s

    @property
    def costs(self) -> list[float]:
        """
        Get the list of costs associated with each cluster.

        Returns:
            A list of costs.
        """
        return self._costs

    @property
    def routing_times_s(self) -> list[float]:
        """
        Get the list of the time taken to route each query.

        Returns:
            A list of routing times in seconds.
        """
        return self._routing_times_s

    @property
    def latency_slo_s(self) -> float:
        """
        Get the latency SLO in seconds.

        Returns:
            The latency SLO in seconds.
        """
        return self._latency_slo_s
    
    def latency_s_at_quantile(self, quantile: float) -> float:
        """
        Calculate the latency at the specified quantile.

        Parameters:
            quantile: The quantile to calculate (between 0 and 1).

        Returns:
            The latency at the specified quantile as a float.

        Raises:
            ValueError: If the quantile is not between 0 and 1.
        """
        if not 0.0 < quantile < 1.0:
            raise ValueError("Quantile must be between 0 and 1.")

        return float(pd.Series(self._latencies_s).quantile(quantile))

    def total_cost(self) -> float:
        """
        Calculate the total cost across all clusters, in dollars.

        Returns:
            The total cost as a float.
        """
        return sum(self._costs)

    def num_slo_violations(self) -> int:
        """
        Calculate the number of SLO violations based on the latency SLO.

        Returns:
            The number of SLO violations as an integer.
        """
        return sum(
            1 for latency in self._latencies_s if latency > self._latency_slo_s
        )

    def num_total_queries(self) -> int:
        """
        Calculate the total number of queries

        Returns:
            The total number of queries as an integer.
        """
        return len(self._latencies_s)

    def slo_violation_rate(self) -> float:
        """
        Calculate the SLO violation rate based on the latency SLO.

        Returns:
            The SLO violation rate as a float between 0 and 1.
        """
        total_queries = len(self._latencies_s)
        if total_queries == 0:
            return 0.0
        violations = self.num_slo_violations()
        return violations / total_queries

    def total_routing_time_s(self) -> float:
        """
        Calculate the total routing time across all queries, in seconds.

        Returns:
            The total routing time as a float.
        """
        return sum(self._routing_times_s)

    def mean_routing_time_s(self) -> float:
        """
        Calculate the mean routing time per query, in seconds.

        Returns:
            The mean routing time as a float.
        """
        total_requests = len(self._routing_times_s)
        if total_requests == 0:
            return 0.0
        total_time = self.total_routing_time_s()
        return total_time / total_requests

    @staticmethod
    def aggregate(
        daily_performances: list["SLOStrategyPerformance"],
    ) -> E2ESLOMetrics:
        """
        Process multiple SLOStrategyPerformance instances (from several days)
        and produce the end to end metrics we care about.

        Parameters:
            daily_performances: A list of SLOStrategyPerformance instances.

        Returns:
            An E2ESLOMetrics instance containing the aggregated metrics.
        """
        total_violations = 0
        total_queries = 0
        total_cost = 0.0
        total_routing_time_s = 0.0

        for perf in daily_performances:
            total_violations += perf.num_slo_violations()
            total_queries += perf.num_total_queries()
            total_cost += perf.total_cost()
            total_routing_time_s += perf.total_routing_time_s()

        msvr = (total_violations / total_queries) if total_queries > 0 else 0.0
        tc = total_cost
        moo = (
            (total_routing_time_s / total_queries) if total_queries > 0 else 0.0
        )
        return E2ESLOMetrics(msvr=msvr, tc=tc, moo=moo)
