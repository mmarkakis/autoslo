from dataclasses import dataclass

from autoslo.slo.slo_metric import LatencySlo, SloMetric


@dataclass(frozen=True)
class SloObjective:
    """Bundles the SLO metric name and feasibility threshold."""

    slo_metric: SloMetric
    slo_threshold: float
    # Meaning of the threshold depends on the metric:
    # - "binary": max allowed violation rate (e.g. 0.01 for 1%)
    # - "absolute_s": max allowed total violation amount in seconds (e.g. 10.0)
    # - "relative": max allowed mean relative violation (e.g. 0.1 for 10%)

    def __init__(
        self,
        slo_metric: SloMetric | str,
        slo_threshold: float,
    ):
        object.__setattr__(
            self,
            "slo_metric",
            (
                slo_metric
                if isinstance(slo_metric, SloMetric)
                else SloMetric(slo_metric)
            ),
        )
        object.__setattr__(self, "slo_threshold", slo_threshold)

    def is_met(self, per_query_latency_slo: list[LatencySlo]) -> bool:
        """Return True if the given per-query (latency, SLO) pairs meet the
        SLO objective."""
        aggregated = self.slo_metric.aggregate_batch(per_query_latency_slo)
        return self.is_met_from_aggregated(aggregated)

    def is_met_from_aggregated(self, aggregated_violation: float) -> bool:
        """Return True if the given aggregated violation meets the SLO
        objective."""
        return aggregated_violation <= self.slo_threshold

    COMPARISON_TOLERANCE = 1e-4

    def _cmp_with_tolerance(
        self,
        a: float,
        b: float,
        tolerance: float = COMPARISON_TOLERANCE,
    ) -> int:
        """
        Compare two amounts with a tolerance.

        Returns -1 if a < b, 0 if approximately equal, 1 if a > b.
        """
        if a + tolerance < b:
            return -1
        elif b + tolerance < a:
            return 1
        return 0

    def is_b_better(
        self,
        metric_value_and_cost_a: tuple[float, float],
        metric_value_and_cost_b: tuple[float, float],
        tolerance: float = COMPARISON_TOLERANCE,
    ) -> bool:
        """
        Compare two (metric_value, cost) pairs according to the SLO objective.

        If both are under threshold, return the cheapest. Otherwise, return the
        one with the better metric value.
        """

        metric_a, cost_a = metric_value_and_cost_a
        metric_b, cost_b = metric_value_and_cost_b

        a_meets = self.is_met_from_aggregated(metric_a)
        b_meets = self.is_met_from_aggregated(metric_b)

        if a_meets and b_meets:
            return self._cmp_with_tolerance(cost_a, cost_b, tolerance) > 0
        return self._cmp_with_tolerance(metric_a, metric_b, tolerance) > 0
