from enum import Enum
from typing import Iterable


class SloMetric(Enum):
    """Which SLO-violation metric to use for routing decisions.

    * ``BINARY``     – 1 if the query violates its SLO, else 0.
    * ``ABSOLUTE_S`` – seconds of overshoot, ``max(0, latency − SLO)``.
    * ``RELATIVE``   – relative overshoot, ``max(0, (latency − SLO) / SLO)``.
    * ``RELATIVE_UNCONSTRAINED`` – relative overshoot, but without the max.

    All three are always *reported*; this enum selects which one drives
    the routing optimiser.
    """

    BINARY = "binary"
    ABSOLUTE_S = "absolute_s"
    RELATIVE = "relative"
    RELATIVE_UNCONSTRAINED = "relative_unconstrained"

    def calculate(self, latency_s: float, slo_s: float) -> float | int:
        """Calculate the SLO violation according to this metric."""
        return self.calculate_batch([(latency_s, slo_s)])[0]

    def calculate_batch(
        self,
        lat_and_slos: Iterable[tuple[float, float]],
    ) -> list[float] | list[int]:
        """Vectorised version of *metric_dependent_violation*."""

        if self is SloMetric.BINARY:
            return [int(lat > slo) for lat, slo in lat_and_slos]
        if self is SloMetric.ABSOLUTE_S:
            return [max(0.0, lat - slo) for lat, slo in lat_and_slos]
        if self is SloMetric.RELATIVE:
            if any(slo <= 0 for _, slo in lat_and_slos):
                raise ValueError(
                    "All SLOs must be positive for relative violation metric."
                )
            return [max(0.0, (lat - slo) / slo) for lat, slo in lat_and_slos]
        if self is SloMetric.RELATIVE_UNCONSTRAINED:
            if any(slo <= 0 for _, slo in lat_and_slos):
                raise ValueError(
                    "All SLOs must be positive for relative violation metric."
                )
            return [(lat - slo) / slo for lat, slo in lat_and_slos]
        raise ValueError(f"Unknown SloMetric: {self}")

    def aggregate(self, violations: list[float] | list[int]) -> float:
        """
        Aggregate multiple per-query violations into a single value.

        For BINARY, this is the violation rate (mean).  For the others, it's
        the total violation (sum).
        """
        if len(violations) == 0:
            return 0.0
        if self is SloMetric.BINARY:
            return sum(violations) / len(violations) if violations else 0.0
        return sum(violations)

    def aggregate_batch(
        self, lat_and_slos: Iterable[tuple[float, float]]
    ) -> float:
        """Convenience: calculate and aggregate in one step."""
        return self.aggregate(self.calculate_batch(lat_and_slos))

    @property
    def aggregate_string_description(self) -> str:
        """Human-friendly description of the aggregated metric, for plot
        labels and such."""
        if self is SloMetric.BINARY:
            return "Violation Rate"
        if self is SloMetric.ABSOLUTE_S:
            return "Violation Amount (s)"
        if self is SloMetric.RELATIVE:
            return "Mean Relative Violation"
        if self is SloMetric.RELATIVE_UNCONSTRAINED:
            return "Mean Relative Violation (Unconstrained)"
        raise ValueError(f"Unknown SloMetric: {self}")
