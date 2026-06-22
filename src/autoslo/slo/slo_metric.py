from enum import Enum
from typing import Iterable, NamedTuple


class LatencySlo(NamedTuple):
    latency_s: float
    slo_s: float


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
        return self.calculate_batch([LatencySlo(latency_s, slo_s)])[0]

    def calculate_batch(
        self,
        lat_and_slos: Iterable[LatencySlo],
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

        All metrics use the mean: violation rate for BINARY, mean absolute
        overshoot for ABSOLUTE_S, mean relative overshoot for RELATIVE /
        RELATIVE_UNCONSTRAINED.
        """
        if len(violations) == 0:
            return 0.0
        return sum(violations) / len(violations)

    def aggregate_from_running_sum(
        self, violation_sum: float, active_count: int
    ) -> float:
        """Equivalent to ``aggregate(calculate_batch(queries))`` but computed
        from a precomputed running sum and query count, avoiding re-iteration.
        """
        if active_count == 0:
            return 0.0
        return violation_sum / active_count

    def aggregate_batch(self, lat_and_slos: Iterable[LatencySlo]) -> float:
        """Convenience: calculate and aggregate in one step."""
        return self.aggregate(self.calculate_batch(lat_and_slos))

    def to_plot_axis_label(self) -> str:
        """Human-friendly description of the aggregated metric, for plot
        labels and such."""
        if self is SloMetric.BINARY:
            return "SLO Violation Rate"
        if self is SloMetric.ABSOLUTE_S:
            return "Mean SLO Violation Amount (s)"
        if self is SloMetric.RELATIVE:
            return "Mean Relative SLO Violation"
        if self is SloMetric.RELATIVE_UNCONSTRAINED:
            return "Mean Relative SLO Violation (Unconstrained)"
        raise ValueError(f"Unknown SloMetric: {self}")
    
    def to_column_name(self) -> str:
        """Name of the column in the summary dataframe where this metric is stored."""
        if self is SloMetric.BINARY:
            return "violation_rate"
        if self is SloMetric.ABSOLUTE_S:
            return "violation_amount_s"
        if self is SloMetric.RELATIVE:
            return "violation_relative_mean"
        if self is SloMetric.RELATIVE_UNCONSTRAINED:
            return "violation_relative_unconstrained_mean"
        raise ValueError(f"Unknown SloMetric: {self}")
    
    def to_plot_axis_scale(self) -> str:
        """Recommended scale for plotting this metric."""
        if self is SloMetric.BINARY:
            return "linear"
        if self is SloMetric.ABSOLUTE_S:
            return "log"
        if self in (SloMetric.RELATIVE, SloMetric.RELATIVE_UNCONSTRAINED):
            return "linear"
        raise ValueError(f"Unknown SloMetric: {self}")
