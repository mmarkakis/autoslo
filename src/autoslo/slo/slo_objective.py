from enum import Enum
from dataclasses import dataclass


class SloMetric(Enum):
    """Which SLO-violation metric to use for routing decisions.

    * ``BINARY``     – 1 if the query violates its SLO, else 0.
    * ``ABSOLUTE_S`` – seconds of overshoot, ``max(0, latency − SLO)``.
    * ``RELATIVE``   – relative overshoot, ``max(0, (latency − SLO) / SLO)``.

    All three are always *reported*; this enum selects which one drives
    the routing optimiser.
    """

    BINARY = "binary"
    ABSOLUTE_S = "absolute_s"
    RELATIVE = "relative"


@dataclass(frozen=True)
class SloObjective:
    """Bundles the SLO metric name and feasibility threshold."""

    slo_metric: SloMetric  # "binary", "absolute_s", or "relative"
    slo_threshold: float
    # Meaning of the threshold depends on the metric:
    # - "binary": max allowed violation rate (e.g. 0.01 for 1%)
    # - "absolute_s": max allowed total violation amount in seconds (e.g. 10.0)
    # - "relative": max allowed mean relative violation (e.g. 0.1 for 10%)

    def is_met(self, per_query_latency_slo: list[tuple[float, float]]) -> bool:
        """Return True if the given per-query (latency, SLO) pairs meet the
        SLO objective."""
        if len(per_query_latency_slo) == 0:
            return False
        if self.slo_metric == SloMetric.BINARY:
            return (
                sum(lat > slo for lat, slo in per_query_latency_slo)
                / len(per_query_latency_slo)
            ) <= self.slo_threshold
        if self.slo_metric == SloMetric.ABSOLUTE_S:
            return (
                sum(max(0.0, lat - slo) for lat, slo in per_query_latency_slo)
                <= self.slo_threshold
            )
        if self.slo_metric == SloMetric.RELATIVE:
            return (
                sum(
                    max(0.0, (lat - slo) / slo)
                    for lat, slo in per_query_latency_slo
                )
                / len(per_query_latency_slo)
            ) <= self.slo_threshold
        raise ValueError(f"Unknown slo_metric: {self.slo_metric!r}")
