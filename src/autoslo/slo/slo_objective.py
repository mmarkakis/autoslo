from dataclasses import dataclass
from enum import Enum

from autoslo.slo.slo_metric import SloMetric


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

    def is_met(self, per_query_latency_slo: list[tuple[float, float]]) -> bool:
        """Return True if the given per-query (latency, SLO) pairs meet the
        SLO objective."""
        return (
            self.slo_metric.aggregate_batch(per_query_latency_slo)
            <= self.slo_threshold
        )
