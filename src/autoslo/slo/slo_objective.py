from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from autoslo.slo.slo_metric import LatencySlo, SloMetric


class ViolationCost(NamedTuple):
    """Aggregated SLO-violation metric paired with execution cost.

    This is the output type of the SLO evaluation pipeline and the
    input type for :meth:`SloObjective.is_b_better` /
    :meth:`SloObjective.idx_of_best`.
    """

    violation: float
    cost: float


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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SloObjective:
        try:
            slo_metric = SloMetric(config["slo_config"]["slo_metric"])
            slo_threshold = config["slo_config"]["slo_threshold"]
            return cls(slo_metric, slo_threshold)
        except KeyError as e:
            raise ValueError(
                "Invalid SLO config: missing key "
                f"{e.args[0]} in {config['slo_config']}"
            ) from e

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
        a: ViolationCost,
        b: ViolationCost,
        tolerance: float = COMPARISON_TOLERANCE,
    ) -> bool:
        """
        Compare two (metric_value, cost) pairs according to the SLO objective.

        If both are under threshold, return the cheapest. Otherwise, return the
        one with the better metric value.
        """

        a_meets = self.is_met_from_aggregated(a.violation)
        b_meets = self.is_met_from_aggregated(b.violation)

        if a_meets and b_meets:
            return self._cmp_with_tolerance(a.cost, b.cost, tolerance) > 0
        return self._cmp_with_tolerance(a.violation, b.violation, tolerance) > 0

    def idx_of_best(self, candidates: list[ViolationCost]) -> int:
        """Return the index of the best ``(violation, cost)`` candidate.

        Applies the same lexicographic policy as :meth:`is_b_better`:
        feasible candidates (metric ≤ threshold) are preferred and ranked
        by cost; if none are feasible, the one with the lowest violation
        (tiebreak: cost) wins.
        """
        best = 0
        for i in range(1, len(candidates)):
            if self.is_b_better(candidates[best], candidates[i]):
                best = i
        return best
