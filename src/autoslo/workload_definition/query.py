from dataclasses import dataclass, field

from intervaltree import Interval  # type: ignore[import]

QueryFeaturization = list[float]


@dataclass
class Query:
    """Class representing a single query in the workload."""

    query_id: str
    start_time_s: float
    tpcds_temp_and_q_idx: str

    featurization: QueryFeaturization = field(default_factory=list)

    cluster_name: str = ""
    stage_latency_prediction_s: float = -1

    latency_s: float = -1
    latency_is_lower_bound: bool = False

    @property
    def state(self) -> str:
        """
        Returns the state of the query as a string.
        """
        error_state = False

        if len(self.featurization) == 0:
            error_state |= len(self.cluster_name) > 0
            error_state |= self.stage_latency_prediction_s >= 0
            error_state |= self.latency_s >= 0
            error_state |= self.latency_is_lower_bound
            return "BARE" if not error_state else "ERROR"

        if len(self.cluster_name) == 0:
            error_state |= self.stage_latency_prediction_s >= 0
            error_state |= self.latency_s >= 0
            error_state |= self.latency_is_lower_bound
            return "FEATURIZED" if not error_state else "ERROR"
        error_state |= self.stage_latency_prediction_s < 0

        if self.latency_s < 0:
            error_state |= self.latency_is_lower_bound
            return "ROUTED" if not error_state else "ERROR"

        return "COMPLETED" if not error_state else "ERROR"

    def as_interval(self) -> Interval:
        """Returns the execution interval of the query as an Interval object."""
        return Interval(
            begin=self.start_time_s,
            end=self.start_time_s + self.latency_s,
            data=self.__dict__,
        )

    def slo_deviation_amount_s(self, slo_s: float) -> float:
        """Returns the amount of deviation from the SLO in seconds.

        This is positive if the query violates the SLO,
        negative if it has SLO slack, and 0 if it meets the SLO exactly.
        """
        return self.latency_s - slo_s

    def violates_slo(self, slo_s: float) -> bool:
        """Returns whether the query violates the SLO."""
        return self.slo_deviation_amount_s(slo_s) > 0

    def slo_violation_amount_s(self, slo_s: float) -> float:
        """Returns the amount of SLO violation in seconds."""
        return max(0.0, self.slo_deviation_amount_s(slo_s))

    def has_slo_slack(self, slo_s: float) -> bool:
        """Returns whether the query has any SLO slack
        (i.e. does not violate the SLO)."""
        return self.slo_deviation_amount_s(slo_s) < 0

    def slo_slack_amount_s(self, slo_s: float) -> float:
        """Returns the amount of SLO slack in seconds."""
        return max(0.0, -self.slo_deviation_amount_s(slo_s))
