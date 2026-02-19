from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

from intervaltree import Interval  # type: ignore[import]

QueryFeaturization: TypeAlias = list[float]
TPCDSTempAndQIdx: TypeAlias = str


@dataclass
class Query:
    """Class representing a single query in the workload."""

    query_id: str
    tpcds_temp_and_q_idx: TPCDSTempAndQIdx
    featurization: QueryFeaturization = field(default_factory=list)

    abs_start_time: datetime = datetime.fromtimestamp(-1, tz=timezone.utc)
    rel_start_time_s: float = -1

    cluster_name: str = ""
    stage_latency_prediction_s: float = -1

    latency_s: float = -1
    latency_is_lower_bound: bool = False

    def __hash__(self):
        return hash(self.query_id)
    
    def __eq__(self, other):
        if not isinstance(other, Query):
            return NotImplemented
        return self.query_id == other.query_id

    def __post_init__(self):
        if (self.abs_start_time.timestamp() < 0) and (
            self.rel_start_time_s < 0
        ):
            raise ValueError(
                "At least one form of start time must be provided."
            )
        elif self.abs_start_time.timestamp() < 0:
            self.abs_start_time = datetime.fromtimestamp(
                self.rel_start_time_s, tz=timezone.utc
            )
        elif self.rel_start_time_s < 0:
            self.rel_start_time_s = self.abs_start_time.timestamp()

    def as_interval(self) -> Interval:
        """Returns the execution interval of the query as an Interval object."""
        return Interval(
            begin=self.rel_start_time_s,
            end=self.rel_start_time_s + self.latency_s,
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

    @staticmethod
    def template_id(temp_and_q_idx: TPCDSTempAndQIdx) -> int:
        """
        Extract the TPC-DS template number from the given template and query
        index string.

        Parameters:
            temp_and_q_idx: The TPC-DS template and query index string.

        Returns:
            The template number as an integer.
        """
        return int(str(temp_and_q_idx).split("_")[0])

    @staticmethod
    def idx_in_template(temp_and_q_idx: TPCDSTempAndQIdx) -> int:
        """
        Extract the TPC-DS query index from the given template and query index
        string.

        Parameters:
            temp_and_q_idx: The TPC-DS template and query index string.

        Returns:
            The query index as an integer.
        """
        return int(str(temp_and_q_idx).split("_")[1])
