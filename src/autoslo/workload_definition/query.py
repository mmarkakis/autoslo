from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

from intervaltree import Interval  # type: ignore[import]

QueryFeaturization: TypeAlias = list[float]

# Generic identifier for what query text a query instance is using.
# Historically this encoded a TPC-DS template + index (e.g. "42_001");
# the field is now called query_text_id so it can accommodate any schema.
QueryTextId: TypeAlias = str



@dataclass
class Query:
    """Class representing a single query in a workload.

    A ``Query`` corresponds to one row in the ``workload`` data schema: it
    carries the query's identity fields (``query_id``, ``query_text_id``,
    ``schema_name``, ``repetition_id``) as well as optional execution-time
    fields populated after a run (latency, featurization, …).
    """

    query_id: str
    query_text_id: QueryTextId
    """Opaque key that identifies the query text within its schema.

    The actual SQL can be retrieved via
    ``QueryTextRegistry.get(schema_name, query_text_id)``.
    """

    featurization: QueryFeaturization = field(default_factory=list)

    abs_start_time: datetime = datetime.fromtimestamp(-1, tz=timezone.utc)
    rel_start_time_s: float = -1

    schema_name: str = ""
    """The database schema this query targets (e.g. ``"ext_tpcds1000"``)."""

    repetition_id: str = ""
    """Identifies repeated instances of the same query text within a workload."""

    cluster_name: str = ""
    stage_latency_prediction_s: float = -1

    latency_s: float = -1
    latency_is_lower_bound: bool = False

    slo_s: float | None = None
    """Per-query SLO in seconds.

    When set, the query carries its own SLO so callers do not need a separate
    ``SloResolver``.  ``None`` means "use the resolver / external SLO".
    """

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
            data={"query_id": self.query_id},
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
