from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

from autoslo.utils.billing import BillingInterval

QueryFeaturization: TypeAlias = list[float]


class QueryTextId(str):
    """Opaque identifier for a query's text.

    Format: schema#template#index
    Example: "ext_tpcds1000#42#001"
    """

    @property
    def schema_name(self) -> str:
        """Extracts the schema name from the query text ID."""
        return self.split("#")[0]

    @property
    def template_id(self) -> str:
        """Extracts the template ID from the query text ID."""
        return self.split("#")[1]

    @property
    def query_index(self) -> str:
        """Extracts the query index from the query text ID."""
        return self.split("#")[2]


@dataclass(frozen=True, eq=False)
class Query:
    """Immutable identity + precomputed features for a single query.

    Execution-context fields (cluster placement, predicted/actual latency,
    censored-loss flag) live outside this class — in ``RoutingResult``, in the
    simulator's latency tracker, or as explicit parameters to the dataset
    builder.
    """

    query_id: str
    query_text_id: QueryTextId

    featurization: QueryFeaturization = field(default_factory=list)

    abs_start_time: datetime = field(
        default_factory=lambda: datetime.fromtimestamp(-1, tz=timezone.utc)
    )
    rel_start_time_s: float = -1

    repetition_id: str = ""
    """Identifies repeated instances of the same query text within a workload."""

    stage_predictions_per_rpu: dict[int, float] = field(default_factory=dict)
    """Pre-computed stage-model predictions keyed by RPU size.

    Populated once at query construction time (routing or trace replay).
    The dataset builder / featuriser picks the relevant entry using the
    cluster's RPU.
    """

    def __hash__(self) -> int:
        return hash(self.query_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Query):
            return NotImplemented
        return self.query_id == other.query_id

    def __post_init__(self) -> None:
        if self.abs_start_time.timestamp() < 0 and self.rel_start_time_s < 0:
            raise ValueError(
                "At least one form of start time must be provided."
            )
        # frozen=True forbids direct assignment; use object.__setattr__
        if self.abs_start_time.timestamp() < 0:
            object.__setattr__(
                self,
                "abs_start_time",
                datetime.fromtimestamp(
                    float(self.rel_start_time_s), tz=timezone.utc
                ),
            )
        elif self.rel_start_time_s < 0:
            object.__setattr__(
                self,
                "rel_start_time_s",
                self.abs_start_time.timestamp(),
            )

    @staticmethod
    def query_interval(
        rel_start_time_s: float, latency_s: float
    ) -> BillingInterval:
        """Build an execution interval for a query."""

        return BillingInterval(rel_start_time_s, rel_start_time_s + latency_s)
