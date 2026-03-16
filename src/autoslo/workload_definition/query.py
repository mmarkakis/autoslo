from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TypeAlias


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


QueryFeaturization: TypeAlias = list[float]


@dataclass(frozen=True)
class QueryTextId:
    """Opaque identifier for a query's text. Includes 3 pound-separated parts:
    - The schema name (e.g. "ext_tpcds1000")
    - The template ID (e.g. "42")
    - The query index within the template (e.g. "001")
    For example, "ext_tpcds1000#42#001".
    """

    value: str

    def __str__(self):
        return self.value

    @property
    def schema_name(self) -> str:
        """Extracts the schema name from the query text ID."""
        return self.value.split("#")[0]

    @property
    def template_id(self) -> str:
        """Extracts the template ID from the query text ID."""
        return self.value.split("#")[1]

    @property
    def query_index(self) -> str:
        """Extracts the query index from the query text ID."""
        return self.value.split("#")[2]


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
    """Opaque key that identifies the query text within its schema.

    The actual SQL can be retrieved via
    ``QueryTextRegistry.get(schema_name, query_text_id)``.
    """

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

    def stage_prediction_for_cluster(self, cluster_name: str) -> float:
        """Convenience: look up the stage prediction for a named cluster."""
        from autoslo.workload_definition.cluster import Cluster

        rpu = Cluster.rpu_for_cluster_name(cluster_name)
        return self.stage_predictions_per_rpu.get(rpu, -1.0)
