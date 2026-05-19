from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TypeAlias

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


class ClusterAwareQueryId(str):
    """
    A query identifier that also carries information about the cluster the query
    was assigned to.
    """

    @classmethod
    def make(cls, cluster_name: str, query_id: str) -> "ClusterAwareQueryId":
        """
        Construct a ClusterAwareQueryId from a cluster name and a query ID.
        """
        return cls(f"{cluster_name}#{query_id}")

    @classmethod
    def for_query(
        cls, cluster_name: str, query: "Query"
    ) -> "ClusterAwareQueryId":
        """
        Convenience: construct a ClusterAwareQueryId for a single Query.
        """
        return cls.make(cluster_name, query.query_id)

    @classmethod
    def for_queries(
        cls, cluster_name: str, queries: "list[Query]"
    ) -> "list[ClusterAwareQueryId]":
        """
        Convenience: construct a list of ClusterAwareQueryIds for a list of
        Queries.
        """
        return [cls.make(cluster_name, q.query_id) for q in queries]

    @property
    def cluster_name(self) -> str:
        """The cluster name embedded in this identifier."""
        return self.split("#")[0]

    @property
    def query_id(self) -> str:
        """The workload-level query ID (e.g. ``"query_42"``)."""
        return self.split("#")[1]


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

    def copy_with_new_info(
        self, new_query_id_prefix: str, new_rel_start_time_s: float
    ) -> "Query":
        """Return a copy with a prefixed query_id and shifted arrival time.

        Used by the forward-looking counterfactual replay to produce
        per-copy instances of window queries with non-colliding ids.
        Both ``rel_start_time_s`` and ``abs_start_time`` are updated
        consistently.
        """
        return replace(
            self,
            query_id=f"{new_query_id_prefix}{self.query_id}",
            rel_start_time_s=new_rel_start_time_s,
            abs_start_time=datetime.fromtimestamp(
                new_rel_start_time_s, tz=timezone.utc
            ),
        )
