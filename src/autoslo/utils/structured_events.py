"""
structured_events.py
--------------------
Typed event dataclasses for the autoslo structured logging system.

Every structured log emission constructs one of the event subclasses
defined here.  ``BaseStructuredEvent.to_dict()`` serialises it to a
flat ``dict`` that the :class:`~autoslo.utils.logging.StructuredLogHandler`
can persist to Parquet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from autoslo.workload_definition.query import QueryTextId


# ---------------------------------------------------------------------------
# Timestamp utility
# ---------------------------------------------------------------------------


def wall_clock_utc() -> float:
    """
    Return the current UTC wall-clock time as epoch seconds.
    """
    return datetime.now(tz=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


@dataclass
class BaseStructuredEvent:
    """Base class for all structured log events.

    Subclasses set ``event_type`` via a class-level default.
    """

    wall_clock_s: float = field(init=False, default_factory=wall_clock_utc)
    rel_time_s: float = 0.0
    event_type: str = field(init=False)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@dataclass
class RunStartEvent(BaseStructuredEvent):
    workload_name: str = ""
    num_queries: int = 0
    routing_policy: str = ""
    closed_loop: bool = False

    def __post_init__(self) -> None:
        self.event_type = "run_start"


@dataclass
class RunFinishEvent(BaseStructuredEvent):
    workload_name: str = ""

    def __post_init__(self) -> None:
        self.event_type = "run_finish"


# ---------------------------------------------------------------------------
# Query lifecycle
# ---------------------------------------------------------------------------


@dataclass
class ArrivalEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")

    def __post_init__(self) -> None:
        self.event_type = "arrival"


@dataclass
class CompletionEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""
    latency_s: float = 0.0
    slo_s: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "completion"


@dataclass
class CompletionIgnoredEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""

    def __post_init__(self) -> None:
        self.event_type = "completion_ignored"


@dataclass
class QueryExecutionStartEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""

    def __post_init__(self) -> None:
        self.event_type = "query_execution_start"


@dataclass
class QueryExecutionFinishEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "query_execution_finish"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


@dataclass
class QueryRoutedEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "query_routed"


@dataclass
class LatencyUpdateEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""
    old_latency_s: float | None = None
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "latency_update"


@dataclass
class RoutingScoreEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""
    latency_s: float = 0.0
    marginal_slo_violation: float = 0.0
    marginal_cost: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "routing_score"


@dataclass
class RoutingDecisionEvent(BaseStructuredEvent):
    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
    cluster_name: str = ""
    latency_s: float = 0.0
    marginal_slo_violation: float = 0.0
    marginal_cost: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "routing"


# ---------------------------------------------------------------------------
# Cluster lifecycle
# ---------------------------------------------------------------------------


@dataclass
class ClusterReadyEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    num_active_clusters: int = 0

    def __post_init__(self) -> None:
        self.event_type = "cluster_ready"


@dataclass
class SpinUpRequestedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        self.event_type = "spin_up_requested"


@dataclass
class SpinUpEvent(BaseStructuredEvent):
    rpu: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        self.event_type = "spin_up"


@dataclass
class TearDownDecisionEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        self.event_type = "tear_down_decision"


@dataclass
class TearDownBlockedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        self.event_type = "tear_down_blocked"


@dataclass
class TearDownRequestedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0

    def __post_init__(self) -> None:
        self.event_type = "tear_down_requested"


@dataclass
class StatsCollectedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "stats_collected"


@dataclass
class ClusterRemovedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0

    def __post_init__(self) -> None:
        self.event_type = "cluster_removed"


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------


@dataclass
class ClusterSpinUpStartedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0

    def __post_init__(self) -> None:
        self.event_type = "cluster_spin_up_started"


@dataclass
class ClusterSpinUpCompletedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "cluster_spin_up_completed"


@dataclass
class ClusterTearDownStartedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0

    def __post_init__(self) -> None:
        self.event_type = "cluster_tear_down_started"


@dataclass
class ClusterTearDownCompletedEvent(BaseStructuredEvent):
    cluster_name: str = ""
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "cluster_tear_down_completed"


# ---------------------------------------------------------------------------
# Capacity checkpoint
# ---------------------------------------------------------------------------


@dataclass
class CapacityCheckpointReconciliationEvent(BaseStructuredEvent):
    detail: str = ""

    def __post_init__(self) -> None:
        self.event_type = "capacity_checkpoint_reconciliation"


# ---------------------------------------------------------------------------
# Autoscaler
# ---------------------------------------------------------------------------


@dataclass
class RpuCounterfactualEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    slo_metric_and_cost: dict[str, float] = field(default_factory=dict)
    slo_threshold: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "rpu_counterfactual"


@dataclass
class RpuSelectionEvent(BaseStructuredEvent):
    cluster_name: str = ""
    rpu: int = 0
    slo_metric_and_cost: dict[str, float] = field(default_factory=dict)
    slo_threshold: float = 0.0

    def __post_init__(self) -> None:
        self.event_type = "rpu_selection"


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResultEvent(BaseStructuredEvent):
    phase: str = ""
    grid_point: int = 0
    workload_idx: int = 0
    violation_rate: float = 0.0
    violation_amount_s: float = 0.0
    violation_relative_mean: float = 0.0
    total_cost: float = 0.0
    num_queries: int = 0

    def __post_init__(self) -> None:
        self.event_type = "scenario_result"
