"""
structured_events.py
--------------------
Typed event dataclasses for the autoslo structured logging system.

Every structured log emission constructs a :class:`BaseStructuredEvent`
(or :class:`QueryRelatedEvent` for query-scoped events), passing in an
:class:`EventType` enum member.  ``BaseStructuredEvent.to_dict()``
serialises the event to a flat ``dict`` that the
:class:`~autoslo.utils.logging.StructuredLogHandler` can persist to
Parquet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
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
# Event type enum
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Enumeration of all structured event types."""

    # Run lifecycle
    RUN_START = "run_start"
    RUN_FINISH = "run_finish"

    # Query lifecycle
    ARRIVAL = "arrival"
    QUERY_EXECUTION_START = "query_execution_start"
    QUERY_EXECUTION_FINISH = "query_execution_finish"
    COMPLETION = "completion"

    # Routing
    QUERY_ROUTED = "query_routed"
    LATENCY_UPDATE = "latency_update"
    ROUTING_SCORE = "routing_score"
    ROUTING = "routing"

    # Cluster lifecycle
    SPIN_UP_DECISION = "spin_up_decision"
    SPIN_UP_REQUESTED = "spin_up_requested"
    SPIN_UP_STARTED = "spin_up_started"
    SPIN_UP_BLOCKED = "spin_up_blocked"
    CLUSTER_READY = "cluster_ready"

    TEAR_DOWN_DECISION = "tear_down_decision"
    TEAR_DOWN_REQUESTED = "tear_down_requested"
    TEAR_DOWN_BLOCKED = "tear_down_blocked"
    TEAR_DOWN_STARTED = "tear_down_started"
    STATS_COLLECTED = "stats_collected"
    CLUSTER_REMOVED = "cluster_removed"

    # Capacity checkpoint
    CAPACITY_CHECKPOINT_RECONCILIATION = "capacity_checkpoint_reconciliation"

    # Autoscaler
    RPU_COUNTERFACTUAL = "rpu_counterfactual"
    RPU_SELECTION = "rpu_selection"

    # ------------------------------------------------------------------
    # Grouped subsets
    # ------------------------------------------------------------------

    @classmethod
    def query_lifecycle_types(cls) -> set[EventType]:
        """Events that track a query from arrival to completion."""
        return {
            cls.ARRIVAL,
            cls.QUERY_EXECUTION_START,
            cls.QUERY_EXECUTION_FINISH,
            cls.COMPLETION,
        }

    @classmethod
    def routing_types(cls) -> set[EventType]:
        """Events emitted during or about query routing."""
        return {
            cls.QUERY_ROUTED,
            cls.LATENCY_UPDATE,
            cls.ROUTING_SCORE,
            cls.ROUTING,
        }

    @classmethod
    def cluster_lifecycle_types(cls) -> set[EventType]:
        """Events that track cluster spin-up, readiness, and tear-down."""
        return {
            cls.SPIN_UP_DECISION,
            cls.SPIN_UP_REQUESTED,
            cls.SPIN_UP_STARTED,
            cls.SPIN_UP_BLOCKED,
            cls.CLUSTER_READY,
            cls.TEAR_DOWN_DECISION,
            cls.TEAR_DOWN_REQUESTED,
            cls.TEAR_DOWN_BLOCKED,
            cls.TEAR_DOWN_STARTED,
            cls.STATS_COLLECTED,
            cls.CLUSTER_REMOVED,
        }

    @classmethod
    def autoscaler_types(cls) -> set[EventType]:
        """Events related to autoscaler RPU decisions."""
        return {
            cls.RPU_COUNTERFACTUAL,
            cls.RPU_SELECTION,
        }


# Required details per event type.  Types not listed here have no requirements.
REQUIRED_DETAILS: dict[EventType, list[str]] = {
    EventType.RUN_START: [
        "workload_name",
        "num_queries",
        "routing_policy",
        "closed_loop",
    ],
    EventType.RUN_FINISH: ["workload_name"],
    EventType.COMPLETION: ["success"],
    EventType.LATENCY_UPDATE: ["old_latency_s", "latency_s"],
    EventType.ROUTING: ["slo_violation", "cost"],
    EventType.SPIN_UP_DECISION: ["rpu", "reason"],
    EventType.SPIN_UP_REQUESTED: ["reason"],
    EventType.SPIN_UP_BLOCKED: [
        "reason",
        "max",
        "used",
        "reserved",
        "available",
    ],
    EventType.TEAR_DOWN_DECISION: ["reason"],
    EventType.TEAR_DOWN_REQUESTED: ["reason", "force"],
    EventType.CAPACITY_CHECKPOINT_RECONCILIATION: [
        "checkpoint_rel_time_s",
        "desired",
        "current",
        "spin_ups_needed",
    ],
    EventType.RPU_COUNTERFACTUAL: ["slo_violation", "cost"],
    EventType.RPU_SELECTION: ["slo_violation", "cost"],
}


# ---------------------------------------------------------------------------
# Concrete event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BaseStructuredEvent:
    """Structured log event.

    Use this class directly for events that are not query-related.
    For query-related events, use :class:`QueryRelatedEvent`.
    """

    rel_time_s: float
    event_type: EventType
    source: str
    cluster_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    wall_clock_s: float = field(init=False, default_factory=wall_clock_utc)

    def __post_init__(self) -> None:
        required = REQUIRED_DETAILS.get(self.event_type, [])
        for key in required:
            if key not in self.details:
                raise ValueError(
                    f"Missing required detail '{key}' in {self.event_type} event."
                )

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["event_type"] = self.event_type.value
        d["details"] = (
            json.dumps(d["details"], default=str) if d["details"] else ""
        )
        return d


@dataclass
class QueryRelatedEvent(BaseStructuredEvent):
    """Structured log event related to a specific query."""

    query_id: str = ""
    query_text_id: QueryTextId = QueryTextId("")
