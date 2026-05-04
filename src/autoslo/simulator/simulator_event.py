from dataclasses import dataclass
from typing import Any


@dataclass
class SimulatorEventType:
    SCHEDULED_SPINUP: str = "scheduled_spinup"
    CLUSTER_READY: str = "cluster_ready"
    QUERY_ARRIVAL: str = "query_arrival"
    QUERY_COMPLETION: str = "query_completion"


@dataclass
class SimulatorEvent:
    rel_time_s: float
    event_type: str
    details: dict[str, Any]

    def __lt__(self, other: "SimulatorEvent") -> bool:
        """Order by rel_time_s, then by event_type for tie-breaking."""
        if self.rel_time_s == other.rel_time_s:
            return self.event_type < other.event_type
        return self.rel_time_s < other.rel_time_s
