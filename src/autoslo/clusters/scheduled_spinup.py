import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from autoslo.clusters.actions import SpinUpAction
from autoslo.filesystem.logging import emit_structured
from autoslo.filesystem.structured_events import BaseStructuredEvent, EventType


@dataclass(frozen=True)
class ScheduledSpinUp:
    """Imperative scheduled spin-up.

    At ``rel_time_s`` (relative to workload start) the system
    unconditionally spins up one cluster of size ``rpu``, regardless of
    what the pool currently contains.  There is no reconciliation with
    existing state — just a spin-up order.

    Parameters
    ----------
    rel_time_s :
        Trigger time in seconds from the start of the workload.
    rpu :
        RPU size of the cluster to spin up.
    """

    rel_time_s: float
    rpu: int

    @staticmethod
    def from_config(cfg: dict) -> list["ScheduledSpinUp"]:
        return [
            ScheduledSpinUp(**su)
            for su in cfg.get("scheduled_spinups", [])
        ]

    def execute(
        self,
        source: str,
        on_spin_up: Callable[[SpinUpAction], None],
        write_text_log: bool = False,
        rel_time_s_getter: Callable[[], float] = lambda: 0.0,
    ) -> None:
        """Unconditionally spin up one cluster of ``rpu`` size."""
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s_getter(),
                event_type=EventType.SCHEDULED_SPINUP_EXECUTED,
                source=source,
                details={
                    "scheduled_rel_time_s": f"{self.rel_time_s}",
                    "rpu": str(self.rpu),
                },
            )
        )

        if write_text_log:
            logging.debug(
                "ScheduledSpinUp t=%.1f — spinning up 1 cluster of %d RPU",
                self.rel_time_s,
                self.rpu,
            )

        action = SpinUpAction(
            rpu=self.rpu,
            from_reserved_budget=True,
            reason=f"scheduled_spinup@t={self.rel_time_s}",
        )
        on_spin_up(action)

    @staticmethod
    def total_spinups(spinups: Iterable["ScheduledSpinUp"]) -> int:
        """Total number of spin-ups (one per entry)."""
        return sum(1 for _ in spinups)
