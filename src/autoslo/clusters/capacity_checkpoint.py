import logging
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import autoslo.utils.config as cfgu
from autoslo.clusters.actions import SpinUpAction
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.utils.logging import LOGGER_NAME, emit_structured

_has_structured = lambda: bool(logging.getLogger(LOGGER_NAME).handlers)


@dataclass(frozen=True)
class CapacityCheckpoint:
    """Declarative capacity checkpoint.

    At ``rel_time_s`` (relative to workload start) the system reconciles
    the declared RPU multiset against the current (READY + PENDING)
    clusters and spins up only the gap.

    Parameters
    ----------
    rel_time_s :
        Trigger time in seconds from the start of the workload.
    min_rpus :
        Desired RPU multiset.  Each element is an RPU size that must
        be present (exact matching, not total-capacity).
    """

    rel_time_s: float
    min_rpus: tuple[int, ...]

    def spin_ups_needed(
        self, current_counts_per_rpu: Counter[int]
    ) -> list[SpinUpAction]:
        """
        Return the spin-up actions needed to reach the declared RPU multiset
        from the given multiset of RPUs.
        """
        desired = Counter(self.min_rpus)
        gap = desired - current_counts_per_rpu

        actions = []
        for rpu, count in sorted(gap.items()):
            for _ in range(count):
                action = SpinUpAction(
                    rpu=rpu,
                    reason=f"capacity_checkpoint@t={self.rel_time_s}",
                )
                actions.append(action)

        return actions

    def reconcile(
        self,
        pool: ManagedClusterPool,
        current_time_s: float,
        source: str,
        on_spin_up: Callable[[SpinUpAction], None],
        write_text_log: bool = False,
    ) -> None:
        """Check against current capacity and trigger spin-ups."""
        current_counts = pool.ready_and_pending_counts_per_rpu()
        spin_ups_needed = self.spin_ups_needed(current_counts)

        if _has_structured():
            emit_structured(
                {
                    "timestamp": current_time_s,
                    "event_type": "capacity_checkpoint_reconciliation",
                    "source": source,
                    "checkpoint_rel_time_s": self.rel_time_s,
                    "desired_rpus": ",".join(
                        f"{rpu}:{count}"
                        for rpu, count in Counter(self.min_rpus).items()
                    ),
                    "current_rpus": ",".join(
                        f"{rpu}:{count}"
                        for rpu, count in current_counts.items()
                    ),
                }
            )

        if not spin_ups_needed:
            if write_text_log:
                logging.debug(
                    "Checkpoint t=%.1f: already satisfied (current %s).",
                    self.rel_time_s,
                    dict(current_counts),
                )
            return

        if write_text_log:
            logging.debug(
                "Checkpoint t=%.1f — spinning up %d clusters",
                self.rel_time_s,
                len(spin_ups_needed),
            )
        for action in spin_ups_needed:
            on_spin_up(action)
            if _has_structured():
                emit_structured(
                    {
                        "timestamp": current_time_s,
                        "event_type": "spin_up",
                        "source": source,
                        "rpu": action.rpu,
                        "reason": f"capacity_checkpoint@t={self.rel_time_s}",
                    }
                )
