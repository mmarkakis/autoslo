from dataclasses import dataclass
import autoslo.utils.config as cfgu

from autoslo.clusters.actions import SpinUpAction
from collections import Counter


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

    @staticmethod
    def parse_from_cfg(cfg: dict) -> list["CapacityCheckpoint"]:
        raw: list[dict] = cfgu.getd(cfg, "capacity_checkpoints", [])
        return [
            CapacityCheckpoint(
                rel_time_s=float(cp["rel_time_s"]),
                min_rpus=tuple(cp["min_rpus"]),
            )
            for cp in raw
        ]

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
