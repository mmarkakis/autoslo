from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Optional

from autoslo.blueprints.cluster_conn_info import ClusterConnInfo


@dataclass(frozen=True, eq=True)
class Cluster:
    """Lightweight descriptor for a compute cluster.

    After Phase-2 cleanup the class is a frozen dataclass — no mutable
    state, no config-file access, no connection-pool management.
    Connection pools are owned by :class:`ManagedClusterPool`.
    """

    # --- Class-level constants -------------------------------------------
    US_EAST_1_COST_PER_RPU_HOUR: ClassVar[float] = 0.375
    ONE_HOUR_S: ClassVar[int] = 3600
    UP_TO_32_RPU_SIZES: ClassVar[list[int]] = [4, 8, 16, 32]
    ALL_ALLOWED_RPU_SIZES: ClassVar[list[int]] = UP_TO_32_RPU_SIZES
    DEFAULT_SPIN_UP_DELAY_S: ClassVar[int] = 300

    _new_counter: ClassVar[itertools.count] = itertools.count()

    # --- Instance fields -------------------------------------------------
    rpu: int
    name: str
    conn_info: Optional[ClusterConnInfo] = field(default=None, compare=True)
    cost_per_rpu_hour: float = field(
        default=US_EAST_1_COST_PER_RPU_HOUR, compare=True
    )

    # --- Derived properties ----------------------------------------------

    @property
    def cost_per_second(self) -> float:
        """Cost per second for the cluster."""
        return self.cost_per_rpu_hour * self.rpu / Cluster.ONE_HOUR_S

    # --- Static helpers --------------------------------------------------

    @staticmethod
    def all_allowed_rpu_sizes() -> list[int]:
        return Cluster.ALL_ALLOWED_RPU_SIZES

    @staticmethod
    def rpu_for_cluster_name(cluster_name: str) -> int:
        """Parse RPU from a cluster name.

        Supports the dynamic naming convention
        ``"cluster_{rpu}_{timestamp}_{counter}"`` as well as static
        config names of the form ``"cluster_{rpu}_..."``.
        """
        parts = cluster_name.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        raise ValueError(
            f"Cannot parse RPU from cluster name: {cluster_name!r}"
        )

    @staticmethod
    def cost_per_second_for_rpu(
        rpu: int,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
    ) -> float:
        """Return the cost-per-second for the given RPU size."""
        return cost_per_rpu_hour * rpu / 3600

    @staticmethod
    def new(
        rpu: int,
        name: str | None = None,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
    ) -> Cluster:
        """Create a spec-only cluster with no config lookup.

        Factory for dynamically provisioned clusters (used by the
        simulator and the capacity controller).  Connection info is
        *not* attached — the :class:`ManagedClusterPool` wires it up
        after the AWS workgroup is created.
        """
        if name is None:
            seq = next(Cluster._new_counter)
            name = f"cluster_{rpu}_{int(datetime.now().timestamp())}_{seq}"
        return Cluster(
            rpu=rpu,
            name=name,
            conn_info=None,
            cost_per_rpu_hour=cost_per_rpu_hour,
        )
