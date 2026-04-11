from typing import Optional

import autoslo.utils.config as cfgu
from autoslo.clusters.autoscaler import (
    CapacityCheckpoint,
)
from autoslo.clusters.managed_cluster_pool import ManagedClusterPoolConfig


# ── policy builders ──────────────────────────────────────────────────────


def build_managed_cluster_pool_config(
    cfg: dict,
) -> Optional[ManagedClusterPoolConfig]:
    """Construct a :class:`ManagedClusterPoolConfig` from the config dict."""
    mcp_raw: Optional[dict] = cfgu.getd(cfg, "managed_cluster_pool_config")
    if mcp_raw is None:
        return None
    mcp_raw = dict(mcp_raw)  # shallow copy — don't mutate the caller's dict
    if "initial_rpus" in mcp_raw and isinstance(mcp_raw["initial_rpus"], list):
        mcp_raw["initial_rpus"] = tuple(mcp_raw["initial_rpus"])
    if "allowed_rpu_sizes" in mcp_raw and isinstance(
        mcp_raw["allowed_rpu_sizes"], list
    ):
        mcp_raw["allowed_rpu_sizes"] = tuple(mcp_raw["allowed_rpu_sizes"])
    return ManagedClusterPoolConfig(**mcp_raw)


def parse_capacity_checkpoints(cfg: dict) -> list[CapacityCheckpoint]:
    """Parse ``autoscaling_config.capacity_checkpoints`` into a typed list."""
    raw: list[dict] = cfgu.getd(
        cfg, "autoscaling_config.capacity_checkpoints", []
    )
    return [
        CapacityCheckpoint(
            rel_time_s=float(cp["rel_time_s"]),
            min_rpus=tuple(cp["min_rpus"]),
        )
        for cp in raw
    ]



