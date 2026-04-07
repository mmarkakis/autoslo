from typing import Optional

import autoslo.utils.config as cfgu
from autoslo.capacity.autoscaling_policy import (
    AutoscalingPolicy,
    CapacityCheckpoint,
    NoOpPolicy,
)
from autoslo.capacity.headroom_policy import HeadroomPolicy
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.cache_aware_policy import CacheAwarePolicy
from autoslo.routing.managed_cluster_pool import ManagedClusterPoolConfig
from autoslo.routing.model_policy import ModelPolicy
from autoslo.routing.routing_policy import RoundRobinPolicy, RoutingPolicy
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver

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


def build_routing_policy(
    cfg: dict,
    iconq_model_id: Optional[str],
    slo_s: float,
    slo_resolver: SloResolver,
    slo_metric: SloMetric,
    iconq_model: Optional["IconqModel"] = None,
) -> RoutingPolicy:
    """Construct a :class:`RoutingPolicy` from the ``routing_config`` section."""
    policy_type: str = cfgu.getd(cfg, "routing_config.routing_policy", "model")

    if policy_type == "model":
        if iconq_model_id is None:
            raise ValueError(
                "iconq_model_id is required for 'model' routing_policy"
            )
        return ModelPolicy(
            iconq_model_id=iconq_model_id,
            default_slo_s=slo_s,
            slo_overrides=slo_resolver.slo_dict,
            slo_metric=slo_metric,
            iconq_model=iconq_model,
        )
    elif policy_type == "round_robin":
        return RoundRobinPolicy()
    elif policy_type == "cache_aware":
        if iconq_model_id is None:
            raise ValueError(
                "iconq_model_id is required for 'cache_aware' routing_policy"
            )
        return CacheAwarePolicy(
            iconq_model_id=iconq_model_id,
            default_slo_s=slo_s,
            slo_overrides=slo_resolver.slo_dict,
            slo_metric=slo_metric,
            forecast_distribution_path=cfgu.getd(
                cfg, "routing_config.forecast_distribution_path"
            ),
            slo_tightness_path=cfgu.getd(
                cfg, "routing_config.slo_tightness_path"
            ),
            cache_risk_lambda=float(
                cfgu.getd(cfg, "routing_config.cache_risk_lambda", 0.0)
            ),
            cache_decay_strategy=cfgu.getd(
                cfg, "routing_config.cache_decay_strategy", "exponential"
            ),
            cache_decay_params=cfgu.getd(
                cfg, "routing_config.cache_decay_params", {}
            ),
            fallback_tightness=float(
                cfgu.getd(cfg, "routing_config.fallback_tightness", 0.5)
            ),
            iconq_model=iconq_model,
        )
    else:
        raise ValueError(
            f"Unknown routing_policy {policy_type!r}. "
            "Expected one of: 'model', 'round_robin', 'cache_aware'."
        )


def build_autoscaling_policy(
    cfg: dict,
    slo_resolver: SloResolver,
    slo_objective: SloObjective,
    iconq_model_id: Optional[str],
    routing_policy: RoutingPolicy,
    allowed_rpu_sizes: list[int],
    iconq_model: Optional["IconqModel"] = None,
) -> AutoscalingPolicy:
    """Construct an :class:`AutoscalingPolicy` from ``autoscaling_config``."""
    policy_type: str = cfgu.getd(
        cfg, "autoscaling_config.autoscaling_policy", "headroom"
    )

    if policy_type == "headroom":
        eta_crit: float = float(
            cfgu.getd(cfg, "autoscaling_config.eta_crit", 0.1)
        )
        idle_periods_before_tear_down: int = int(
            cfgu.getd(
                cfg,
                "autoscaling_config.idle_periods_before_tear_down",
                5,
            )
        )
        min_cluster_lifetime_s: float = float(
            cfgu.getd(
                cfg,
                "autoscaling_config.min_cluster_lifetime_s",
                1200.0,
            )
        )
        if iconq_model is None:
            iconq_model = (
                IconqModel.load(iconq_model_id) if iconq_model_id else None
            )

        return HeadroomPolicy(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            eta_crit=eta_crit,
            idle_periods_before_tear_down=idle_periods_before_tear_down,
            min_cluster_lifetime_s=min_cluster_lifetime_s,
            allowed_rpu_sizes=allowed_rpu_sizes,
            iconq_model=iconq_model,
            routing_policy=routing_policy,
        )
    elif policy_type == "noop":
        return NoOpPolicy()
    else:
        raise ValueError(
            f"Unknown autoscaling_policy {policy_type!r}. "
            "Expected one of: 'headroom', 'noop'."
        )
