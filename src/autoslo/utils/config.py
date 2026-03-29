"""Shared config-parsing helpers for WorkloadSimulator and WorkloadRunner."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import yaml

from autoslo.blueprint_selection.slo_resolver import SloResolver
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
from autoslo.workload_definition.query import SloMetric

# ── pure utilities ────────────────────────────────────────────────────────


def apply_overrides(cfg: dict, overrides: dict[str, object]) -> None:
    """Apply dot-delimited key overrides to a nested config dict *in place*.

    Example::

        apply_overrides(cfg, {"slo_config.slo_s": 5.0})
        # equivalent to  cfg["slo_config"]["slo_s"] = 5.0
    """
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value


def cfg_get(
    cfg: dict,
    section_key: str,
    key: str,
    default=None,
    *,
    required: bool = False,
):
    """Read *key* from a named section, falling back to the root level."""
    section = cfg.get(section_key)
    if section and key in section:
        return section[key]
    if key in cfg:
        return cfg[key]
    if required:
        raise KeyError(
            f"Required config key '{key}' not found in section '{section_key}' or at root level"
        )
    return default


# ── policy builders ──────────────────────────────────────────────────────


def build_routing_policy(
    cfg: dict,
    iconq_model_id: Optional[str],
    slo_s: float,
    slo_resolver: SloResolver,
    slo_metric: SloMetric,
    iconq_model: Optional["IconqModel"] = None,
) -> RoutingPolicy:
    """Construct a :class:`RoutingPolicy` from the ``routing_config`` section."""
    routing_cfg: dict = cfg.get("routing_config") or {}
    policy_type: str = routing_cfg.get(
        "routing_policy", cfg.get("routing_policy", "model")
    )

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
            forecast_distribution_path=routing_cfg[
                "forecast_distribution_path"
            ],
            slo_tightness_path=routing_cfg["slo_tightness_path"],
            cache_risk_lambda=float(routing_cfg.get("cache_risk_lambda", 0.0)),
            cache_decay_strategy=routing_cfg.get(
                "cache_decay_strategy", "exponential"
            ),
            cache_decay_params=routing_cfg.get("cache_decay_params", {}),
            fallback_tightness=float(
                routing_cfg.get("fallback_tightness", 0.5)
            ),
            iconq_model=iconq_model,
        )
    else:
        raise ValueError(
            f"Unknown routing_policy {policy_type!r}. "
            "Expected one of: 'model', 'round_robin', 'cache_aware'."
        )


def build_managed_cluster_pool_config(
    cfg: dict,
) -> Optional[ManagedClusterPoolConfig]:
    """Construct a :class:`ManagedClusterPoolConfig` from the config dict."""
    mcp_raw: Optional[dict] = cfg.get("managed_cluster_pool_config")
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


def build_autoscaling_policy(
    cfg: dict,
    slo_resolver: SloResolver,
    slo_metric: SloMetric,
    slo_threshold: float,
    iconq_model_id: Optional[str],
    routing_policy: RoutingPolicy,
    allowed_rpu_sizes: list[int],
    iconq_model: Optional["IconqModel"] = None,
) -> AutoscalingPolicy:
    """Construct an :class:`AutoscalingPolicy` from ``autoscaling_config``."""
    policy_type: str = cfg_get(
        cfg, "autoscaling_config", "autoscaling_policy", "headroom"
    )

    if policy_type == "headroom":
        return HeadroomPolicy(
            slo_resolver=slo_resolver,
            slo_metric=slo_metric,
            eta_crit=float(cfg_get(cfg, "autoscaling_config", "eta_crit", 0.1)),
            idle_periods_before_tear_down=int(
                cfg_get(
                    cfg,
                    "autoscaling_config",
                    "idle_periods_before_tear_down",
                    5,
                )
            ),
            min_cluster_lifetime_s=float(
                cfg_get(
                    cfg,
                    "autoscaling_config",
                    "min_cluster_lifetime_s",
                    1200.0,
                )
            ),
            allowed_rpu_sizes=allowed_rpu_sizes,
            iconq_model=(
                iconq_model
                if iconq_model is not None
                else (IconqModel.load(iconq_model_id) if iconq_model_id else None)
            ),
            routing_policy=routing_policy,
            slo_threshold=slo_threshold,
        )
    elif policy_type == "noop":
        return NoOpPolicy()
    else:
        raise ValueError(
            f"Unknown autoscaling_policy {policy_type!r}. "
            "Expected one of: 'headroom', 'noop'."
        )


def parse_capacity_checkpoints(cfg: dict) -> list[CapacityCheckpoint]:
    """Parse ``autoscaling_config.capacity_checkpoints`` into a typed list."""
    raw: list[dict] = (
        cfg_get(cfg, "autoscaling_config", "capacity_checkpoints", []) or []
    )
    return [
        CapacityCheckpoint(
            time_s=float(cp["time_s"]),
            min_rpus=tuple(cp["min_rpus"]),
        )
        for cp in raw
    ]


def load_config_from_cli(description: str) -> tuple[dict, str]:
    """Parse CLI args, load YAML config, apply ``--set`` overrides.

    Returns ``(cfg, config_path)`` where *cfg* is the fully-resolved
    config dict and *config_path* is the path to the YAML file.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "config",
        help="Path to the YAML config file (e.g. data/__run_configs/test.yml).",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override config values using dot-delimited keys, e.g. "
            "--set slo_config.slo_s=5.0 basic_config.schema_name=my_schema"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    overrides: dict[str, object] = {}
    for item in getattr(args, "set"):
        key, sep, val = item.partition("=")
        if not key or not sep:
            parser.error(
                f"Invalid --set format: {item!r}  (expected KEY=VALUE)"
            )
        overrides[key] = yaml.safe_load(val)
    apply_overrides(cfg, overrides)

    return cfg, args.config
