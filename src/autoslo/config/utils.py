"""Shared config-parsing helpers for WorkloadSimulator and WorkloadRunner."""

from __future__ import annotations

import copy
from typing import Any


def copy_and_apply_overrides(
    initial_config: dict, dot_delimited_overrides: dict[str, object]
) -> dict:
    """
    Apply dot-delimited key overrides to a *deep copy* of the initial config,
    returning the modified copy without mutating the original.

    Example::
        copy_and_apply_overrides(cfg, {"slo_config.slo_s": 5.0})
        # equivalent to  cfg["slo_config"]["slo_s"] = 5.0
    """
    internal_cfg = copy.deepcopy(initial_config)
    for dotted_key, value in dot_delimited_overrides.items():
        parts = dotted_key.split(".")
        d = internal_cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return internal_cfg


def getd(
    cfg: dict,
    dot_delimited_key: str,
    default: object = None,
    required: bool = False,
) -> Any:
    """Read a dot-delimited key from a nested config dict."""
    parts = dot_delimited_key.split(".")
    d = cfg
    for part in parts:
        if not isinstance(d, dict) or part not in d:
            if required:
                raise KeyError(
                    f"Required key '{dot_delimited_key}' not found in config."
                )
            return default
        d = d[part]
    return d
