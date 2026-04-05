"""Shared config-parsing helpers for WorkloadSimulator and WorkloadRunner."""

from __future__ import annotations

import argparse
import copy
import logging
from typing import Any

import yaml


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


def diff_to_overrides(
    initial_config: dict, modified_config: dict
) -> dict[str, object]:
    """
    Given an initial config and a modified config, compute the dot-delimited
    overrides that would transform the initial config into the modified
    config.

    Example:
        diff_into_overrides(
            {"a": {"b": 1, "c": 2}, "d": 3},
            {"a": {"b": 10, "c": 2}, "d": 30},
        )
        # returns {"a.b": 10, "d": 30}
    """
    overrides = {}

    def recurse(d_initial: dict, d_modified: dict, prefix: str = ""):
        for key in set(d_initial.keys()) | set(d_modified.keys()):
            full_key = f"{prefix}.{key}" if prefix else key
            if key not in d_initial:
                overrides[full_key] = d_modified[key]
            elif key not in d_modified:
                overrides[full_key] = None  # or some sentinel for deletion
            else:
                val_initial = d_initial[key]
                val_modified = d_modified[key]
                if isinstance(val_initial, dict) and isinstance(
                    val_modified, dict
                ):
                    recurse(val_initial, val_modified, full_key)
                elif val_initial != val_modified:
                    overrides[full_key] = val_modified

    recurse(initial_config, modified_config)
    return overrides


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
    cfg = copy_and_apply_overrides(cfg, overrides)

    return cfg, args.config
