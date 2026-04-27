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


def load_config_from_cli(description: str) -> tuple[dict, str, bool]:
    """Parse CLI args, load YAML config, apply ``--set`` overrides.

    Returns ``(cfg, config_path, force)`` where *cfg* is the fully-resolved
    config dict, *config_path* is the path to the YAML file, and *force*
    is the value of the ``--force`` flag (``False`` unless explicitly
    passed).  Callers that don't care about ``--force`` may simply discard
    it; users that pass ``--force`` to a command which ignores it will see
    it parsed (and ignored) without error.
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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing run/output directory if present. "
            "Commands that don't manage a run directory ignore this flag."
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

    return cfg, args.config, args.force
