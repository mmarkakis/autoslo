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


def parse_params(raw: list[str]) -> dict[str, str]:
    """Parse a list of ``KEY=value`` strings into a dict.

    Parameters
    ----------
    raw:
        List of strings, each of the form ``KEY=value``, as produced by
        ``argparse`` with ``action="append"``.

    Returns
    -------
    dict[str, str]
        Mapping of key → value.

    Raises
    ------
    ValueError
        If any entry does not contain exactly one ``=``, or has an empty key.
    """
    result: dict[str, str] = {}
    for entry in raw:
        parts = entry.split("=", 1)
        if len(parts) != 2 or not parts[0]:
            raise ValueError(
                f"Invalid --param format: {entry!r}. Expected KEY=value."
            )
        result[parts[0]] = parts[1]
    return result


def make_run_id(stems: list[str], params: dict[str, str]) -> str:
    """Build a ``__``-separated run identifier from config stems and params.

    The stems are joined first (in the order given), followed by
    ``KEY=value`` segments for every entry in *params*, sorted
    lexicographically by key so the result is deterministic regardless of
    the order params were supplied on the command line.

    Parameters
    ----------
    stems:
        Ordered list of config file stems, e.g.
        ``["base_iconq", "sampled"]``.
    params:
        Substitution parameters, e.g. ``{"TARGET_DATE": "2024-05-27"}``.

    Returns
    -------
    str
        A ``__``-separated identifier, e.g.
        ``"base_iconq__sampled__TARGET_DATE=2024-05-27"``.
        When *params* is empty this degrades to ``"__".join(stems)``.
    """
    parts = list(stems)
    for key in sorted(params):
        parts.append(f"{key}={params[key]}")
    return "__".join(parts)
