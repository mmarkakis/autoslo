"""Centralized YAML dump helper that correctly quotes zero-padded numeric
string keys (e.g. ``"008"``, ``"019"``).

The standard :func:`yaml.safe_dump` leaves these unquoted, which produces
inconsistent output and can confuse non-PyYAML parsers that interpret
leading-zero integers as octal.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml

_LEADING_ZERO_RE = re.compile(r"^0\d+$")


class _QuotingSafeDumper(yaml.SafeDumper):
    """SafeDumper subclass that single-quotes zero-padded numeric strings."""


def _quote_ambiguous_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if _LEADING_ZERO_RE.match(data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_QuotingSafeDumper.add_representer(str, _quote_ambiguous_str)


def dump_yaml(
    data: Any,
    path: str | Path,
    header_string: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Write *data* as YAML to *path*, quoting ambiguous string keys.

    Accepts the same keyword arguments as :func:`yaml.dump`
    (``default_flow_style``, ``sort_keys``, etc.).  ``Dumper`` is always
    overridden to :class:`_QuotingSafeDumper`.
    """
    kwargs.pop("Dumper", None)
    kwargs.setdefault("default_flow_style", False)
    kwargs.setdefault("sort_keys", False)
    # Make sure the output directory exists.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if header_string is not None:
            f.write(f"{header_string}\n")
        yaml.dump(data, f, Dumper=_QuotingSafeDumper, **kwargs)


def load_yaml(path: str | Path) -> Any:
    """
    Load YAML from *path*.

    If the file is empty, return an empty dict instead of None for convenience.
    """
    with open(path) as f:
        return yaml.safe_load(f) or {}
