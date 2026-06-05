"""Centralized YAML dump helper that correctly quotes zero-padded numeric
string keys (e.g. ``"008"``, ``"019"``).

The standard :func:`yaml.safe_dump` leaves these unquoted, which produces
inconsistent output and can confuse non-PyYAML parsers that interpret
leading-zero integers as octal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

_LEADING_ZERO_RE = re.compile(r"^0\d+$")


class _QuotingSafeDumper(yaml.SafeDumper):
    """SafeDumper subclass that single-quotes zero-padded numeric strings."""


def _quote_ambiguous_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if _LEADING_ZERO_RE.match(data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _posix_path_as_str(dumper: yaml.SafeDumper, data: Path) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", data.as_posix(), style="'"
    )


def _dataclass_using_asdict(
    dumper: yaml.SafeDumper, data: Any
) -> yaml.MappingNode:
    if is_dataclass(data) and not isinstance(data, type):
        return dumper.represent_dict(asdict(data))
    raise yaml.representer.RepresenterError(
        f"_QuotingSafeDumper cannot represent an arbitrary Python object: {data!r}"
    )


_QuotingSafeDumper.add_representer(str, _quote_ambiguous_str)
# Catches str subclasses (e.g. QueryTextId) that do not match the exact str
# representer above, coercing them to plain str before serializing.
_QuotingSafeDumper.add_multi_representer(
    str, lambda d, v: _quote_ambiguous_str(d, str(v))
)
_QuotingSafeDumper.add_multi_representer(Path, _posix_path_as_str)
# Numpy scalar types: represent as their nearest Python primitive.
_QuotingSafeDumper.add_multi_representer(
    np.floating, lambda d, v: d.represent_float(float(v))
)
_QuotingSafeDumper.add_multi_representer(
    np.integer, lambda d, v: d.represent_int(int(v))
)
_QuotingSafeDumper.add_multi_representer(
    np.bool_, lambda d, v: d.represent_bool(bool(v))
)
_QuotingSafeDumper.add_multi_representer(object, _dataclass_using_asdict)


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


_PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]*)>")


def detect_placeholders(text: str) -> list[str]:
    """Return a list of placeholder names found in *text*.

    Placeholders have the form ``<UPPERCASE_NAME>``, e.g. ``<TARGET_DATE>``.
    The returned list may contain duplicates if the same placeholder appears
    more than once.
    """
    return _PLACEHOLDER_RE.findall(text)


def load_yaml_with_params(path: str | Path, params: dict[str, str]) -> Any:
    """Load YAML from *path*, substituting ``<KEY>`` placeholders with values
    from *params* before parsing.

    Parameters
    ----------
    path:
        Path to the YAML file (may contain ``<KEY>`` placeholder tokens).
    params:
        Mapping of placeholder name → replacement string.  Keys must not
        include the surrounding angle brackets.

    Returns
    -------
    Any
        The parsed YAML value after substitution, or ``{}`` for an empty file.

    Raises
    ------
    ValueError
        If any ``<KEY>`` placeholder remains unresolved after substitution.
    """

    with open(path) as f:
        text = f.read()

    original_placeholders = set(detect_placeholders(text))

    for key, value in params.items():
        text = text.replace(f"<{key}>", str(value))

    remaining = detect_placeholders(text)
    if remaining:
        raise ValueError(
            f"Config '{Path(path).name}' contains unresolved placeholders after "
            f"substitution: {', '.join(f'<{p}>' for p in sorted(set(remaining)))}. "
            f"Provide them via --param KEY=value."
        )

    unused = set(params) - original_placeholders
    if unused:
        logging.debug(
            "load_yaml_with_params: the following params were not used in "
            "'%s': %s",
            Path(path).name,
            ", ".join(sorted(unused)),
        )

    return yaml.safe_load(text) or {}
