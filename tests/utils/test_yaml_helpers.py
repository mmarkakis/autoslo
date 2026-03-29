"""Tests for autoslo.utils.yaml_helpers.dump_config."""

from __future__ import annotations

import io

import yaml

from autoslo.utils.yaml_helpers import dump_config


def test_zero_padded_keys_are_quoted():
    """All zero-padded numeric string keys should survive a YAML round-trip."""
    data = {str(i).zfill(3): float(i) for i in range(100)}
    buf = io.StringIO()
    dump_config(data, buf)

    raw = buf.getvalue()
    reloaded = yaml.safe_load(raw)

    # Every key must still be a 3-char zero-padded string after reload.
    for i in range(100):
        key = str(i).zfill(3)
        assert key in reloaded, f"Key '{key}' lost during round-trip"
        assert reloaded[key] == float(i)


def test_non_numeric_strings_unquoted():
    """Plain strings should not get spurious quoting."""
    data = {"hello": 1, "world": 2}
    buf = io.StringIO()
    dump_config(data, buf)
    raw = buf.getvalue()
    # No single-quoted keys expected for plain alpha strings.
    assert "'hello'" not in raw
    assert "'world'" not in raw


def test_sort_keys_kwarg():
    """sort_keys=False should be honoured."""
    data = {"b": 1, "a": 2}
    buf = io.StringIO()
    dump_config(data, buf, sort_keys=False)
    raw = buf.getvalue()
    assert raw.index("b:") < raw.index("a:")


def test_default_flow_style_false():
    """Output should use block style by default."""
    data = {"nested": {"x": 1, "y": 2}}
    buf = io.StringIO()
    dump_config(data, buf)
    raw = buf.getvalue()
    # Block style: no braces.
    assert "{" not in raw
