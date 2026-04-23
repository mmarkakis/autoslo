"""Tests for autoslo.utils.yaml_helpers.dump."""

from __future__ import annotations

from pathlib import Path

import yaml

from autoslo.utils.yaml_helpers import dump_yaml


def test_zero_padded_keys_are_quoted(tmp_path: Path):
    """All zero-padded numeric string keys should survive a YAML round-trip."""
    data = {str(i).zfill(3): float(i) for i in range(100)}
    out = tmp_path / "out.yml"
    dump_yaml(data, str(out))

    reloaded = yaml.safe_load(out.read_text())

    # Every key must still be a 3-char zero-padded string after reload.
    for i in range(100):
        key = str(i).zfill(3)
        assert key in reloaded, f"Key '{key}' lost during round-trip"
        assert reloaded[key] == float(i)


def test_non_numeric_strings_unquoted(tmp_path: Path):
    """Plain strings should not get spurious quoting."""
    data = {"hello": 1, "world": 2}
    out = tmp_path / "out.yml"
    dump_yaml(data, str(out))
    raw = out.read_text()
    # No single-quoted keys expected for plain alpha strings.
    assert "'hello'" not in raw
    assert "'world'" not in raw


def test_sort_keys_kwarg(tmp_path: Path):
    """sort_keys=False should be honoured."""
    data = {"b": 1, "a": 2}
    out = tmp_path / "out.yml"
    dump_yaml(data, str(out), sort_keys=False)
    raw = out.read_text()
    assert raw.index("b:") < raw.index("a:")


def test_default_flow_style_false(tmp_path: Path):
    """Output should use block style by default."""
    data = {"nested": {"x": 1, "y": 2}}
    out = tmp_path / "out.yml"
    dump_yaml(data, str(out))
    raw = out.read_text()
    # Block style: no braces.
    assert "{" not in raw
