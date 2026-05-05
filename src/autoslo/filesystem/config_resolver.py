"""
Config reference resolver.

Supported syntax:
  exec:<path>    →  data/execution_configs/<path>.yml
  tuner:<name>   →  data/tuner_configs/<name>.yml
  eval:<name>    →  data/simulator_eval_specs/<name>.yml
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path

import autoslo.filesystem.path_utils as pu
from autoslo.config.utils import make_run_id
from autoslo.filesystem.yaml_helpers import load_yaml

_SCHEME_DIRS: dict[str, str] = {
    "exec": "execution_configs",
    "tuner": "tuner_configs",
    "eval": "simulator_eval_specs",
}


def resolve_config(ref: str) -> Path:
    """Resolve a prefixed config reference to an absolute Path."""
    data = Path(pu.get_data_path())
    scheme, _, path_part = ref.partition(":")
    sub_dir = _SCHEME_DIRS.get(scheme.lower())
    if sub_dir is None:
        raise ValueError(
            f"Unknown config scheme in '{ref}'. Use exec:, tuner:, or eval:."
        )
    if path_part.endswith(".yml"):
        path_part = path_part[:-4]
    return data / sub_dir / (path_part + ".yml")


@functools.lru_cache(maxsize=64)
def _load_trial_spec(spec_path: Path) -> dict:
    return load_yaml(spec_path)


def resolve_series_exec_config_id(
    entry: dict, root: Path | None = None
) -> str | None:
    """
    Return the exec_config_id string for a plot-spec series entry.

    Handles two formats:
      trial_ref:    {spec_dir, trial_id} -> make_run_id from trial_spec.yml
      baseline_ref: {exec_config, params} -> make_run_id([exec_stem], params)
    """
    if root is None:
        root = Path(pu.AUTOSLO_ROOT)

    if "trial_ref" in entry:
        ref = entry["trial_ref"]
        spec_dir = Path(ref["spec_dir"])
        if not spec_dir.is_absolute():
            spec_dir = root / spec_dir
        trial_spec = _load_trial_spec(spec_dir / "trial_spec.yml")
        trial_id: str = ref["trial_id"]
        default_exec: str = trial_spec.get("exec_config", "")
        default_tuner: str = trial_spec.get("tuner_config", "")
        for trial in trial_spec.get("trials", []):
            if trial["trial_id"] == trial_id:
                exec_cfg = trial.get("exec_config", default_exec)
                tuner_cfg = trial.get("tuner_config", default_tuner)
                params = dict(trial.get("params") or {})
                return make_run_id(
                    [
                        resolve_config(exec_cfg).stem,
                        resolve_config(tuner_cfg).stem,
                    ],
                    params,
                )
        warnings.warn(
            f"trial_id '{trial_id}' not found in {spec_dir / 'trial_spec.yml'}"
        )
        return None

    if "baseline_ref" in entry:
        ref = entry["baseline_ref"]
        exec_path = resolve_config(ref["exec_config"])
        params = dict(ref.get("params") or {})
        return make_run_id([exec_path.stem], params)
    return None


# ---------------------------------------------------------------------------
# Baseline group expansion (Idea D)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_baseline_groups() -> dict:
    path = (
        Path(pu.get_data_path())
        / "simulator_eval_specs"
        / "baseline_groups.yml"
    )
    return load_yaml(path) if path.exists() else {}


def _expand_group(name: str, override_params: dict) -> list[dict]:
    """Return the members of a baseline group with override_params merged in."""
    groups = _load_baseline_groups()
    members = groups.get(name)
    if members is None:
        raise ValueError(f"Unknown baseline group '{name}'.")
    return [
        {
            "exec_config": m["exec_config"],
            "label": m["label"],
            "formatting_id": m["formatting_id"],
            "params": {**dict(m.get("params") or {}), **override_params},
        }
        for m in members
    ]


def expand_eval_baselines(baselines: list[dict]) -> list[dict]:
    """Expand {group, params} entries in an eval spec's baselines list."""
    result: list[dict] = []
    for b in baselines:
        if "group" not in b:
            result.append(b)
        else:
            result.extend(
                _expand_group(b["group"], dict(b.get("params") or {}))
            )
    return result


def expand_series_entries(entries: list[dict]) -> list[dict]:
    """Expand {baseline_group, workload_id} entries in a plot-spec series list."""
    result: list[dict] = []
    for entry in entries:
        if "baseline_group" not in entry:
            result.append(entry)
            continue
        bg = entry["baseline_group"]
        wid = entry.get("workload_id", "")
        for m in _expand_group(bg["name"], dict(bg.get("params") or {})):
            result.append(
                {
                    "baseline_ref": {
                        "exec_config": m["exec_config"],
                        "params": m["params"],
                    },
                    "workload_id": wid,
                    "label": m["label"],
                    "formatting_id": m["formatting_id"],
                }
            )
    return result
