"""Tuner configuration dataclass and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class TunerConfig:
    """Settings that control the policy tuning process.

    Loaded from a ``tuner_config.yml`` file via :func:`load_tuner_config`.
    """

    # -- Workload sampling ------------------------------------------------
    num_scenarios: int = 20
    train_fraction: float = 0.6
    random_seed: int = 42
    target_start: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    target_end: datetime = field(default_factory=lambda: datetime(2024, 1, 2))
    forecast_policy: str = "recency_weighted"

    # -- Aggregation ------------------------------------------------------
    aggregation_metric: str = "p90"  # "mean" | "p90" | "p99"

    # -- Checkpoint optimization (step 4) ---------------------------------
    checkpoint_budget: int = 5
    checkpoint_epsilon: float = 0.01
    sliding_window_s: float = 300.0
    violation_threshold: float = 0.1

    # -- Autoscaler sweep (step 5) ----------------------------------------
    autoscaler_ranges: dict[str, list] = field(default_factory=dict)

    # -- Routing sweep (step 6) -------------------------------------------
    routing_ranges: dict[str, list] = field(default_factory=dict)

    # -- Reservoir --------------------------------------------------------
    classify_arrivals: bool = True

    # -- Execution --------------------------------------------------------
    parallelism: int | str = "auto"  # "auto" or explicit int

    @property
    def n_train(self) -> int:
        """Number of training scenarios."""
        return int(self.num_scenarios * self.train_fraction)

    @property
    def n_val(self) -> int:
        """Number of validation scenarios."""
        return self.num_scenarios - self.n_train


def load_tuner_config(path: str | Path) -> TunerConfig:
    """Load a :class:`TunerConfig` from a YAML file.

    Datetime fields (``target_start``, ``target_end``) are parsed from
    ISO-8601 strings via :meth:`datetime.fromisoformat`.
    """
    path = Path(path)
    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    # Flatten nested "target_period" into top-level start/end.
    tp = raw.pop("target_period", {})
    if "start" in tp:
        raw.setdefault("target_start", tp["start"])
    if "end" in tp:
        raw.setdefault("target_end", tp["end"])

    # Parse datetime strings.
    for key in ("target_start", "target_end"):
        val = raw.get(key)
        if isinstance(val, str):
            raw[key] = datetime.fromisoformat(val)

    # Filter to only fields that TunerConfig recognises.
    valid_keys = {f.name for f in TunerConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}

    return TunerConfig(**filtered)
