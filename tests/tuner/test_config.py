"""Tests for TunerConfig and load_tuner_config."""

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from autoslo.tuner.config import TunerConfig, load_tuner_config


class TestTunerConfigDefaults:
    def test_default_values(self):
        cfg = TunerConfig()
        assert cfg.num_scenarios == 20
        assert cfg.train_fraction == 0.6
        assert cfg.random_seed == 42
        assert cfg.forecast_policy == "recency_weighted"
        assert cfg.aggregation_metric == "p90"
        assert cfg.checkpoint_budget == 5
        assert cfg.parallelism == "auto"

    def test_n_train_n_val(self):
        cfg = TunerConfig(num_scenarios=10, train_fraction=0.7)
        assert cfg.n_train == 7
        assert cfg.n_val == 3

    def test_n_train_n_val_round_down(self):
        cfg = TunerConfig(num_scenarios=10, train_fraction=0.55)
        assert cfg.n_train == 5
        assert cfg.n_val == 5


class TestLoadTunerConfig:
    def test_round_trip(self, tmp_path: Path):
        raw = {
            "num_scenarios": 30,
            "train_fraction": 0.8,
            "random_seed": 99,
            "target_period": {
                "start": "2024-06-01T00:00:00",
                "end": "2024-06-02T00:00:00",
            },
            "forecast_policy": "uniform",
            "aggregation_metric": "mean",
            "checkpoint_budget": 10,
            "checkpoint_epsilon": 0.005,
            "sliding_window_s": 600.0,
            "violation_threshold": 0.2,
            "autoscaler_ranges": {"scale_up_threshold": [0.5, 0.7, 0.9]},
            "routing_ranges": {"cost_weight": [0.1, 0.5]},
            "parallelism": 4,
        }
        yaml_path = tmp_path / "tuner_config.yml"
        with open(yaml_path, "w") as f:
            yaml.dump(raw, f)

        cfg = load_tuner_config(yaml_path)
        assert cfg.num_scenarios == 30
        assert cfg.train_fraction == 0.8
        assert cfg.random_seed == 99
        assert cfg.target_start == datetime(2024, 6, 1)
        assert cfg.target_end == datetime(2024, 6, 2)
        assert cfg.forecast_policy == "uniform"
        assert cfg.aggregation_metric == "mean"
        assert cfg.checkpoint_budget == 10
        assert cfg.checkpoint_epsilon == 0.005
        assert cfg.sliding_window_s == 600.0
        assert cfg.violation_threshold == 0.2
        assert cfg.autoscaler_ranges == {"scale_up_threshold": [0.5, 0.7, 0.9]}
        assert cfg.routing_ranges == {"cost_weight": [0.1, 0.5]}
        assert cfg.parallelism == 4

    def test_unknown_keys_ignored(self, tmp_path: Path):
        raw = {"num_scenarios": 5, "totally_unknown_key": "should be ignored"}
        yaml_path = tmp_path / "tuner_config.yml"
        with open(yaml_path, "w") as f:
            yaml.dump(raw, f)

        cfg = load_tuner_config(yaml_path)
        assert cfg.num_scenarios == 5
        assert not hasattr(cfg, "totally_unknown_key")

    def test_empty_file_uses_defaults(self, tmp_path: Path):
        yaml_path = tmp_path / "tuner_config.yml"
        yaml_path.write_text("")
        cfg = load_tuner_config(yaml_path)
        assert cfg.num_scenarios == 20

    def test_flat_target_start_end(self, tmp_path: Path):
        """target_start/target_end can be specified at top level too."""
        raw = {
            "target_start": "2025-01-15T08:00:00",
            "target_end": "2025-01-16T08:00:00",
        }
        yaml_path = tmp_path / "tuner_config.yml"
        with open(yaml_path, "w") as f:
            yaml.dump(raw, f)

        cfg = load_tuner_config(yaml_path)
        assert cfg.target_start == datetime(2025, 1, 15, 8, 0, 0)
        assert cfg.target_end == datetime(2025, 1, 16, 8, 0, 0)
