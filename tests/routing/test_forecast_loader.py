"""Tests for ForecastDistributionLoader.

These tests use YAML fixtures written to tmp directories and a
stub featurizer to avoid hitting real artefacts.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest
import yaml

from autoslo.routing.cache_risk_scorer import FutureQueryMix
from autoslo.routing.forecast_loader import ForecastDistributionLoader


# ---------------------------------------------------------------------------
# Stub featurizer
# ---------------------------------------------------------------------------

class _StubFeaturizer:
    """Returns a deterministic feature vector for known QueryTextIds."""

    def __init__(self, known: dict[str, list[float]], m: int = 2) -> None:
        # known: template_id -> table_vector (length n)
        self._known = known
        self._m = m

    def featurize_from_query_text_id(self, qtid) -> list[float]:
        # QueryTextId is a str subclass: "schema#tid#index"
        parts = str(qtid).split("#")
        tid = parts[1]
        if tid not in self._known:
            raise KeyError(f"Unknown template {tid}")
        table_vec = self._known[tid]
        # Full featurization = 2*m zeros (operator dims) + table_vec
        return [0.0] * (2 * self._m) + table_vec

    def table_vector_for(self, qtid) -> np.ndarray:
        # QueryTextId is a str subclass: "schema#tid#index"
        parts = str(qtid).split("#")
        tid = parts[1]
        if tid not in self._known:
            raise KeyError(f"Unknown template {tid}")
        return np.array(self._known[tid], dtype=np.float64)


# ---------------------------------------------------------------------------
# YAML fixture helpers
# ---------------------------------------------------------------------------

def _write_forecast_yaml(path: str, bins: list[dict], schema: str = "test_schema") -> None:
    data = {
        "schema_name": schema,
        "window_minutes": 60,
        "bins": bins,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def _write_tightness_yaml(path: str, entries: dict[str, dict], schema: str = "test_schema") -> None:
    data = {
        "schema_name": schema,
        "reference_rpu": 8,
        "stage_model_id": "12345",
        "slo_source": "test_slo",
        "entries": entries,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_TABLE = 3
M_OPERATOR = 2

TEMPLATES = {
    "1": [1.0, 0.0, 0.0],
    "2": [0.0, 1.0, 0.0],
    "3": [0.0, 0.0, 1.0],
}


@pytest.fixture
def stub_featurizer():
    return _StubFeaturizer(known=TEMPLATES, m=M_OPERATOR)


@pytest.fixture
def yaml_dir(tmp_path):
    forecast_bins = [
        {
            "day_of_week": 0,
            "hour": 9,
            "templates": [
                {"template_id": "1", "probability": 0.5},
                {"template_id": "2", "probability": 0.3},
                {"template_id": "3", "probability": 0.2},
            ],
        },
        {
            "day_of_week": 4,
            "hour": 17,
            "templates": [
                {"template_id": "1", "probability": 0.8},
                {"template_id": "3", "probability": 0.2},
            ],
        },
    ]

    tightness_entries = {
        "1": {"isolated_prediction_s": 2.0, "slo_s": 5.0, "tightness": 0.4},
        "2": {"isolated_prediction_s": 1.0, "slo_s": 10.0, "tightness": 0.1},
        "3": {"isolated_prediction_s": 4.0, "slo_s": 5.0, "tightness": 0.8},
    }

    forecast_path = str(tmp_path / "forecast.yml")
    tightness_path = str(tmp_path / "tightness.yml")

    _write_forecast_yaml(forecast_path, forecast_bins)
    _write_tightness_yaml(tightness_path, tightness_entries)

    return forecast_path, tightness_path


@pytest.fixture
def loader(yaml_dir, stub_featurizer):
    forecast_path, tightness_path = yaml_dir
    return ForecastDistributionLoader(
        forecast_distribution_path=forecast_path,
        slo_tightness_path=tightness_path,
        iconq_query_featurizer=stub_featurizer,
        n_table_dims=N_TABLE,
        m_operator_dims=M_OPERATOR,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestForecastDistributionLoaderConstruction:
    def test_schema_name(self, loader):
        assert loader.schema_name == "test_schema"

    def test_available_bins(self, loader):
        assert loader.available_bins == [(0, 9), (4, 17)]

    def test_all_template_ids(self, loader):
        assert loader.all_template_ids == ["1", "2", "3"]


class TestBinLookup:
    def test_matching_bin_returns_correct_mix(self, loader):
        # Monday 9am UTC
        dt = datetime(2025, 7, 7, 9, 30, tzinfo=timezone.utc)  # Monday
        mix = loader.get_future_query_mix(base_datetime=dt)

        assert mix.template_ids == ["1", "2", "3"]
        np.testing.assert_allclose(mix.probabilities, [0.5, 0.3, 0.2])
        assert mix.table_vectors.shape == (3, N_TABLE)
        np.testing.assert_allclose(mix.slo_tightness, [0.4, 0.1, 0.8])

    def test_second_bin(self, loader):
        # Friday 17:xx
        dt = datetime(2025, 7, 11, 17, 0, tzinfo=timezone.utc)  # Friday
        mix = loader.get_future_query_mix(base_datetime=dt)

        assert mix.template_ids == ["1", "3"]
        np.testing.assert_allclose(mix.probabilities, [0.8, 0.2])

    def test_timestamp_s_lookup(self, loader):
        # Monday 9:15 UTC via epoch
        dt = datetime(2025, 7, 7, 9, 15, tzinfo=timezone.utc)
        ts = dt.timestamp()
        mix = loader.get_future_query_mix(timestamp_s=ts)
        assert mix.template_ids == ["1", "2", "3"]

    def test_nonmatching_bin_returns_default(self, loader):
        # Sunday 3am — no bin
        dt = datetime(2025, 7, 6, 3, 0, tzinfo=timezone.utc)
        mix = loader.get_future_query_mix(base_datetime=dt)

        # Default: uniform over all 3 templates
        assert set(mix.template_ids) == {"1", "2", "3"}
        np.testing.assert_allclose(mix.probabilities, [1 / 3, 1 / 3, 1 / 3])

    def test_no_args_uses_now(self, loader):
        # Should not raise; we don't assert exact bin since it depends on
        # current time, just check it returns a valid FutureQueryMix.
        mix = loader.get_future_query_mix()
        assert isinstance(mix, FutureQueryMix)
        assert len(mix.template_ids) > 0


class TestTableVectors:
    def test_table_vectors_correct_values(self, loader):
        dt = datetime(2025, 7, 7, 9, 0, tzinfo=timezone.utc)
        mix = loader.get_future_query_mix(base_datetime=dt)

        # Template "1" → [1,0,0],  "2" → [0,1,0],  "3" → [0,0,1]
        np.testing.assert_array_equal(mix.table_vectors[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(mix.table_vectors[1], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(mix.table_vectors[2], [0.0, 0.0, 1.0])


class TestFallbackTightness:
    def test_missing_tightness_uses_fallback(self, tmp_path, stub_featurizer):
        # Template "3" not in tightness table → should get fallback
        forecast_bins = [
            {
                "day_of_week": 0,
                "hour": 9,
                "templates": [
                    {"template_id": "1", "probability": 0.5},
                    {"template_id": "3", "probability": 0.5},
                ],
            },
        ]
        tightness_entries = {
            "1": {"isolated_prediction_s": 2.0, "slo_s": 5.0, "tightness": 0.4},
            # "3" intentionally missing
        }

        fp = str(tmp_path / "f.yml")
        tp = str(tmp_path / "t.yml")
        _write_forecast_yaml(fp, forecast_bins)
        _write_tightness_yaml(tp, tightness_entries)

        loader = ForecastDistributionLoader(
            forecast_distribution_path=fp,
            slo_tightness_path=tp,
            iconq_query_featurizer=stub_featurizer,
            n_table_dims=N_TABLE,
            m_operator_dims=M_OPERATOR,
            fallback_tightness=0.75,
        )

        dt = datetime(2025, 7, 7, 9, 0, tzinfo=timezone.utc)
        mix = loader.get_future_query_mix(base_datetime=dt)
        # "1" → 0.4, "3" → 0.75 (fallback)
        np.testing.assert_allclose(mix.slo_tightness, [0.4, 0.75])


class TestUnknownTemplate:
    def test_unknown_template_gets_zero_vector(self, tmp_path):
        # Template "999" not in the featurizer → zero vector fallback
        featurizer = _StubFeaturizer(known={"1": [1.0, 0.0, 0.0]}, m=M_OPERATOR)
        forecast_bins = [
            {
                "day_of_week": 0,
                "hour": 9,
                "templates": [
                    {"template_id": "1", "probability": 0.5},
                    {"template_id": "999", "probability": 0.5},
                ],
            },
        ]
        tightness_entries = {
            "1": {"isolated_prediction_s": 2.0, "slo_s": 5.0, "tightness": 0.4},
        }

        fp = str(tmp_path / "f.yml")
        tp = str(tmp_path / "t.yml")
        _write_forecast_yaml(fp, forecast_bins)
        _write_tightness_yaml(tp, tightness_entries)

        loader = ForecastDistributionLoader(
            forecast_distribution_path=fp,
            slo_tightness_path=tp,
            iconq_query_featurizer=featurizer,
            n_table_dims=N_TABLE,
            m_operator_dims=M_OPERATOR,
        )

        dt = datetime(2025, 7, 7, 9, 0, tzinfo=timezone.utc)
        mix = loader.get_future_query_mix(base_datetime=dt)
        # Template "999" → zero vector
        np.testing.assert_array_equal(mix.table_vectors[1], [0.0, 0.0, 0.0])


class TestEmptyBins:
    def test_empty_forecast_gives_degenerate_default(self, tmp_path):
        featurizer = _StubFeaturizer(known={}, m=M_OPERATOR)

        fp = str(tmp_path / "f.yml")
        tp = str(tmp_path / "t.yml")
        _write_forecast_yaml(fp, [])
        _write_tightness_yaml(tp, {})

        loader = ForecastDistributionLoader(
            forecast_distribution_path=fp,
            slo_tightness_path=tp,
            iconq_query_featurizer=featurizer,
            n_table_dims=N_TABLE,
            m_operator_dims=M_OPERATOR,
        )

        mix = loader.get_future_query_mix(base_datetime=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc))
        assert mix.template_ids == ["_empty"]
        np.testing.assert_array_equal(mix.probabilities, [1.0])
        np.testing.assert_array_equal(mix.slo_tightness, [0.0])


class TestMixIsValidFutureQueryMix:
    """Ensure every returned mix passes FutureQueryMix's post_init checks."""

    def test_bin_mix_passes_validation(self, loader):
        dt = datetime(2025, 7, 7, 9, 0, tzinfo=timezone.utc)
        mix = loader.get_future_query_mix(base_datetime=dt)
        # Re-construct through dataclass to trigger __post_init__
        FutureQueryMix(
            template_ids=mix.template_ids,
            probabilities=mix.probabilities,
            table_vectors=mix.table_vectors,
            slo_tightness=mix.slo_tightness,
        )

    def test_default_mix_passes_validation(self, loader):
        dt = datetime(2025, 7, 6, 3, 0, tzinfo=timezone.utc)
        mix = loader.get_future_query_mix(base_datetime=dt)
        FutureQueryMix(
            template_ids=mix.template_ids,
            probabilities=mix.probabilities,
            table_vectors=mix.table_vectors,
            slo_tightness=mix.slo_tightness,
        )
