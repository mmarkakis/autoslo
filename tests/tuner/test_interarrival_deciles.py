"""Tests for the interarrival-decile arrival-time policy.

Covers:
  - _compute_weighted_gap_deciles  (pure function, no I/O)
  - _sample_hour_from_deciles      (pure function, no I/O)
  - QueryReservoir arrivals table  (constructor, save/load, bin access)
"""

from __future__ import annotations

from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pytest

from autoslo.forecasting.forecaster import Forecaster
from autoslo.tuner.reservoir import QueryReservoir

_BASE_DATE = date(2024, 1, 31)


class _DummyForecasterConfig:
    def __init__(self, max_arrivals_per_hour_safety_cap: int) -> None:
        self.max_arrivals_per_hour_safety_cap = max_arrivals_per_hour_safety_cap


class _DummyForecaster:
    def __init__(self, max_arrivals_per_hour_safety_cap: int) -> None:
        self.forecaster_config = _DummyForecasterConfig(
            max_arrivals_per_hour_safety_cap
        )


def _sample_hour_from_deciles(
    deciles: np.ndarray,
    safety_cap: int,
    rng: np.random.Generator,
) -> list[float]:
    return Forecaster._sample_hour_from_deciles(
        cast(
            Forecaster,
            _DummyForecaster(max_arrivals_per_hour_safety_cap=safety_cap),
        ),
        deciles,
        rng,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_df(
    dates: list[date],
    hours: list[int] | None = None,
    count: int = 10,
) -> pd.DataFrame:
    if hours is None:
        hours = [0]
    rows = [
        {"date": d, "hour": h, "query_text_id": "s#01#001", "count": count}
        for d in dates
        for h in hours
    ]
    return pd.DataFrame(rows)


def _arrivals_df(
    dates: list[date],
    hours: list[int] | None = None,
    seconds: list[float] | None = None,
) -> pd.DataFrame:
    """Arrivals with evenly-spaced offsets within the hour by default."""
    if hours is None:
        hours = [0]
    if seconds is None:
        seconds = [i * 360.0 for i in range(10)]  # 10 arrivals, 360 s apart
    rows = [
        {"date": d, "hour": h, "second_of_hour": s}
        for d in dates
        for h in hours
        for s in seconds
    ]
    return pd.DataFrame(rows)


def _make_reservoir(
    dates: list[date],
    hours: list[int] | None = None,
    seconds: list[float] | None = None,
) -> QueryReservoir:
    return QueryReservoir(
        count_df=_count_df(dates, hours),
        arrivals_df=_arrivals_df(dates, hours, seconds),
    )


# ---------------------------------------------------------------------------
# _compute_weighted_gap_deciles
# ---------------------------------------------------------------------------


class TestComputeWeightedGapDeciles:

    def test_returns_11_boundaries(self):
        arrivals = np.array([0.0, 360.0, 720.0, 1080.0, 1440.0])
        result = Forecaster._compute_weighted_gap_deciles(
            [arrivals], [1.0], min_gaps_required=1
        )
        assert result is not None
        assert result.shape == (11,)

    def test_returns_none_below_threshold(self):
        arrivals = np.array([0.0, 360.0])  # 2 gaps only
        result = Forecaster._compute_weighted_gap_deciles(
            [arrivals], [1.0], min_gaps_required=10
        )
        assert result is None

    def test_returns_none_for_empty_input(self):
        result = Forecaster._compute_weighted_gap_deciles(
            [np.array([])], [1.0], min_gaps_required=1
        )
        assert result is None

    def test_returns_none_for_all_empty_days(self):
        result = Forecaster._compute_weighted_gap_deciles(
            [np.array([]), np.array([])], [1.0, 1.0], min_gaps_required=1
        )
        assert result is None

    def test_monotone_non_decreasing(self):
        rng = np.random.default_rng(42)
        arrivals = np.sort(rng.uniform(0, 3600, size=200))
        result = Forecaster._compute_weighted_gap_deciles(
            [arrivals], [1.0], min_gaps_required=1
        )
        assert result is not None
        assert np.all(np.diff(result) >= 0)

    def test_identical_days_equal_single_day(self):
        """N identical days with equal weights must give the same deciles as one day."""
        rng = np.random.default_rng(0)
        arrivals = np.sort(rng.uniform(0, 3600, size=80))
        result_one = Forecaster._compute_weighted_gap_deciles(
            [arrivals], [1.0], min_gaps_required=1
        )
        result_three = Forecaster._compute_weighted_gap_deciles(
            [arrivals, arrivals, arrivals], [1.0, 1.0, 1.0], min_gaps_required=1
        )
        assert result_one is not None and result_three is not None
        np.testing.assert_allclose(result_one, result_three, rtol=1e-6)

    def test_higher_weight_shifts_median_toward_that_day(self):
        # Day 1: dense (small gaps ~10 s).  Day 2: sparse (large gaps ~500 s).
        # Upweighting day 2 should raise the median boundary.
        dense = np.array([i * 10.0 for i in range(200)])
        sparse = np.array([i * 500.0 for i in range(7)])
        result_equal = Forecaster._compute_weighted_gap_deciles(
            [dense, sparse], [1.0, 1.0], min_gaps_required=1
        )
        result_upweighted = Forecaster._compute_weighted_gap_deciles(
            [dense, sparse], [1.0, 100.0], min_gaps_required=1
        )
        assert result_equal is not None and result_upweighted is not None
        # Median = boundary at index 5 (of 11)
        assert result_upweighted[5] > result_equal[5]

    def test_empty_day_does_not_affect_result(self):
        arrivals = np.array([100.0, 400.0, 900.0, 1500.0])
        result_plain = Forecaster._compute_weighted_gap_deciles(
            [arrivals], [1.0], min_gaps_required=1
        )
        result_with_empty = Forecaster._compute_weighted_gap_deciles(
            [arrivals, np.array([])], [1.0, 1.0], min_gaps_required=1
        )
        assert result_plain is not None and result_with_empty is not None
        np.testing.assert_allclose(result_plain, result_with_empty)

    def test_min_gap_clamp_applied(self):
        # Arrivals at identical timestamps → raw gap is 0, should be clamped.
        arrivals = np.array([0.0, 0.0, 0.0, 360.0, 720.0])
        result = Forecaster._compute_weighted_gap_deciles(
            [arrivals], [1.0], min_gaps_required=1, min_gap_s=1.0
        )
        assert result is not None
        assert result[0] >= 1.0  # minimum boundary must be >= clamp value


# ---------------------------------------------------------------------------
# _sample_hour_from_deciles
# ---------------------------------------------------------------------------


class TestSampleHourFromDeciles:

    @staticmethod
    def _uniform_deciles(gap: float = 300.0) -> np.ndarray:
        """All-equal deciles: every gap will be exactly `gap` seconds."""
        return np.full(11, gap)

    def test_all_offsets_strictly_within_hour(self):
        deciles = self._uniform_deciles(200.0)
        offsets = _sample_hour_from_deciles(
            deciles, safety_cap=10_000, rng=np.random.default_rng(0)
        )
        assert all(0.0 <= t < 3600.0 for t in offsets)

    def test_offsets_sorted(self):
        rng = np.random.default_rng(99)
        deciles = np.sort(rng.uniform(50, 600, 11))
        offsets = _sample_hour_from_deciles(
            deciles, safety_cap=10_000, rng=np.random.default_rng(1)
        )
        assert offsets == sorted(offsets)

    def test_deterministic_under_same_seed(self):
        deciles = self._uniform_deciles(180.0)
        r1 = _sample_hour_from_deciles(
            deciles, safety_cap=10_000, rng=np.random.default_rng(7)
        )
        r2 = _sample_hour_from_deciles(
            deciles, safety_cap=10_000, rng=np.random.default_rng(7)
        )
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        rng = np.random.default_rng(5)
        deciles = np.sort(rng.uniform(10, 600, 11))
        r1 = _sample_hour_from_deciles(
            deciles, safety_cap=10_000, rng=np.random.default_rng(1)
        )
        r2 = _sample_hour_from_deciles(
            deciles, safety_cap=10_000, rng=np.random.default_rng(2)
        )
        assert r1 != r2

    def test_count_emerges_and_varies_across_seeds(self):
        """Hourly count must not be fixed — it must vary across different seeds."""
        rng = np.random.default_rng(3)
        deciles = np.sort(rng.uniform(50, 1200, 11))
        counts = {
            len(
                _sample_hour_from_deciles(
                    deciles, safety_cap=10_000, rng=np.random.default_rng(seed)
                )
            )
            for seed in range(30)
        }
        assert len(counts) > 1, "Expected emergent (varying) hourly counts"

    def test_safety_cap_terminates_degenerate_case(self):
        """Very small gap → safety cap must prevent runaway loop."""
        deciles = np.full(11, 1e-9)
        offsets = _sample_hour_from_deciles(
            deciles, safety_cap=50, rng=np.random.default_rng(0)
        )
        assert len(offsets) <= 50

    def test_empty_result_when_gap_exceeds_hour(self):
        """If every sampled gap > 3600 s the result is empty."""
        deciles = np.full(11, 7200.0)
        offsets = _sample_hour_from_deciles(
            deciles, safety_cap=10_000, rng=np.random.default_rng(0)
        )
        assert offsets == []

    def test_approximate_count_matches_expected_rate(self):
        """Mean gap ~360 s → ~10 arrivals/hour on average."""
        deciles = np.full(11, 360.0)
        counts = [
            len(
                _sample_hour_from_deciles(
                    deciles, safety_cap=10_000, rng=np.random.default_rng(s)
                )
            )
            for s in range(200)
        ]
        mean_count = sum(counts) / len(counts)
        # 3600 / 360 = 10; allow generous tolerance for the log-interpolation effect.
        assert 7 <= mean_count <= 13, f"Mean count {mean_count:.1f} outside [7, 13]"


# ---------------------------------------------------------------------------
# QueryReservoir arrivals table
# ---------------------------------------------------------------------------


class TestReservoirArrivals:

    def test_has_arrivals_true_when_provided(self):
        r = _make_reservoir([_BASE_DATE])
        assert r.has_arrivals is True

    def test_has_arrivals_false_when_not_provided(self):
        r = QueryReservoir(count_df=_count_df([_BASE_DATE]))
        assert r.has_arrivals is False

    def test_arrivals_bin_df_correct_columns(self):
        r = _make_reservoir([_BASE_DATE], hours=[0, 1])
        df = r.arrivals_bin_df(_BASE_DATE, 0)
        assert list(df.columns) == QueryReservoir.ARRIVALS_DF_COLUMNS

    def test_arrivals_bin_df_filters_by_date_and_hour(self):
        other_date = date(2024, 1, 30)
        r = _make_reservoir([_BASE_DATE, other_date], hours=[0, 1])
        df = r.arrivals_bin_df(_BASE_DATE, 0)
        assert (df["date"] == _BASE_DATE).all()
        assert (df["hour"] == 0).all()

    def test_arrivals_bin_df_correct_row_count(self):
        seconds = [i * 200.0 for i in range(15)]
        r = _make_reservoir([_BASE_DATE], hours=[0], seconds=seconds)
        df = r.arrivals_bin_df(_BASE_DATE, 0)
        assert len(df) == len(seconds)

    def test_arrivals_bin_df_raises_without_arrivals(self):
        r = QueryReservoir(count_df=_count_df([_BASE_DATE]))
        with pytest.raises(RuntimeError, match="Arrivals data is not available"):
            r.arrivals_bin_df(_BASE_DATE, 0)

    def test_arrivals_df_property_raises_without_arrivals(self):
        r = QueryReservoir(count_df=_count_df([_BASE_DATE]))
        with pytest.raises(RuntimeError, match="Arrivals data is not available"):
            _ = r.arrivals_df

    def test_arrivals_bin_df_invalid_hour_raises(self):
        r = _make_reservoir([_BASE_DATE])
        with pytest.raises(ValueError, match="Invalid hour"):
            r.arrivals_bin_df(_BASE_DATE, 25)

    def test_save_writes_both_files(self, tmp_path):
        r = _make_reservoir([_BASE_DATE])
        r.save(tmp_path)
        assert (tmp_path / "reservoir.parquet").exists()
        assert (tmp_path / "reservoir_arrivals.parquet").exists()

    def test_save_without_arrivals_writes_only_count_file(self, tmp_path):
        r = QueryReservoir(count_df=_count_df([_BASE_DATE]))
        r.save(tmp_path)
        assert (tmp_path / "reservoir.parquet").exists()
        assert not (tmp_path / "reservoir_arrivals.parquet").exists()

    def test_load_with_arrivals_roundtrip(self, tmp_path):
        seconds = [100.0, 500.0, 1200.0, 2000.0, 3000.0]
        r = _make_reservoir([_BASE_DATE], seconds=seconds)
        r.save(tmp_path)
        loaded = QueryReservoir.load(tmp_path)
        assert loaded.has_arrivals is True
        pd.testing.assert_frame_equal(
            r.arrivals_bin_df(_BASE_DATE, 0),
            loaded.arrivals_bin_df(_BASE_DATE, 0),
        )

    def test_load_without_arrivals_file_succeeds(self, tmp_path):
        """Backward compatibility: reservoirs saved before this feature must load."""
        _count_df([_BASE_DATE]).to_parquet(
            tmp_path / "reservoir.parquet", index=False
        )
        # Deliberately do NOT write reservoir_arrivals.parquet.
        loaded = QueryReservoir.load(tmp_path)
        assert loaded.has_arrivals is False

    def test_count_df_unaffected_by_arrivals(self):
        """The existing count-based API must be unchanged when arrivals are added."""
        r = _make_reservoir([_BASE_DATE], hours=[0, 1])
        df = r.bin_df(_BASE_DATE, 0)
        assert list(df.columns) == QueryReservoir.BIN_DF_COLUMNS
        assert len(df) == 1
        assert df.iloc[0]["count"] == 10
