"""Tests for forecast policies."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone, date

import numpy as np
import pandas as pd
import pytest

from typing import Optional

from autoslo.tuner.forecast_policy import (
    OneDayForecastPolicy,
    SevenDaysFlatForecastPolicy,
    SameDayOnceForecastPolicy,
    SameDayExponentialForecastPolicy,
)
from autoslo.tuner.reservoir import QueryReservoir

from scipy import stats

_1H = timedelta(hours=1)
_BASE_DATE = date(2024, 1, 31)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _create_df(
    start_date: date = _BASE_DATE,
    num_days: int = 1,
    hour_to_hour_increase: int = 0,
    day_to_day_increase: int = 0,
    template_to_frequency: dict[int, int] = {1: 1},
    initial_count: int = 10,
    force_day_idx_empty: Optional[int] = None,
) -> pd.DataFrame:
    """Create a workload dataframe"""

    template_rng = np.random.default_rng(42)
    total_freq = sum(template_to_frequency.values())
    template_to_normalized_freq = {
        template_id: freq / total_freq
        for template_id, freq in template_to_frequency.items()
    }
    rows = []
    start_time = pd.Timestamp(start_date)
    for d in range(num_days):
        initial_count_for_day = initial_count + (day_to_day_increase * d)
        current_time = start_time + pd.Timedelta(days=d)

        if force_day_idx_empty is not None and d == force_day_idx_empty:
            continue

        for h in range(24):
            count = initial_count_for_day + (hour_to_hour_increase * h)
            for i in range(count):
                template_id = template_rng.choice(
                    list(template_to_normalized_freq.keys()),
                    p=list(template_to_normalized_freq.values()),
                )
                rows.append(
                    {
                        "query_id": f"q_{d}_{h}_{i:04d}",
                        "abs_start_time": current_time + pd.Timedelta(hours=h),
                        "query_text_id": f"s#{template_id:02d}#001",
                        "repetition_id": "r1",
                    }
                )
    df = pd.DataFrame(rows)
    return df


def _create_reservoir(
    start_date: date = _BASE_DATE,
    num_days: int = 1,
    hour_to_hour_increase: int = 0,
    day_to_day_increase: int = 0,
    template_to_frequency: dict[int, int] = {1: 1},
    initial_count: int = 10,
    force_day_idx_empty: Optional[int] = None,
) -> QueryReservoir:
    """Create a reservoir."""

    df = _create_df(
        start_date=start_date,
        num_days=num_days,
        hour_to_hour_increase=hour_to_hour_increase,
        day_to_day_increase=day_to_day_increase,
        template_to_frequency=template_to_frequency,
        initial_count=initial_count,
        force_day_idx_empty=force_day_idx_empty,
    )
    return QueryReservoir(df=df)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestOneDayForecastPolicy:

    def test_forecast_matches_yesterday_stable(self):
        """
        Forecast should match yesterday's counts when there is no hour-to-hour
        or day-to-day increase.
        """
        reservoir = _create_reservoir(start_date=_BASE_DATE - timedelta(days=1))
        policy = OneDayForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        # Expect 10 queries per hour, same as yesterday.
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            assert counts.get(hour, 0) == 10

    def test_forecast_matches_yesterday_variable(self):
        """
        Forecast should match yesterday's counts when there is an hour-to-hour
        increase, but no day-to-day increase.
        """
        reservoir = _create_reservoir(
            hour_to_hour_increase=5, start_date=_BASE_DATE - timedelta(days=1)
        )
        policy = OneDayForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            expected_count = 10 + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_start_times_are_roughly_poisson(self):
        """
        The inter-arrival times in the forecast should
        be roughly Poisson, i.e. exponentially distributed.
        """
        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=1), initial_count=1000
        )
        policy = OneDayForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        forecast_workload.df.sort_values("abs_start_time", inplace=True)
        for h in range(24):
            hour_df = forecast_workload.df[
                forecast_workload.df["abs_start_time"].dt.hour == h
            ]
            if len(hour_df) < 2:
                continue  # not enough data to test distribution
            inter_arrival_times = (
                hour_df["abs_start_time"].diff().dropna().dt.total_seconds()
            )
            # Test that inter-arrival times are roughly exponentially distributed.
            # We allow some tolerance since this is a stochastic test.
            mean_inter_arrival = inter_arrival_times.mean()
            assert (
                mean_inter_arrival > 0
            ), "Mean inter-arrival time should be positive"
            # Check that the distribution of inter-arrival times is not too far from exponential.
            ks_statistic, p_value = stats.kstest(
                inter_arrival_times, "expon", args=(0, mean_inter_arrival)
            )
            assert (
                p_value > 0.01
            ), f"Inter-arrival times for hour {h} do not appear to be exponentially distributed (KS p-value={p_value:.3f})"

    def test_extra_past_days_do_not_matter(self):
        """
        Forecast should match yesterday's counts even if there are multiple
        past days in the reservoir, with a day-to-day increase.
        """
        reservoir = _create_reservoir(
            day_to_day_increase=10,
            num_days=3,
            start_date=_BASE_DATE - timedelta(days=3),
        )
        policy = OneDayForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            assert counts.get(hour, 0) == 30

    def test_random_seed_makes_forecast_deterministic(self):
        """
        A fixed random seed should make the forecast deterministic, even if
        there is random sampling involved in the forecast.
        """
        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=1),
            template_to_frequency={1: 1, 2: 2, 3: 3},
        )
        policy = OneDayForecastPolicy(reservoir)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=42)
        pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_different_random_seeds_make_different_forecasts(self):
        """
        Different random seeds should produce different forecasts, even if the
        reservoir and other parameters are the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=1),
            template_to_frequency={1: 1, 2: 2, 3: 3},
        )
        policy = OneDayForecastPolicy(reservoir)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=43)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_n_scenarios_are_different(self):
        """
        When generating multiple scenarios, they should not all be the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=1),
            template_to_frequency={1: 1, 2: 2, 3: 3},
        )
        policy = OneDayForecastPolicy(reservoir)
        workloads, paths = policy.forecast_n_scenarios(
            _BASE_DATE, n_scenarios=5, initial_seed=42
        )
        assert len(workloads) == 5
        assert paths is None
        for i in range(4):
            for j in range(i + 1, 5):
                with pytest.raises(AssertionError):
                    pd.testing.assert_frame_equal(
                        workloads[i].df, workloads[j].df
                    )

    def test_correct_template_distribution(self):
        """
        In the presence of multiple templates with different frequencies, the
        forecast should reflect the correct distribution of templates.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=1),
            template_to_frequency={1: 1, 2: 2},
            initial_count=100,
        )
        policy = OneDayForecastPolicy(reservoir)
        forecast = policy.forecast(_BASE_DATE, seed=42)
        template_counts = forecast.df["query_text_id"].value_counts()

        total_count = template_counts.sum()
        expected_distribution = {
            f"s#{template_id:02d}#001": freq / 3  # total frequency is 1+2=3
            for template_id, freq in {1: 1, 2: 2}.items()
        }
        for template_id, expected_freq in expected_distribution.items():
            actual_freq = template_counts.get(template_id, 0) / total_count
            # Allow some variability due to random sampling, but should be close.
            assert math.isclose(
                actual_freq, expected_freq, abs_tol=0.1
            ), f"Template {template_id} has frequency {actual_freq:.2f}, expected {expected_freq:.2f}"


class TestSevenDaysFlatForecastPolicy:

    def test_forecast_uses_only_same_hour_from_past_days(self):
        """
        Forecast for each hour should be based only on the same hour from the past
        7 days, not on other hours.
        """

        reservoir = _create_reservoir(
            day_to_day_increase=0,
            hour_to_hour_increase=5,
            num_days=7,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=7),
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            assert counts.get(hour, 0) == 10 + (hour * 5)

    def test_forecast_matches_average_of_past_7_days(self):
        """
        Forecast should match the average of the past 7 days, even if there is a
        day-to-day increase.
        """

        reservoir = _create_reservoir(
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            num_days=7,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=7),
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        base_expected_count = sum(10 + (d * 10) for d in range(7)) // 7
        for hour in range(24):
            expected_count = base_expected_count + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_forecast_with_insufficient_history(self):
        """
        If there are fewer than 7 days of history, the forecast should still be
        based on the average of whatever history is available. Importantly, it
        should not incorrectly still divide by 7.
        """

        reservoir = _create_reservoir(
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            num_days=3,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=3),
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        base_expected_count = sum(10 + (d * 10) for d in range(3)) // 3
        for hour in range(24):
            expected_count = base_expected_count + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_extra_past_days_do_not_matter(self):
        """
        Having more than 7 days of history should not change the forecast.
        """

        reservoir = _create_reservoir(
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            num_days=14,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=14),
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        base_expected_count = sum(10 + 7 * 10 + (d * 10) for d in range(7)) // 7
        for hour in range(24):
            expected_count = base_expected_count + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_random_seed_makes_forecast_deterministic(self):
        """
        A fixed random seed should make the forecast deterministic, even if there
        is random sampling involved in the forecast.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
            num_days=7,
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=42)
        pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_different_random_seeds_make_different_forecasts(self):
        """
        Different random seeds should produce different forecasts, even if the
        reservoir and other parameters are the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
            num_days=7,
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=43)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_n_scenarios_are_different(self):
        """
        When generating multiple scenarios, they should not all be the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        workloads, paths = policy.forecast_n_scenarios(
            _BASE_DATE, n_scenarios=5, initial_seed=42
        )
        assert len(workloads) == 5
        assert paths is None
        for i in range(4):
            for j in range(i + 1, 5):
                with pytest.raises(AssertionError):
                    pd.testing.assert_frame_equal(
                        workloads[i].df, workloads[j].df
                    )

    def test_correct_template_distribution(self):
        """
        In the presence of multiple templates with different frequencies, the
        forecast should reflect the correct distribution of templates.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2},
            initial_count=100,
            num_days=7,
        )
        policy = SevenDaysFlatForecastPolicy(reservoir)
        forecast = policy.forecast(_BASE_DATE, seed=42)
        template_counts = forecast.df["query_text_id"].value_counts()

        total_count = template_counts.sum()
        expected_distribution = {
            f"s#{template_id:02d}#001": freq / 3  # total frequency is 1+2=3
            for template_id, freq in {1: 1, 2: 2}.items()
        }
        for template_id, expected_freq in expected_distribution.items():
            actual_freq = template_counts.get(template_id, 0) / total_count
            # Allow some variability due to random sampling, but should be close.
            assert math.isclose(
                actual_freq, expected_freq, abs_tol=0.1
            ), f"Template {template_id} has frequency {actual_freq:.2f}, expected {expected_freq:.2f}"


class TestSameDayOnceForecastPolicy:

    def test_forecast_matches_same_day(self):
        """
        Forecast should match the same day's counts.
        """
        reservoir = _create_reservoir(
            num_days=7,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=7),
        )
        policy = SameDayOnceForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            expected_count = 10 + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_forecast_empty_if_empty_with_enough_history(self):
        """
        If there is an empty day but the cluster was active before it, we
        should return an empty forecast.
        """

        reservoir = _create_reservoir(
            num_days=8,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=8),
            force_day_idx_empty=1,
        )
        policy = SameDayOnceForecastPolicy(reservoir)

        bin_df = policy._build_bin_df(_BASE_DATE, 0)
        assert bin_df.empty, "Expected empty bin_df"
        assert (
            bin_df.columns.tolist() == QueryReservoir.BIN_DF_COLUMNS
        ), f"Expected columns {QueryReservoir.BIN_DF_COLUMNS}, got {bin_df.columns.tolist()}"

        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        assert forecast_workload.df.empty, "Expected empty forecast workload"

    def test_forecast_fallback_if_empty_with_insufficient_history(self):
        """
        If there is an empty day and the cluster was not active before it, we
        should fall back to using the previous day's data for the forecast.
        """

        reservoir = _create_reservoir(
            num_days=1,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=1),
        )
        policy = SameDayOnceForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            expected_count = 10 + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_extra_past_days_do_not_matter(self):
        """
        Having more than 7 days of history should not change the forecast,
        even if some are the same day of week.
        """

        reservoir = _create_reservoir(
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            num_days=14,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=14),
        )
        policy = SameDayOnceForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        one_week_in_base_count = 10 + (7 * 10)
        for hour in range(24):
            expected_count = one_week_in_base_count + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_random_seed_makes_forecast_deterministic(self):
        """
        A fixed random seed should make the forecast deterministic, even if there
        is random sampling involved in the forecast.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
            num_days=7,
        )
        policy = SameDayOnceForecastPolicy(reservoir, seed=42)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=42)
        pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_different_random_seeds_make_different_forecasts(self):
        """
        Different random seeds should produce different forecasts, even if the
        reservoir and other parameters are the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
            num_days=7,
        )
        policy = SameDayOnceForecastPolicy(reservoir)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=43)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_n_scenarios_are_different(self):
        """
        When generating multiple scenarios, they should not all be the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
        )
        policy = SameDayOnceForecastPolicy(reservoir, seed=42)
        workloads, paths = policy.forecast_n_scenarios(
            _BASE_DATE, n_scenarios=5, initial_seed=42
        )
        assert len(workloads) == 5
        assert paths is None
        for i in range(4):
            for j in range(i + 1, 5):
                with pytest.raises(AssertionError):
                    pd.testing.assert_frame_equal(
                        workloads[i].df, workloads[j].df
                    )

    def test_correct_template_distribution(self):
        """
        In the presence of multiple templates with different frequencies, the
        forecast should reflect the correct distribution of templates.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2},
            initial_count=100,
            num_days=7,
        )
        policy = SameDayOnceForecastPolicy(reservoir)
        forecast = policy.forecast(_BASE_DATE, seed=42)
        template_counts = forecast.df["query_text_id"].value_counts()

        total_count = template_counts.sum()
        expected_distribution = {
            f"s#{template_id:02d}#001": freq / 3  # total frequency is 1+2=3
            for template_id, freq in {1: 1, 2: 2}.items()
        }
        for template_id, expected_freq in expected_distribution.items():
            actual_freq = template_counts.get(template_id, 0) / total_count
            # Allow some variability due to random sampling, but should be close.
            assert math.isclose(
                actual_freq, expected_freq, abs_tol=0.1
            ), f"Template {template_id} has frequency {actual_freq:.2f}, expected {expected_freq:.2f}"


class TestSameDayExponentialForecastPolicy:

    def test_forecast_matches_same_day(self):
        """
        Forecast should match the same day's counts.
        """
        reservoir = _create_reservoir(
            num_days=7,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=7),
        )
        policy = SameDayExponentialForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            expected_count = 10 + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_forecast_applies_decay_on_older_day(self):
        """
        For older instances of the target weekday, we should apply decay.
        """
        decay = 0.2
        reservoir = _create_reservoir(
            num_days=14,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=14),
            force_day_idx_empty=7,
        )
        policy = SameDayExponentialForecastPolicy(reservoir, decay_factor=decay)
        forecast_workload = policy.forecast(_BASE_DATE, seed=42)
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            expected_count = int(round((10 + (hour * 5)) * decay / (1 + decay)))
            assert counts.get(hour, 0) == expected_count

    def test_forecast_with_multiple_decayed_days(self):
        """
        If there are multiple past instances of the same day, we should apply
        decay to all of them.
        """
        decay = 0.2
        reservoir = _create_reservoir(
            num_days=21,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=21),
        )
        policy = SameDayExponentialForecastPolicy(reservoir, decay_factor=decay)
        forecast_workload = policy.forecast(
            _BASE_DATE,
            seed=42,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            expected_count = int(
                round(
                    (
                        ((10 + (hour * 5)) * decay**2)
                        + ((10 + 7 * 10 + (hour * 5)) * decay)
                        + (10 + 14 * 10 + (hour * 5))
                    )
                    / (1 + decay + decay**2)
                )
            )
            assert counts.get(hour, 0) == expected_count

    def test_forecast_empty_if_empty_with_enough_history(self):
        """
        If there is an empty day but the cluster was active before it, we
        should return an empty forecast.
        """

        reservoir = _create_reservoir(
            num_days=8,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=8),
            force_day_idx_empty=1,
        )
        policy = SameDayExponentialForecastPolicy(reservoir)

        bin_df = policy._build_bin_df(_BASE_DATE, 0)
        assert bin_df.empty, "Expected empty bin_df"
        assert (
            bin_df.columns.tolist() == QueryReservoir.BIN_DF_COLUMNS
        ), f"Expected columns {QueryReservoir.BIN_DF_COLUMNS}, got {bin_df.columns.tolist()}"

        forecast_workload = policy.forecast(
            _BASE_DATE,
        )
        assert forecast_workload.df.empty, "Expected empty forecast workload"

    def test_forecast_fallback_if_empty_with_insufficient_history(self):
        """
        If there is an empty day and the cluster was not active before it, we
        should fallback to the previous day's data.
        """

        reservoir = _create_reservoir(
            num_days=1,
            day_to_day_increase=10,
            hour_to_hour_increase=5,
            initial_count=10,
            start_date=_BASE_DATE - timedelta(days=1),
        )
        policy = SameDayExponentialForecastPolicy(reservoir)
        forecast_workload = policy.forecast(
            _BASE_DATE,
            seed=42,
        )
        counts = forecast_workload.df.groupby(
            forecast_workload.df["abs_start_time"].dt.hour
        ).size()
        for hour in range(24):
            expected_count = 10 + (hour * 5)
            assert counts.get(hour, 0) == expected_count

    def test_random_seed_makes_forecast_deterministic(self):
        """
        A fixed random seed should make the forecast deterministic, even if there
        is random sampling involved in the forecast.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
            num_days=7,
        )
        policy = SameDayExponentialForecastPolicy(reservoir)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=42)
        pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_different_random_seeds_make_different_forecasts(self):
        """
        Different random seeds should produce different forecasts, even if the
        reservoir and other parameters are the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2, 3: 3},
            num_days=7,
        )
        policy = SameDayExponentialForecastPolicy(reservoir)
        forecast1 = policy.forecast(_BASE_DATE, seed=42)
        forecast2 = policy.forecast(_BASE_DATE, seed=43)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(forecast1.df, forecast2.df)

    def test_n_scenarios_are_different(self):
        """
        When generating multiple scenarios, they should not all be the same.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=14),
            template_to_frequency={1: 1, 2: 2, 3: 3},
        )
        policy = SameDayExponentialForecastPolicy(reservoir, decay_factor=0.5)
        workloads, paths = policy.forecast_n_scenarios(
            _BASE_DATE, n_scenarios=5, initial_seed=42
        )
        assert len(workloads) == 5
        assert paths is None
        for i in range(4):
            for j in range(i + 1, 5):
                with pytest.raises(AssertionError):
                    pd.testing.assert_frame_equal(
                        workloads[i].df, workloads[j].df
                    )

    def test_correct_template_distribution(self):
        """
        In the presence of multiple templates with different frequencies, the
        forecast should reflect the correct distribution of templates.
        """

        reservoir = _create_reservoir(
            start_date=_BASE_DATE - timedelta(days=7),
            template_to_frequency={1: 1, 2: 2},
            initial_count=100,
            num_days=7,
        )
        policy = SameDayExponentialForecastPolicy(reservoir)
        forecast = policy.forecast(_BASE_DATE, seed=42)
        template_counts = forecast.df["query_text_id"].value_counts()

        total_count = template_counts.sum()
        expected_distribution = {
            f"s#{template_id:02d}#001": freq / 3  # total frequency is 1+2=3
            for template_id, freq in {1: 1, 2: 2}.items()
        }
        for template_id, expected_freq in expected_distribution.items():
            actual_freq = template_counts.get(template_id, 0) / total_count
            # Allow some variability due to random sampling, but should be close.
            assert math.isclose(
                actual_freq, expected_freq, abs_tol=0.1
            ), f"Template {template_id} has frequency {actual_freq:.2f}, expected {expected_freq:.2f}"

    def test_decay_influences_template_distribution(self):
        """
        The decay factor should influence the template distribution in the
        forecast, since it changes the relative contribution of different days
        which may have different template distributions.
        """

        df1 = _create_df(
            start_date=_BASE_DATE - timedelta(days=14),
            num_days=7,
            hour_to_hour_increase=0,
            day_to_day_increase=0,
            template_to_frequency={1: 1},
            initial_count=100,
        )
        df2 = _create_df(
            start_date=_BASE_DATE - timedelta(days=7),
            num_days=7,
            hour_to_hour_increase=0,
            day_to_day_increase=0,
            template_to_frequency={2: 1},
            initial_count=100,
        )
        reservoir = QueryReservoir(df=pd.concat([df1, df2]))

        # Without decay
        policy_no_decay = SameDayExponentialForecastPolicy(
            reservoir, decay_factor=1.0
        )
        forecast_no_decay = policy_no_decay.forecast(_BASE_DATE, seed=42)
        template_counts_no_decay = forecast_no_decay.df[
            "query_text_id"
        ].value_counts()
        total_count_no_decay = template_counts_no_decay.sum()
        expected_distribution_no_decay = {
            f"s#{template_id:02d}#001": freq / 2  # total frequency is 1+1=2
            for template_id, freq in {1: 1, 2: 1}.items()
        }
        for (
            template_id,
            expected_freq,
        ) in expected_distribution_no_decay.items():
            actual_freq = (
                template_counts_no_decay.get(template_id, 0)
                / total_count_no_decay
            )
            assert math.isclose(
                actual_freq, expected_freq, abs_tol=0.1
            ), f"No decay: Template {template_id} has frequency {actual_freq:.2f}, expected {expected_freq:.2f}"

        # With decay, the older day (template 1) should have less influence.
        decay = 0.5
        policy_with_decay = SameDayExponentialForecastPolicy(
            reservoir, decay_factor=decay
        )
        forecast_with_decay = policy_with_decay.forecast(_BASE_DATE, seed=42)
        template_counts_with_decay = forecast_with_decay.df[
            "query_text_id"
        ].value_counts()
        total_count_with_decay = template_counts_with_decay.sum()
        expected_distribution_with_decay = {
            f"s#01#001": (1 * decay)
            / (1 + decay),  # template 1 is from the older day
            f"s#02#001": 1
            / (1 + decay),  # template 2 is from the more recent day
        }
        for (
            template_id,
            expected_freq,
        ) in expected_distribution_with_decay.items():
            actual_freq = (
                template_counts_with_decay.get(template_id, 0)
                / total_count_with_decay
            )
            assert math.isclose(
                actual_freq, expected_freq, abs_tol=0.1
            ), f"With decay: Template {template_id} has frequency {actual_freq:.2f}, expected {expected_freq:.2f}"
