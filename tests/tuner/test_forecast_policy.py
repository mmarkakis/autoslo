"""Tests for forecast policies."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from autoslo.tuner.forecast_policy import (
    RecencyWeightedForecastPolicy,
    UniformForecastPolicy,
)

_1H = timedelta(hours=1)


class TestRecencyWeightedWeight:
    def test_same_hour_nonzero(self):
        policy = RecencyWeightedForecastPolicy(half_life_days=14.0, dow_boost=2.0)
        obs = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)  # Mon 09:00
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)  # Mon 09:00 +1wk
        w = policy.weight((obs, obs + _1H), (target, target + _1H))
        assert w > 0

    def test_different_hour_zero(self):
        policy = RecencyWeightedForecastPolicy()
        obs = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        target = datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)  # different hour
        assert policy.weight((obs, obs + _1H), (target, target + _1H)) == 0.0

    def test_recency_decay(self):
        policy = RecencyWeightedForecastPolicy(half_life_days=14.0, dow_boost=1.0)
        target = datetime(2024, 6, 17, 9, 0, tzinfo=timezone.utc)

        obs_recent = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)  # 7 days ago
        obs_old = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)  # 14 days ago

        w_recent = policy.weight((obs_recent, obs_recent + _1H), (target, target + _1H))
        w_old = policy.weight((obs_old, obs_old + _1H), (target, target + _1H))

        assert w_recent > w_old
        # At half-life, weight should be 0.5 (modulo dow_boost=1).
        assert abs(w_old - 0.5) < 0.01

    def test_dow_boost(self):
        policy = RecencyWeightedForecastPolicy(half_life_days=14.0, dow_boost=2.0)
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)  # Monday

        # Same weekday (Monday, 7 days ago).
        obs_same_dow = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        # Different weekday (Tuesday, 8 days ago).
        obs_diff_dow = datetime(2024, 6, 4, 9, 0, tzinfo=timezone.utc)

        w_same = policy.weight((obs_same_dow, obs_same_dow + _1H), (target, target + _1H))
        w_diff = policy.weight((obs_diff_dow, obs_diff_dow + _1H), (target, target + _1H))

        # The same-dow weight should be roughly double (modulo slight recency diff).
        ratio = w_same / w_diff if w_diff > 0 else float("inf")
        assert ratio > 1.5  # should be close to 2.0
        assert ratio < 3.0

    def test_invalid_half_life(self):
        with pytest.raises(ValueError, match="half_life_days"):
            RecencyWeightedForecastPolicy(half_life_days=0)

    def test_invalid_dow_boost(self):
        with pytest.raises(ValueError, match="dow_boost"):
            RecencyWeightedForecastPolicy(dow_boost=-1)


class TestRecencyWeightedExpectedCount:
    def _make_bin_obs(self, counts_per_day: list[int]) -> pd.DataFrame:
        """Create a bin_observations DF with __obs_day_idx."""
        rows = []
        for day_idx, count in enumerate(counts_per_day):
            for _ in range(count):
                rows.append(
                    {
                        "day_of_week": 0,
                        "hour": 9,
                        "timestamp_within_hour": 0.0,
                        "query_text_id": "s#1#001",
                        "repetition_id": "r1",
                        "__obs_day_idx": day_idx,
                    }
                )
        return pd.DataFrame(rows)

    def test_uniform_weights(self):
        policy = RecencyWeightedForecastPolicy(half_life_days=14.0, dow_boost=1.0)
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        # Two days with 10 and 20 queries respectively, equal weights.
        obs = self._make_bin_obs([10, 20])
        result = policy.expected_count((target, target + _1H), obs, [1.0, 1.0])
        assert result == 15  # weighted avg of 10 and 20 with equal weights

    def test_weighted_average(self):
        policy = RecencyWeightedForecastPolicy()
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        # Day 0: 10 queries (weight 1.0), Day 1: 30 queries (weight 3.0).
        obs = self._make_bin_obs([10, 30])
        result = policy.expected_count((target, target + _1H), obs, [1.0, 3.0])
        # weighted_avg = (1*10 + 3*30) / (1+3) = 100/4 = 25
        assert result == 25

    def test_empty_weights(self):
        policy = RecencyWeightedForecastPolicy()
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        obs = self._make_bin_obs([])
        assert policy.expected_count((target, target + _1H), obs, []) == 0

    def test_zero_weights(self):
        policy = RecencyWeightedForecastPolicy()
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        obs = self._make_bin_obs([10, 20])
        assert policy.expected_count((target, target + _1H), obs, [0.0, 0.0]) == 0


class TestUniformForecastPolicy:
    def test_weight_same_hour(self):
        policy = UniformForecastPolicy()
        obs = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        assert policy.weight((obs, obs + _1H), (target, target + _1H)) == 1.0

    def test_weight_different_hour(self):
        policy = UniformForecastPolicy()
        obs = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        target = datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)
        assert policy.weight((obs, obs + _1H), (target, target + _1H)) == 0.0

    def test_expected_count_averages(self):
        policy = UniformForecastPolicy()
        target = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        # 30 rows from 3 unique days → expect 10 per day.
        rows = []
        for day_idx in range(3):
            for _ in range(10):
                rows.append(
                    {
                        "day_of_week": 0,
                        "hour": 9,
                        "timestamp_within_hour": 0.0,
                        "query_text_id": "s#1#001",
                        "repetition_id": "r1",
                        "__obs_day_idx": day_idx,
                    }
                )
        obs = pd.DataFrame(rows)
        assert policy.expected_count((target, target + _1H), obs, [1.0, 1.0, 1.0]) == 10
