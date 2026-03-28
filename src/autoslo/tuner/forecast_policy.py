"""Forecast policies — determine how historical observations map to future bins."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd


class ForecastPolicy(ABC):
    """Abstract base class for forecast policies.

    A forecast policy answers two questions:

    1. **weight** — Given a historical observation interval and a target
       interval we want to forecast, how relevant is that observation?
    2. **expected_count** — Given the weighted observations that fall into
       the same (hour-of-day, [day-of-week]) bin, how many queries should
       we sample for the target interval?
    """

    @abstractmethod
    def weight(
        self,
        obs_interval: tuple[datetime, datetime],
        target_interval: tuple[datetime, datetime],
    ) -> float:
        """Return a non-negative relevance weight.

        Parameters
        ----------
        obs_interval :
            ``(start, end)`` of a historical observation.  Its start's
            *day_of_week* and *hour* determine the bin.
        target_interval :
            ``(start, end)`` of the target interval being forecast.
        """

    @abstractmethod
    def expected_count(
        self,
        target_interval: tuple[datetime, datetime],
        bin_observations: pd.DataFrame,
        weights: list[float],
    ) -> int:
        """Return the expected number of query arrivals for a target interval.

        Parameters
        ----------
        target_interval :
            ``(start, end)`` of the target bin.
        bin_observations :
            Reservoir DataFrame filtered to historical rows that share the
            same hour as the target interval's start.
        weights :
            Per-unique-observation-day weight (aligned with the unique
            observation days in *bin_observations*).
        """


class RecencyWeightedForecastPolicy(ForecastPolicy):
    """Weight historical bins by recency, with a boost for matching weekday.

    Parameters
    ----------
    half_life_days :
        Exponential-decay half-life.  An observation from *half_life_days*
        ago receives weight 0.5 (before the weekday boost).
    dow_boost :
        Multiplicative boost applied when the observed day shares the same
        weekday as the target.
    """

    def __init__(
        self,
        half_life_days: float = 14.0,
        dow_boost: float = 2.0,
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if dow_boost < 0:
            raise ValueError("dow_boost must be non-negative")
        self.half_life_days = half_life_days
        self.dow_boost = dow_boost

    def weight(
        self,
        obs_interval: tuple[datetime, datetime],
        target_interval: tuple[datetime, datetime],
    ) -> float:
        obs_start = obs_interval[0]
        target_start = target_interval[0]

        # Only match the same hour-of-day.
        if obs_start.hour != target_start.hour:
            return 0.0

        days_apart = abs((target_start - obs_start).total_seconds()) / 86400.0
        w = math.exp(-math.log(2) * days_apart / self.half_life_days)

        if obs_start.weekday() == target_start.weekday():
            w *= self.dow_boost

        return w

    def expected_count(
        self,
        target_interval: tuple[datetime, datetime],
        bin_observations: pd.DataFrame,
        weights: list[float],
    ) -> int:
        """Weighted average of per-day counts, rounded to the nearest int.

        Each *weight* corresponds to one unique historical day that
        contributed observations to this bin.  ``bin_observations`` has a
        ``day_of_week`` column but we need per-observation-day counts.
        The caller provides per-unique-day weights.
        """
        if not weights or sum(weights) == 0:
            return 0

        # bin_observations must have an "__obs_day_idx" column added by the
        # sampler so we can align per-day counts with the weight list.
        if "__obs_day_idx" in bin_observations.columns:
            day_counts = (
                bin_observations.groupby("__obs_day_idx")
                .size()
                .to_dict()
            )
            total_w = 0.0
            weighted_sum = 0.0
            for idx, w in enumerate(weights):
                c = day_counts.get(idx, 0)
                weighted_sum += w * c
                total_w += w
        else:
            # Fallback: just use total count / number of weights.
            total_w = sum(weights)
            weighted_sum = len(bin_observations) / max(1, len(weights)) * total_w

        if total_w == 0:
            return 0
        return max(0, round(weighted_sum / total_w))


class UniformForecastPolicy(ForecastPolicy):
    """Equal weight for all matching historical bins (simple average).

    Useful as a baseline or when recency weighting is not desired.
    """

    def weight(
        self,
        obs_interval: tuple[datetime, datetime],
        target_interval: tuple[datetime, datetime],
    ) -> float:
        if obs_interval[0].hour != target_interval[0].hour:
            return 0.0
        return 1.0

    def expected_count(
        self,
        target_interval: tuple[datetime, datetime],
        bin_observations: pd.DataFrame,
        weights: list[float],
    ) -> int:
        if not weights or sum(weights) == 0:
            return 0
        if "__obs_day_idx" in bin_observations.columns:
            n_days = bin_observations["__obs_day_idx"].nunique()
        else:
            n_days = max(1, len(weights))
        return max(0, round(len(bin_observations) / n_days))
