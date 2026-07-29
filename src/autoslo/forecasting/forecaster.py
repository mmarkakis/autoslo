"""Forecast policies — determine how historical observations map to future bins."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import ForecasterConfig, WorkloadConfig
from autoslo.forecasting.forecast_policy import (
    ArrivalTimePolicy,
    ForecastPolicy,
)
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.workload_definition.workload import Workload

logger = logging.getLogger(__name__)


class Forecaster:
    """Abstract base class for forecast policies."""

    def __init__(
        self,
        forecaster_config: ForecasterConfig,
    ) -> None:
        """
        Initialize the forecaster.

        Parameters
        ----------
        forecaster_config :
            Configuration for the forecaster, including the forecast policy and
            any nested configurations such as the query reservoir.
        """

        self.forecaster_config = forecaster_config
        self.forecast_policy = ForecastPolicy(
            forecaster_config.forecast_policy_name
        )
        self.arrival_time_policy = ArrivalTimePolicy(
            forecaster_config.arrival_time_policy_name
        )
        if self.forecast_policy != ForecastPolicy.NONE:
            reservoir_config = forecaster_config.reservoir_config
            if reservoir_config is None:
                raise ValueError(
                    "reservoir_config must be provided for forecast policies other than 'none'"
                )
            self.reservoir = QueryReservoir(reservoir_config)
            if (
                self.arrival_time_policy
                == ArrivalTimePolicy.INTERARRIVAL_DECILES
                and not self.reservoir.has_arrivals
            ):
                raise ValueError(
                    "arrival_time_policy_name='interarrival_deciles' requires a "
                    "reservoir built from a workload file, but this reservoir has "
                    "no per-arrival timing data."
                )

    def forecast(
        self,
        target_date: date | str,
        seed: int = 42,
        out_dir: Optional[Path] = None,
        workload_name: str = "forecast",
        rescale_factor: float = 1.0,
        use_fixed_queries_per_hour: bool = False,
    ) -> WorkloadConfig:
        """
        Return a forecasted workload for the target interval. The forecasted
        workload is also written out to the specified directory.

        Parameters
        ----------.
        target_date :
            Date of the target interval to forecast for.
        seed :
            Random seed for reproducibility.
        out_dir :
            The directory to write the forecasted workload to. If None, the
            workload will be written to `data/workloads/{workload_name}.parquet`.
        workload_name :
            Name for the forecasted workload.
        use_fixed_queries_per_hour :
            If true, ignore the historical counts and use a fixed number of
            queries per hour as specified in the forecaster config.
        """
        if self.forecast_policy == ForecastPolicy.NONE:
            raise ValueError(
                "Cannot call forecast when forecast policy is 'none'"
            )

        if isinstance(target_date, str):
            target_date = pd.Timestamp(target_date).date()
        assert isinstance(target_date, date)

        rng = np.random.default_rng(seed)
        rows = []
        query_idx = 0

        for i in range(24):
            bin_df = self._build_bin_df(target_date, i)
            if bin_df.empty:
                continue

            if (
                self.arrival_time_policy
                == ArrivalTimePolicy.INTERARRIVAL_DECILES
                and not use_fixed_queries_per_hour
            ):
                # use_fixed_queries_per_hour forces the UNIFORM path so that
                # callers (e.g. cache-aware routing) that need a fixed count
                # are unaffected by the arrival-time policy.
                deciles = self._get_gap_deciles_for_bin(target_date, i)
                if deciles is not None:
                    second_offsets = self._sample_hour_from_deciles(
                        deciles,
                        rng,
                    )
                    n_samples = len(second_offsets)
                else:
                    # Sparse-bin fallback: count-based uniform timing.
                    n_samples = self._n_samples(target_date, i, bin_df)
                    if n_samples == 0:
                        continue
                    second_offsets = list(
                        np.sort(rng.uniform(0, 3600, size=n_samples))
                    )
            else:
                # UNIFORM path, also used when use_fixed_queries_per_hour=True.
                n_samples = self._n_samples(target_date, i, bin_df)
                if use_fixed_queries_per_hour:
                    n_samples = self.forecaster_config.fixed_queries_per_hour
                if n_samples == 0:
                    continue
                second_offsets = list(
                    np.sort(rng.uniform(0, 3600, size=n_samples))
                )

            if n_samples == 0:
                continue

            # Sample query templates weighted by historical counts.
            sampled_ids = bin_df.sample(
                n=n_samples,
                weights="count",
                replace=True,
                random_state=rng,
            )["query_text_id"]

            base_start_time = pd.Timestamp(target_date) + pd.Timedelta(hours=i)

            for query_text_id, rel_start_time in zip(
                sampled_ids, second_offsets
            ):
                rows.append(
                    {
                        "query_id": f"forecast_{query_idx:06d}",
                        "abs_start_time": (
                            base_start_time
                            + pd.Timedelta(seconds=rel_start_time)
                        ),
                        "query_text_id": query_text_id,
                        "repetition_id": 0,
                    }
                )
                query_idx += 1

        forecast_df = pd.DataFrame(
            rows, columns=Workload.WORKLOAD_SCHEMA_COLUMNS
        )
        out_dir = out_dir or pu.get_data_path() / "workloads"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{workload_name}.parquet"
        forecast_df.to_parquet(out_path, index=False)

        workload_config = WorkloadConfig(
            workload_name=workload_name,
            workload_dir=out_dir,
            rescale_factor=rescale_factor,
        )
        workload = Workload(workload_config=workload_config)
        workload.save(
            out_dir=out_dir, overwrite=True
        )  # Save again after rescale.
        return workload_config

    def forecast_n_scenarios(
        self,
        target_date: date | str,
        n_scenarios: int,
        initial_seed: int = 42,
        workload_name_prefix: str = "f",
        out_dir: Optional[Path] = None,
        use_fixed_queries_per_hour: bool = False,
        rescale_factor: float = 1.0,
    ) -> list[WorkloadConfig]:
        """
        Return a list of forecasted workloads for the target day. Optionally
        also persist them.

        Parameters
        ----------.
        target_date :
           Date of the target day to forecast for.
        n_scenarios :
            Number of independent forecasted workloads to generate for the
            target day.  Each workload is generated by calling `forecast`
            with a different seed to ensure different random samples across
            scenarios.
        initial_seed :
            Initial random seed for reproducibility.  The method will derive
            separate seeds **sequentially** for each scenario to ensure
            different random samples across scenarios.
        workload_name_prefix :
            Prefix for the forecasted workload names. The full name for each
            workload will be `{workload_name_prefix}_{i}`, where `i` is the
            scenario index (starting from 0).
        out_dir :
            If provided, the directory to persist the forecasted workloads to.

        Returns
        -------
        workload_configs :
            The list of forecasted workload configurations.
        """
        if self.forecast_policy == ForecastPolicy.NONE:
            raise ValueError(
                "Cannot call forecast_n_scenarios when forecast policy is 'none'"
            )

        workload_configs: list[WorkloadConfig] = []
        for i in range(n_scenarios):
            workload_name = f"{workload_name_prefix}_{i}"
            workload_seed = initial_seed + i
            workload_config = self.forecast(
                target_date,
                seed=workload_seed,
                workload_name=workload_name,
                out_dir=out_dir,
                use_fixed_queries_per_hour=use_fixed_queries_per_hour,
                rescale_factor=rescale_factor,
            )
            workload_configs.append(workload_config)

        return workload_configs

    def _build_bin_df(self, target_date: date, hour: int) -> pd.DataFrame:
        """
        Return the right DataFrame of historical observations for the specified
        bin.
        """
        yesterday = target_date - pd.Timedelta(days=1)
        one_week_ago = target_date - pd.Timedelta(days=7)

        if self.forecast_policy == ForecastPolicy.ONE_DAY:
            return self.reservoir.bin_df(yesterday, hour)
        elif self.forecast_policy == ForecastPolicy.SEVEN_DAYS_FLAT:
            superbin_list = [
                self.reservoir.bin_df(one_week_ago + pd.Timedelta(days=d), hour)
                for d in range(7)
            ]
            return pd.concat(superbin_list)
        elif self.forecast_policy == ForecastPolicy.SAME_DAY_ONCE:
            if one_week_ago >= self.reservoir.min_date:
                return self.reservoir.bin_df(one_week_ago, hour)
            return self.reservoir.bin_df(yesterday, hour)
        elif self.forecast_policy == ForecastPolicy.SAME_DAY_EXPONENTIAL:
            return self._build_bin_df_exponential(target_date, hour)
        else:
            raise ValueError(
                f"Unsupported forecast policy: {self.forecast_policy}"
            )

    def _n_samples(
        self, target_date: date, hour: int, bin_df: pd.DataFrame
    ) -> int:
        """Return the number of samples to draw for the specified bin."""

        if self.forecast_policy in (
            {
                ForecastPolicy.ONE_DAY,
                ForecastPolicy.SEVEN_DAYS_FLAT,
                ForecastPolicy.SAME_DAY_ONCE,
            }
        ):
            num_active_days = bin_df["date"].nunique()
            if num_active_days == 0:
                return 0
            return int(round(bin_df["count"].sum() / num_active_days))
        elif self.forecast_policy == ForecastPolicy.SAME_DAY_EXPONENTIAL:
            return self._n_samples_exponential(target_date, hour, bin_df)
        else:
            raise ValueError(
                f"Unsupported forecast policy: {self.forecast_policy}"
            )

    def _build_bin_df_exponential(
        self, target_date: date, hour: int
    ) -> pd.DataFrame:
        weight = 1.0
        day = target_date - pd.Timedelta(days=7)
        superbin_list = []
        while day >= self.reservoir.min_date:
            bin_df = self.reservoir.bin_df(day, hour)
            if not bin_df.empty:
                bin_df = bin_df.copy()
                bin_df["count"] *= weight  # apply the weight to the count
            weight *= (
                self.forecaster_config.decay_factor
            )  # decay by the specified factor every week
            day -= pd.Timedelta(days=7)
            superbin_list.append(bin_df)

        if len(superbin_list) > 0:
            return pd.concat(superbin_list)

        if (target_date - pd.Timedelta(days=7)) >= self.reservoir.min_date:
            # Should return an empty DataFrame with the correct columns if there
            # are no observations but the date is within the reservoir's range.
            return pd.DataFrame(columns=self.reservoir.BIN_DF_COLUMNS)

        # If we get here, it means there are no observations and the date is
        # before the reservoir's range, so we should try one day ago instead.
        one_day_ago = target_date - pd.Timedelta(days=1)
        one_day_ago_bin_df = self.reservoir.bin_df(one_day_ago, hour)

        return one_day_ago_bin_df

    def _n_samples_exponential(
        self, target_date: date, hour: int, bin_df: pd.DataFrame
    ) -> int:

        # Are we in the fallback case where we had to look at one day ago
        # instead of one week ago?
        if bin_df["date"].nunique() == 1 and (
            bin_df["date"].iloc[0] == (target_date - pd.Timedelta(days=1))
        ):
            return int(bin_df["count"].sum())

        # Take a weighted average of the past counts, where the weights are the
        # same as those applied in _build_bin_df.
        total_weight = 0.0
        weighted_count_sum = 0.0
        weight = 1.0
        day = target_date - pd.Timedelta(days=7)
        while day >= self.reservoir.min_date:
            day_bin_df = self.reservoir.bin_df(day, hour)
            day_count = day_bin_df["count"].sum()
            weighted_count_sum += weight * day_count
            total_weight += weight
            weight *= (
                self.forecaster_config.decay_factor
            )  # decay by the specified factor every week
            day -= pd.Timedelta(days=7)
        if total_weight == 0:
            return 0

        return int(round(weighted_count_sum / total_weight))

    def _get_gap_deciles_for_bin(
        self, target_date: date, hour: int
    ) -> np.ndarray | None:
        """
        Return gap decile boundaries (linear scale) for the given bin, or None
        if all fallback levels fail to accumulate min_gaps_for_deciles gaps.

        Fallback order (hard-coded):
          1. Policy-window source days for this hour (mirrors _build_bin_df).
          2. All available history for this hour (equal weights).
          3. Union of hours h-1, h, h+1 across all available history.
        """
        min_gaps = self.forecaster_config.min_gaps_for_deciles
        yesterday = target_date - pd.Timedelta(days=1)
        one_week_ago = target_date - pd.Timedelta(days=7)

        def _fetch(day: date, h: int) -> np.ndarray:
            return np.sort(
                self.reservoir.arrivals_bin_df(day, h)[
                    "second_of_hour"
                ].to_numpy()
            )

        # Level 1: policy-window source days.
        if self.forecast_policy == ForecastPolicy.ONE_DAY:
            policy_arrivals = [_fetch(yesterday, hour)]
            policy_weights = [1.0]
        elif self.forecast_policy == ForecastPolicy.SEVEN_DAYS_FLAT:
            days = [one_week_ago + pd.Timedelta(days=d) for d in range(7)]
            policy_arrivals = [_fetch(d, hour) for d in days]
            policy_weights = [1.0] * 7
        elif self.forecast_policy == ForecastPolicy.SAME_DAY_ONCE:
            source = (
                one_week_ago
                if one_week_ago >= self.reservoir.min_date
                else yesterday
            )
            policy_arrivals = [_fetch(source, hour)]
            policy_weights = [1.0]
        elif self.forecast_policy == ForecastPolicy.SAME_DAY_EXPONENTIAL:
            weight = 1.0
            day = target_date - pd.Timedelta(days=7)
            policy_arrivals = []
            policy_weights = []
            while day >= self.reservoir.min_date:
                policy_arrivals.append(_fetch(day, hour))
                policy_weights.append(weight)
                weight *= self.forecaster_config.decay_factor
                day -= pd.Timedelta(days=7)
            if not policy_arrivals:
                if (
                    target_date - pd.Timedelta(days=7)
                ) < self.reservoir.min_date:
                    # Date is before the reservoir range: fall back to yesterday.
                    policy_arrivals = [_fetch(yesterday, hour)]
                    policy_weights = [1.0]
                # else: date is in range but no same-weekday data; keep lists empty.
        else:
            raise ValueError(
                f"Unsupported forecast policy: {self.forecast_policy}"
            )

        result = self._compute_weighted_gap_deciles(
            policy_arrivals, policy_weights, min_gaps
        )
        if result is not None:
            return result

        # Level 2: all available history for this hour, equal weights.
        all_dates = self.reservoir.arrivals_df["date"].unique()
        all_arrivals = [_fetch(d, hour) for d in all_dates]
        result = self._compute_weighted_gap_deciles(
            all_arrivals, [1.0] * len(all_dates), min_gaps
        )
        if result is not None:
            return result

        # Level 3: union of hours h-1, h, h+1 across all available history.
        neighbor_hours = [h for h in (hour - 1, hour, hour + 1) if 0 <= h < 24]
        neighbor_arrivals = [
            _fetch(d, h) for h in neighbor_hours for d in all_dates
        ]
        return self._compute_weighted_gap_deciles(
            neighbor_arrivals, [1.0] * len(neighbor_arrivals), min_gaps
        )  # Returns None if still insufficient -> caller uses uniform fallback.

    @staticmethod
    def _compute_weighted_gap_deciles(
        per_day_arrivals: list[np.ndarray],
        per_day_weights: list[float],
        min_gaps_required: int,
        min_gap_s: float = 0.001,
        n_quantiles: int = 10,
    ) -> np.ndarray | None:
        """
        Compute n_quantiles+1 interarrival-gap decile boundaries from pooled
        historical arrivals.

        For each source day, gaps are the consecutive differences starting from 0
        (i.e. the gap from the start of the hour to the first arrival is included).
        Each gap in day k is assigned weight w_k / N_k so that day k's total
        contribution to the pool equals w_k exactly.

        Returns an array of shape (n_quantiles + 1,), or None when the total gap
        count is below min_gaps_required.
        """
        all_gaps: list[float] = []
        all_weights: list[float] = []

        for arrivals, day_weight in zip(per_day_arrivals, per_day_weights):
            if len(arrivals) == 0:
                continue
            sorted_arrivals = np.sort(arrivals)
            gaps = np.diff(np.concatenate([[0.0], sorted_arrivals]))
            gaps = np.clip(gaps, min_gap_s, 3600.0)
            n_gaps = len(gaps)
            gap_weight = (
                day_weight / n_gaps
            )  # each gap shares this day's weight equally
            all_gaps.extend(float(g) for g in gaps)
            all_weights.extend([gap_weight] * n_gaps)

        if len(all_gaps) < min_gaps_required:
            return None

        gaps_arr = np.array(all_gaps)
        weights_arr = np.array(all_weights)

        # Weighted quantiles via sorted cumulative-weight interpolation.
        sort_idx = np.argsort(gaps_arr)
        sorted_gaps = gaps_arr[sort_idx]
        sorted_weights = weights_arr[sort_idx]
        cumw = np.cumsum(sorted_weights)
        cumw /= cumw[-1]  # normalize to [0, 1]
        # Prepend so that quantile 0 maps to the minimum observed gap.
        cumw_ext = np.concatenate([[0.0], cumw])
        sorted_gaps_ext = np.concatenate([[sorted_gaps[0]], sorted_gaps])

        quantile_levels = np.linspace(0.0, 1.0, n_quantiles + 1)
        deciles = np.interp(quantile_levels, cumw_ext, sorted_gaps_ext)
        # Guard against floating-point non-monotonicity.
        deciles = np.maximum.accumulate(deciles)
        return deciles

    def _sample_hour_from_deciles(
        self,
        deciles: np.ndarray,
        rng: np.random.Generator,
        min_gap_s: float = 0.001,
    ) -> list[float]:
        """
        Sample within-hour arrival offsets (seconds) from a gap decile model.

        Gaps are interpolated in log-space to handle right-skewed distributions.
        Arrivals accumulate until the running total reaches 3600 s.  The hourly
        count emerges naturally — it is not fixed in advance.
        """
        n_quantiles = len(deciles) - 1
        # Pre-log the boundaries; clip to avoid log(0).
        log_deciles = np.log(np.clip(deciles, min_gap_s, None))

        t = 0.0
        arrivals: list[float] = []
        safety_cap = self.forecaster_config.max_arrivals_per_hour_safety_cap
        for _ in range(safety_cap):
            u = rng.uniform(0.0, 1.0)
            k = min(int(u * n_quantiles), n_quantiles - 1)
            alpha = u * n_quantiles - k
            log_gap = log_deciles[k] + alpha * (
                log_deciles[k + 1] - log_deciles[k]
            )
            gap = float(np.exp(log_gap))
            t += gap
            if t >= 3600.0:
                break
            arrivals.append(t)
        else:
            logger.warning(
                "max_arrivals_per_hour_safety_cap (%d) reached. "
                "Consider increasing the cap or inspecting the gap distribution.",
                safety_cap,
            )
        return arrivals
