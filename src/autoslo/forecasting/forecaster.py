"""Forecast policies — determine how historical observations map to future bins."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import (
    ForecasterConfig,
    WorkloadConfig,
)
from autoslo.forecasting.forecast_policy import ForecastPolicy
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.workload_definition.workload import Workload


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
        if self.forecast_policy != ForecastPolicy.NONE:
            reservoir_config = forecaster_config.reservoir_config
            if reservoir_config is None:
                raise ValueError(
                    "reservoir_config must be provided for forecast policies other than 'none'"
                )
            self.reservoir = QueryReservoir(reservoir_config)

    def forecast(
        self,
        target_date: date | str,
        seed: int = 42,
        out_dir: Optional[Path | str] = None,
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

        rows = []
        query_idx = 0

        for i in range(24):
            bin_df = self._build_bin_df(target_date, i)
            if bin_df.empty:
                continue

            n_samples = self._n_samples(target_date, i, bin_df)
            if use_fixed_queries_per_hour:
                n_samples = self.forecaster_config.fixed_queries_per_hour
            if n_samples == 0:
                continue

            # Sample with replacement from the 'query_text_id' column, based on
            # the relative weights of the "count" column.
            sampled_ids = bin_df.sample(
                n=n_samples,
                weights="count",
                replace=True,
                random_state=seed,
            )["query_text_id"]

            # Given the number of arrivals, the arrival times for a Poisson
            # process are uniformly distributed within the hour.
            sampled_relative_start_times = list(
                np.sort(
                    np.random.default_rng(seed).uniform(0, 3600, size=n_samples)
                )
            )
            base_start_time = pd.Timestamp(target_date) + pd.Timedelta(hours=i)

            for query_text_id, rel_start_time in zip(
                sampled_ids, sampled_relative_start_times
            ):
                if rel_start_time > 3600:
                    break
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
        out_dir = (
            Path(out_dir)
            if out_dir is not None
            else Path(pu.get_data_path()) / "workloads"
        )
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
