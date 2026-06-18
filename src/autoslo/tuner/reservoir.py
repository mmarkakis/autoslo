"""QueryReservoir — stores historical query arrivals for workload sampling."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from autoslo.config.component_configs import ReservoirConfig
from autoslo.tuner.tuner_console import console
from autoslo.workload_definition.workload import Workload

logger = logging.getLogger(__name__)


class QueryReservoir:
    """
    A reservoir of historical query arrivals indexed by (day_of_week, hour).
    """

    BIN_DF_COLUMNS = ["date", "hour", "query_text_id", "count"]
    ARRIVALS_DF_COLUMNS = ["date", "hour", "second_of_hour"]

    def __init__(
        self,
        reservoir_config: Optional[ReservoirConfig] = None,
        count_df: Optional[pd.DataFrame] = None,
        arrivals_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if (reservoir_config is None) == (count_df is None):
            raise ValueError(
                "Must specify exactly one of reservoir_config or count_df."
            )

        if reservoir_config is not None:
            workload_config = reservoir_config.to_workload_config()
            workload = Workload(workload_config=workload_config)
            df = workload.df.copy()

            # Input parsing/validation.
            if df.empty:
                raise ValueError("Cannot build reservoir from empty workload.")

            # Slice
            tz = df["abs_start_time"].dt.tz
            last_day_end = (
                pd.Timestamp(reservoir_config.last_day_date_inclusive)
                .normalize()
                .tz_localize(tz)
            ) + pd.Timedelta(days=1)
            first_day_start = last_day_end - pd.Timedelta(
                days=reservoir_config.num_days
            )
            df = df[
                (df["abs_start_time"] >= first_day_start)
                & (df["abs_start_time"] < last_day_end)
            ].reset_index(drop=True)

            # Set up bins.
            # Key is (date, hour_of_day), Monday is 0
            # Value is a dictionary from query_text_id to query count
            df["date"] = df["abs_start_time"].dt.date
            df["hour"] = df["abs_start_time"].dt.hour
            count_df = (
                df.groupby(["date", "hour", "query_text_id"])
                .size()
                .reset_index(name="count")
            )

            # Build per-arrival timing table from raw timestamps.
            _arrivals = df[["date", "hour"]].copy()
            _arrivals["second_of_hour"] = (
                df["abs_start_time"] - df["abs_start_time"].dt.floor("h")
            ).dt.total_seconds()
            arrivals_df = _arrivals[self.ARRIVALS_DF_COLUMNS].reset_index(drop=True)

            console.print(
                f"  Built reservoir based on workload "
                f"{reservoir_config.workload_name} based on "
                f"{reservoir_config.num_days} days of data ending on "
                f"{reservoir_config.last_day_date_inclusive} (inclusive)."
            )
        assert count_df is not None
        self._count_df = count_df
        self._arrivals_df = arrivals_df

    @property
    def min_date(self) -> date:
        return self._count_df["date"].min()

    @property
    def count_df(self) -> pd.DataFrame:
        return self._count_df

    @property
    def has_arrivals(self) -> bool:
        """True if per-arrival timing data is available."""
        return self._arrivals_df is not None

    @property
    def arrivals_df(self) -> pd.DataFrame:
        """The full arrivals DataFrame. Raises RuntimeError if unavailable."""
        if self._arrivals_df is None:
            raise RuntimeError(
                "Arrivals data is not available for this reservoir. "
                "Build the reservoir from a workload file or provide arrivals_df."
            )
        return self._arrivals_df

    def save(self, directory: Path) -> None:
        """
        Returns the paths to both files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        count_df_path = directory / "reservoir.parquet"
        self._count_df.to_parquet(count_df_path, index=False)

        if self._arrivals_df is not None:
            arrivals_path = directory / "reservoir_arrivals.parquet"
            self._arrivals_df.to_parquet(arrivals_path, index=False)

    @classmethod
    def load(cls, directory: Path) -> "QueryReservoir":
        directory = Path(directory)
        count_df_path = directory / "reservoir.parquet"
        if not count_df_path.exists():
            raise FileNotFoundError(
                f"Reservoir file not found at {count_df_path}"
            )
        count_df = pd.read_parquet(count_df_path)

        arrivals_df = None
        arrivals_path = directory / "reservoir_arrivals.parquet"
        if arrivals_path.exists():
            arrivals_df = pd.read_parquet(arrivals_path)

        return cls(count_df=count_df, arrivals_df=arrivals_df)

    def bin_df(self, target_date: date, hour: int) -> pd.DataFrame:
        if not (0 <= hour < 24):
            raise ValueError(f"Invalid hour: {hour}. Must be in [0, 23].")

        mask = (self._count_df["date"] == target_date) & (
            self._count_df["hour"] == hour
        )
        return self._count_df.loc[mask].reset_index(drop=True)

    def arrivals_bin_df(self, target_date: date, hour: int) -> pd.DataFrame:
        """
        Return the per-arrival second_of_hour values for the given (date, hour)
        bin. Raises RuntimeError if has_arrivals is False.
        """
        if not (0 <= hour < 24):
            raise ValueError(f"Invalid hour: {hour}. Must be in [0, 23].")
        if self._arrivals_df is None:
            raise RuntimeError(
                "Arrivals data is not available for this reservoir."
            )
        mask = (self._arrivals_df["date"] == target_date) & (
            self._arrivals_df["hour"] == hour
        )
        return self._arrivals_df.loc[mask].reset_index(drop=True)
