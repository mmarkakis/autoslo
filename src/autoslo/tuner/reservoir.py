"""QueryReservoir — stores historical query arrivals for workload sampling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from autoslo.workload_definition.workload import Workload
from datetime import datetime


logger = logging.getLogger(__name__)


class QueryReservoir:
    """
    A reservoir of historical query arrivals indexed by (day_of_week, hour).
    """

    BIN_DF_COLUMNS = ["date", "hour", "query_text_id", "count"]

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        workload: Optional[Workload] = None,
        count_df: Optional[pd.DataFrame] = None,
    ) -> None:

        # When loading, don't do anything else.
        self._count_df: pd.DataFrame
        self._schema_name = "ext_tpcds1000"
        if count_df is not None:
            self._count_df = count_df
            return

        # Input parsing/validation.
        if df is None:
            if workload is None:
                raise ValueError("Either `workload` or `df` must be provided.")
            else:
                df = workload.df
        if ("abs_start_time" not in df.columns) or (
            "query_text_id" not in df.columns
        ):
            raise ValueError(
                f"Columns `abs_start_time` and `query_text_id` are required in "
                f"the reservoir DataFrame."
            )
        if df.empty:
            raise ValueError("Cannot build reservoir from empty DataFrame.")

        # Set up bins.
        # Key is (date, hour_of_day), Monday is 0
        # Value is a dictionary from query_text_id to query count
        df["date"] = df["abs_start_time"].dt.date
        df["hour"] = df["abs_start_time"].dt.hour
        self._count_df = (
            df.groupby(["date", "hour", "query_text_id"])
            .size()
            .reset_index(name="count")
        )

    @property
    def schema_name(self) -> str:
        return self._schema_name

    @property
    def min_date(self) -> pd.Timestamp:
        return pd.to_datetime(self._count_df["date"].min()).date()

    @property
    def count_df(self) -> pd.DataFrame:
        return self._count_df

    def save(self, directory: Path) -> Path:
        """
        Returns the paths to both files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        count_df_path = directory / "reservoir.parquet"
        self._count_df.to_parquet(count_df_path, index=False)

        return count_df_path

    @classmethod
    def load(cls, directory: Path) -> "QueryReservoir":
        directory = Path(directory)
        count_df_path = directory / "reservoir.parquet"
        if not count_df_path.exists():
            raise FileNotFoundError(
                f"Reservoir file not found at {count_df_path}"
            )
        count_df = pd.read_parquet(count_df_path)

        return cls(count_df=count_df)

    def bin_df(self, date: pd.Timestamp|datetime, hour: int) -> pd.DataFrame:
        if not (0 <= hour < 24):
            raise ValueError(f"Invalid hour: {hour}. Must be in [0, 23].")

        date_normed = pd.to_datetime(date).date()

        mask = (self._count_df["date"] == date_normed) & (
            self._count_df["hour"] == hour
        )
        return self._count_df.loc[mask].reset_index(drop=True)
