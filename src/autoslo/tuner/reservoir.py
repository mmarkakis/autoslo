"""QueryReservoir — stores historical query arrivals for workload sampling."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

import autoslo.utils.config as cfgu
from autoslo.workload_definition.workload import Workload

console = Console()


logger = logging.getLogger(__name__)


class QueryReservoir:
    """
    A reservoir of historical query arrivals indexed by (day_of_week, hour).
    """

    BIN_DF_COLUMNS = ["date", "hour", "query_text_id", "count"]

    def __init__(
        self,
        count_df: pd.DataFrame,
        schema_name: str = "ext_tpcds1000",
    ) -> None:

        # When loading, don't do anything else.
        self._count_df = count_df
        self._schema_name = schema_name

    @property
    def schema_name(self) -> str:
        return self._schema_name

    @property
    def min_date(self) -> date:
        return self._count_df["date"].min()

    @property
    def count_df(self) -> pd.DataFrame:
        return self._count_df

    def save(self, directory: Path) -> None:
        """
        Returns the paths to both files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        count_df_path = directory / "reservoir.parquet"
        self._count_df.to_parquet(count_df_path, index=False)

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

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> QueryReservoir:
        schema_name = cfgu.getd(cfg, "basic_config.schema_name", required=True)
        workload_name = cfgu.getd(
            cfg, "workload_config.workload_name", required=True
        )
        workload = Workload(workload_name, schema_name)
        start = cfgu.getd(cfg, "forecast_config.history_abs_start_time_start")
        end = cfgu.getd(cfg, "forecast_config.history_abs_start_time_end")
        workload.slice_by_abs_time(start=start, end=end)

        # Input parsing/validation.
        if workload.df.empty:
            raise ValueError("Cannot build reservoir from empty DataFrame.")

        # Set up bins.
        # Key is (date, hour_of_day), Monday is 0
        # Value is a dictionary from query_text_id to query count
        df = workload.df.copy()
        df["date"] = df["abs_start_time"].dt.date
        df["hour"] = df["abs_start_time"].dt.hour
        count_df = (
            df.groupby(["date", "hour", "query_text_id"])
            .size()
            .reset_index(name="count")
        )

        console.print(
            f"  Built reservoir based on workload {workload_name} over the "
            f"period {start} to {end}."
        )

        return cls(count_df=count_df, schema_name=schema_name)

    def bin_df(self, target_date: date, hour: int) -> pd.DataFrame:
        if not (0 <= hour < 24):
            raise ValueError(f"Invalid hour: {hour}. Must be in [0, 23].")

        mask = (self._count_df["date"] == target_date) & (
            self._count_df["hour"] == hour
        )
        return self._count_df.loc[mask].reset_index(drop=True)
