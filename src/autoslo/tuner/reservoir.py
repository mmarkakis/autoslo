"""QueryReservoir — stores historical query arrivals for workload sampling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class QueryReservoir:
    """A reservoir of historical query arrivals indexed by (day_of_week, hour).

    The backing :class:`~pandas.DataFrame` has columns:

    - ``day_of_week`` (int, 0 = Monday … 6 = Sunday)
    - ``hour`` (int, 0–23)
    - ``timestamp_within_hour`` (float, seconds 0–3600)
    - ``query_text_id`` (str)
    - ``repetition_id`` (str)

    A YAML sidecar (``reservoir_meta.yml``) stores metadata such as the
    schema name, time range, and per-template arrival-pattern classifications.
    """

    # Required DataFrame columns.
    COLUMNS = [
        "day_of_week",
        "hour",
        "timestamp_within_hour",
        "query_text_id",
        "repetition_id",
    ]

    def __init__(self, df: pd.DataFrame, meta: dict[str, Any]) -> None:
        missing = [c for c in self.COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Reservoir DataFrame missing columns: {missing}")
        self.df = df
        self.meta = meta

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        workloads: list,
        schema_name: str,
        use_repetition_id: bool = True,
    ) -> "QueryReservoir":
        """Build a reservoir from one or more historical workloads.

        Parameters
        ----------
        workloads :
            :class:`~autoslo.workload_definition.workload.Workload` objects
            whose ``abs_start_time`` column contains timezone-aware datetimes.
        schema_name :
            Schema identifier (e.g. ``"ext_tpcds1000"``).
        use_repetition_id :
            When *True*, use each row's ``repetition_id`` to group recurring
            query instances.  When *False* (or when the field is empty),
            fall back to ``query_text_id``.
        """
        rows: list[dict[str, Any]] = []

        for wl in workloads:
            df = wl.df
            for _, row in df.iterrows():
                dt = row["abs_start_time"]  # tz-aware datetime / Timestamp
                hour_floor = dt.replace(minute=0, second=0, microsecond=0)
                ts_within_hour = (dt - hour_floor).total_seconds()

                qtid = str(row["query_text_id"])

                rid_raw = row.get("repetition_id", "")
                rid = str(rid_raw) if (use_repetition_id and rid_raw) else qtid

                rows.append(
                    {
                        "day_of_week": dt.weekday(),
                        "hour": dt.hour,
                        "timestamp_within_hour": ts_within_hour,
                        "query_text_id": qtid,
                        "repetition_id": rid,
                    }
                )

        reservoir_df = pd.DataFrame(rows, columns=cls.COLUMNS)

        meta: dict[str, Any] = {
            "schema_name": schema_name,
            "num_workloads": len(workloads),
            "num_arrivals": len(reservoir_df),
            "classifications": {},
        }

        return cls(reservoir_df, meta)

    # ------------------------------------------------------------------
    # Arrival classification
    # ------------------------------------------------------------------

    def classify_arrivals(
        self,
        grouping_key: str = "repetition_id",
        min_samples: int = 10,
    ) -> dict[str, dict[str, Any]]:
        """Classify per-group arrival patterns using the forecasting detectors.

        Groups query arrivals by *grouping_key* and runs
        :class:`~autoslo.forecasting.windowed_template_detector.WindowedTemplateDetector`
        on each group.  Results are stored in ``self.meta["classifications"]``
        and returned.

        Parameters
        ----------
        grouping_key :
            Column to group by (``"repetition_id"`` or ``"query_text_id"``).
        min_samples :
            Minimum number of arrivals required for a group to be classified;
            groups below this threshold are labelled ``"too_few_samples"``.
        """
        from autoslo.forecasting.windowed_template_detector import (
            WindowedTemplateDetector,
        )
        from autoslo.workload_definition.query import Query, QueryTextId

        classifications: dict[str, dict[str, Any]] = {}

        for group_id, group_df in self.df.groupby(grouping_key):
            group_id_str = str(group_id)
            if len(group_df) < min_samples:
                classifications[group_id_str] = {
                    "classification": "too_few_samples",
                    "num_samples": len(group_df),
                }
                continue

            # Build lightweight Query objects just for the detector.
            queries = [
                Query(
                    query_id=f"res_{i}",
                    query_text_id=QueryTextId(row["query_text_id"]),
                    rel_start_time_s=float(row["timestamp_within_hour"]),
                )
                for i, (_, row) in enumerate(group_df.iterrows())
            ]

            detector = WindowedTemplateDetector(
                queries, min_samples=min_samples
            )
            result = detector.detect()

            if result.get("is_windowed"):
                classifications[group_id_str] = {
                    "classification": "windowed",
                    "num_samples": len(group_df),
                    "period_s": result.get("period_s"),
                    "active_length_s": result.get("active_length_s"),
                    "on_window_rel_start_s": result.get(
                        "on_window_rel_start_s"
                    ),
                }
            else:
                classifications[group_id_str] = {
                    "classification": "normal",
                    "num_samples": len(group_df),
                }

        self.meta["classifications"] = classifications
        return classifications

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> tuple[Path, Path]:
        """Write ``reservoir.parquet`` and ``reservoir_meta.yml``.

        Returns the paths to both files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        parquet_path = directory / "reservoir.parquet"
        self.df.to_parquet(parquet_path, index=False)

        meta_path = directory / "reservoir_meta.yml"
        with open(meta_path, "w") as f:
            yaml.dump(self.meta, f, default_flow_style=False, sort_keys=False)

        return parquet_path, meta_path

    @classmethod
    def load(cls, directory: Path) -> "QueryReservoir":
        """Load a reservoir from ``reservoir.parquet`` + ``reservoir_meta.yml``."""
        directory = Path(directory)
        df = pd.read_parquet(directory / "reservoir.parquet")
        with open(directory / "reservoir_meta.yml") as f:
            meta = yaml.safe_load(f) or {}
        return cls(df, meta)

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def query_rate_per_hour(
        self, day_of_week: int, hour: int
    ) -> float:
        """Return the mean query arrival rate (queries / hour) for a bin.

        Computed as the total number of arrivals for this
        ``(day_of_week, hour)`` bin divided by the number of distinct
        historical workloads that contributed to the reservoir.
        """
        mask = (self.df["day_of_week"] == day_of_week) & (
            self.df["hour"] == hour
        )
        count = int(mask.sum())
        n_workloads = max(1, self.meta.get("num_workloads", 1))
        return count / n_workloads

    def unique_query_text_ids(
        self, day_of_week: int, hour: int
    ) -> list[str]:
        """Return the distinct ``query_text_id`` values for a bin."""
        mask = (self.df["day_of_week"] == day_of_week) & (
            self.df["hour"] == hour
        )
        return sorted(self.df.loc[mask, "query_text_id"].unique().tolist())

    def bin_df(self, day_of_week: int, hour: int) -> pd.DataFrame:
        """Return the reservoir rows for a specific (day_of_week, hour) bin."""
        mask = (self.df["day_of_week"] == day_of_week) & (
            self.df["hour"] == hour
        )
        return self.df.loc[mask].reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Return a summary table of (day_of_week, hour) → count."""
        return (
            self.df.groupby(["day_of_week", "hour"])
            .size()
            .reset_index(name="count")
        )
