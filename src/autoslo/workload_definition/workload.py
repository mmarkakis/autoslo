from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from rich import print
from rich.table import Table

import autoslo.utils.paths as pu
from autoslo.workload_definition.query import Query, QueryTextId

# Columns that every workload file is expected to provide.
WORKLOAD_SCHEMA_COLUMNS: list[str] = [
    "query_id",
    "abs_start_time",
    "query_text_id",
    "repetition_id",
]


class Workload:
    """A workload backed by a file in the ``workload`` data-schema format.

    Each row in the backing ``DataFrame`` corresponds to one submitted query
    and must include the columns listed in :data:`WORKLOAD_SCHEMA_COLUMNS`.
    The ``workload_name`` column is uniform across all rows and serves as the
    workload's :attr:`name`.

    The class can also be used as a base class.  Subclasses that manage their
    own data store should override :attr:`name` and :meth:`queries`; they do
    not need to call ``super().__init__()``.
    """

    def __init__(
        self,
        workload_name: str,
        schema_name: str,
        df: pd.DataFrame | None = None,
    ) -> None:
        """
        Parameters
        ----------
        workload_name:
            The name of the workload.
        schema_name:
            The name of the schema.
        df:
            Optional DataFrame to use directly instead of loading from disk.
            Must contain all columns listed in :data:`WORKLOAD_SCHEMA_COLUMNS`.
            When *None* (default), the workload is loaded from the standard
            file path under the ``__workloads`` data directory.

        Raises
        ------
        ValueError
            If the DataFrame (loaded or supplied) is missing any of the
            required columns from :data:`WORKLOAD_SCHEMA_COLUMNS`.
        FileNotFoundError
            If *df* is *None* and no parquet file exists at the expected path.
        """
        self._workload_name = workload_name
        self._schema_name = schema_name

        if df is not None:
            self._df = df.copy()
        else:
            path = os.path.join(
                pu.get_data_path(),
                "__workloads",
                schema_name,
                f"{workload_name}.parquet",
            )
            self._df = pd.read_parquet(path)

        for col in WORKLOAD_SCHEMA_COLUMNS:
            if col not in self._df.columns:
                raise ValueError(
                    f"DataFrame is missing required column {col!r} from "
                    f"WORKLOAD_SCHEMA_COLUMNS."
                )

        self.set_rel_start_times_from_abs()

        self._queries_cache: list[Query] | None = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @property
    def workload_name(self) -> str:
        """The workload identifier, taken from the ``workload_name`` column."""
        return self._workload_name

    @property
    def df(self) -> pd.DataFrame:
        """
        The underlying :class:`~pandas.DataFrame`.
        """

        return self._df

    def queries(self) -> list[Query]:
        """Return the list of :class:`~autoslo.workload_definition.query.Query`
        objects derived from the workload rows.

        Each row contributes exactly one :class:`Query`.  Optional execution
        fields (latency, featurization, …) are left at their defaults; callers
        should populate them after a run.
        """
        if self._queries_cache is not None:
            return self._queries_cache
        if self._df is None:
            raise NotImplementedError(
                f"{type(self).__name__} must either be backed by a DataFrame "
                "or override 'queries'."
            )
        result: list[Query] = []
        for _, row in self._df.iterrows():
            result.append(
                Query(
                    query_id=str(row["query_id"]),
                    query_text_id=QueryTextId(row["query_text_id"]),
                    repetition_id=str(row.get("repetition_id", "")),
                    rel_start_time_s=float(row.get("rel_start_time_s", -1)),
                )
            )
        self._queries_cache = result
        return result

    # ------------------------------------------------------------------
    # Start time manipulation
    # ------------------------------------------------------------------

    def set_rel_start_times_from_abs(self) -> None:
        """
        Create a column of relative start times (``rel_start_time_s``) derived
        from absolute start times (``abs_start_time``) using epoch timestamps.

        This is a mutating operation that adds or overwrites the
        ``rel_start_time_s`` column in the backing DataFrame.

        """
        self._df["rel_start_time_s"] = self._df["abs_start_time"].apply(
            lambda t: t.timestamp()
        )
        self._queries_cache = None

    def set_rel_start_times_from_zero(self) -> None:
        """
        Create a column of relative start times (``rel_start_time_s``) derived
        from absolute start times (``abs_start_time``) by rebasing to zero.

        This is a mutating operation that adds or overwrites the
        ``rel_start_time_s`` column in the backing DataFrame.
        """
        min_timestamp = self._df["abs_start_time"].min().timestamp()
        self._df["rel_start_time_s"] = self._df["abs_start_time"].apply(
            lambda t: t.timestamp() - min_timestamp
        )
        self._queries_cache = None

    def rescale_rel_start_times(self, factor: float) -> None:
        """
        Rescale the relative start times (``rel_start_time_s``) by a constant
        factor.

        This is a mutating operation that modifies the existing
        ``rel_start_time_s`` column in the backing DataFrame.  Absolute start
        times are left unchanged.

        Parameters
        ----------
        factor:
            The constant factor by which to multiply all relative start times.
        """
        self._df["rel_start_time_s"] = self._df["rel_start_time_s"] * factor
        self._queries_cache = None

    def slice_by_abs_time(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        """
        Filter the workload to only include queries whose ``abs_start_time``
        falls within [*start*, *end*] (both bounds inclusive and optional).

        Timestamps are parsed with :func:`pandas.Timestamp` and compared
        against the ``abs_start_time`` column.  If the column is
        timezone-aware and the supplied string has no timezone, UTC is
        assumed.  The DataFrame index is reset after filtering.

        This is a mutating operation; ``rel_start_time_s`` values are *not*
        updated — call :meth:`set_rel_start_times_from_zero` afterwards if
        you want them rebased to the new first query.

        Parameters
        ----------
        start:
            ISO 8601 string for the lower bound (inclusive).  ``None`` means
            no lower bound.
        end:
            ISO 8601 string for the upper bound (inclusive).  ``None`` means
            no upper bound.
        """
        tz = self._df["abs_start_time"].dt.tz

        def _parse(ts_str: str) -> pd.Timestamp:
            ts = pd.Timestamp(ts_str)
            if tz is not None and ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            elif tz is None and ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts

        mask = pd.Series(True, index=self._df.index)
        if start is not None:
            mask &= self._df["abs_start_time"] >= _parse(start)
        if end is not None:
            mask &= self._df["abs_start_time"] <= _parse(end)

        self._df = self._df[mask].reset_index(drop=True)
        self._queries_cache = None

    def save(self, overwrite: bool = False) -> Path:
        """Persist the workload DataFrame to the standard file path.

        The file is written to
        ``<data_root>/__workloads/<schema_name>/<workload_name>.parquet``.
        Parent directories are created automatically.

        Parameters
        ----------
        overwrite:
            If *False* (default) and the file already exists, raises
            :class:`FileExistsError`.  Set to *True* to overwrite.

        Returns
        -------
        Path
            The path of the written file.

        Raises
        ------
        FileExistsError
            If a file already exists at the target path and *overwrite* is
            *False*.
        """
        path = (
            Path(pu.get_data_path())
            / "__workloads"
            / self._schema_name
            / f"{self._workload_name}.parquet"
        )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Workload file already exists at {path}. "
                "Pass overwrite=True to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(path, index=False)
        return path

    def print_summary(self) -> None:
        """Print a summary of the workload using rich."""
        self.print_summary_from_df(self._df)

    @staticmethod
    def print_summary_from_df(workload_df):
        """Print a summary of the workload DataFrame using rich."""
        stats_table = Table(title="Workload Summary")

        stats_table.add_column("Stat", style="cyan", no_wrap=True)
        stats_table.add_column("Value", style="magenta")
        stats_table.add_row("Total Queries", str(len(workload_df)))
        num_unique_templates = (
            workload_df["query_text_id"]
            .apply(lambda x: QueryTextId(x).template_id)
            .nunique()
        )
        stats_table.add_row("Unique Query Templates", str(num_unique_templates))
        stats_table.add_row(
            "Unique Template+Query Index",
            str(workload_df["query_text_id"].nunique()),
        )
        stats_table.add_row(
            "Absolute Time Range",
            f"{workload_df['abs_start_time'].min()} to {workload_df['abs_start_time'].max()}",
        )
        stats_table.add_row(
            "Relative Time Range (seconds)",
            f"{workload_df['rel_start_time_s'].min()} to {workload_df['rel_start_time_s'].max()}",
        )
        stats_table.add_row(
            "Mean Inter-Arrival Time (seconds)",
            str(workload_df["abs_start_time"].diff().dt.total_seconds().mean()),
        )
        num_unique_days_with_queries = (
            workload_df["abs_start_time"].dt.normalize().nunique()
        )
        stats_table.add_row(
            "Mean Queries per Day",
            str(len(workload_df) / num_unique_days_with_queries),
        )
        print(stats_table)
