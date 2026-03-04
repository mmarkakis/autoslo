from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoslo.workload_definition.query import Query


# Columns that every workload file is expected to provide.
WORKLOAD_SCHEMA_COLUMNS: list[str] = [
    "workload_name",
    "query_id",
    "abs_start_time",
    "schema_name",
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

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df:
            A :class:`~pandas.DataFrame` matching the ``workload`` schema.
            Pass ``None`` only when constructing a subclass that supplies its
            own data.

        Raises
        ------
        ValueError
            If *df* is not ``None`` and is missing any of the required columns
            from :data:`WORKLOAD_SCHEMA_COLUMNS`.
        ValueError
            If *df* contains multiple distinct values in the ``workload_name``
            column or the ``schema_name`` column, which are expected to be
            uniform across the workload.
        """
        self._df = df
        for col in WORKLOAD_SCHEMA_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"DataFrame is missing required column {col!r} from "
                    f"WORKLOAD_SCHEMA_COLUMNS."
                )

        if (
            len(df["workload_name"].unique()) > 1
            or len(df["schema_name"].unique()) > 1
        ):
            raise ValueError(
                "All rows in the workload DataFrame must share the same "
                "workload_name and schema_name, but found multiple: "
                f"workload_name: {df['workload_name'].unique()}, "
                f"schema_name: {df['schema_name'].unique()}"
            )
        self._workload_name = str(df["workload_name"].iloc[0])
        self._schema_name = str(df["schema_name"].iloc[0])
        self.set_rel_start_times_from_abs()

        self._queries_cache: list[Query] | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Workload":
        """Load a workload from a Parquet file that follows the workload schema.

        Parameters
        ----------
        path:
            Absolute or relative path to the Parquet file.

        Returns
        -------
        Workload
            A :class:`Workload` instance backed by the file's contents.
        """
        return cls(pd.read_parquet(path))

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @property
    def workload_name(self) -> str:
        """The workload identifier, taken from the ``workload_name`` column."""
        return self._workload_name

    @property
    def schema_name(self) -> str:
        """The schema name, taken from the ``schema_name`` column."""
        return self._schema_name

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
                    query_text_id=str(row["query_text_id"]),
                    schema_name=str(row.get("schema_name", "")),
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
