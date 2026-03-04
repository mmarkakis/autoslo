from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoslo.workload_definition.query import Query


# Columns that every workload file is expected to provide.
WORKLOAD_SCHEMA_COLUMNS: list[str] = [
    "workload_id",
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
    The ``workload_id`` column is uniform across all rows and serves as the
    workload's :attr:`name`.

    The class can also be used as a base class.  Subclasses that manage their
    own data store should override :attr:`name` and :meth:`queries`; they do
    not need to call ``super().__init__()``.
    """

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        """
        Parameters
        ----------
        df:
            A :class:`~pandas.DataFrame` matching the ``workload`` schema.
            Pass ``None`` only when constructing a subclass that supplies its
            own data.
        """
        self._df = df
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
    def name(self) -> str:
        """The workload identifier, taken from the ``workload_id`` column."""
        if self._df is None:
            raise NotImplementedError(
                f"{type(self).__name__} must either be backed by a DataFrame "
                "or override 'name'."
            )
        return str(self._df["workload_id"].iloc[0])

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
                    abs_start_time=row["abs_start_time"],
                )
            )
        self._queries_cache = result
        return result

    # ------------------------------------------------------------------
    # DataFrame access
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """The underlying :class:`~pandas.DataFrame`.

        Raises
        ------
        RuntimeError
            If this workload instance is not backed by a DataFrame.
        """
        if self._df is None:
            raise RuntimeError(
                f"{type(self).__name__} is not backed by a DataFrame."
            )
        return self._df