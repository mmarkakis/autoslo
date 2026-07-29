"""Per-schema query-text registry.

Maps ``query_text_id`` values to their SQL strings on a per-schema basis.
The mapping for each schema is loaded lazily on first access and kept in a
module-level cache so subsequent lookups are O(1).

Storage layout
--------------
``{data_root}/__query_texts/{schema_name}/query_texts.parquet``

The Parquet file must contain at least two columns:

* ``query_text_id`` (``str``) – the opaque key used in the workload schema.
* ``query_text``    (``str``) – the SQL text for that key.

Usage
-----
>>> from autoslo.workload_definition.query_text_registry import QueryTextRegistry
>>> sql = QueryTextRegistry.get("ext_tpcds1000", "42_001")

To populate the registry for testing or new schemas without writing a
Parquet file, use :meth:`QueryTextRegistry.register`:

>>> QueryTextRegistry.register("my_schema", {"q1": "SELECT 1", "q2": "SELECT 2"})
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import autoslo.filesystem.path_utils as pu
from autoslo.workload_definition.query import QueryTextId

_REGISTRY_SUBDIR = "__query_texts"
_REGISTRY_FILENAME = "query_texts.parquet"

# Module-level lazy cache: schema_name → {query_text_id → sql}
_cache: dict[str, dict[str, str]] = {}


class QueryTextRegistry:
    """Lazily-loaded, per-schema registry mapping ``query_text_id`` to SQL."""

    def __init__(self, schema_name: str, one_statement_per_query: bool = True):
        self.schema_name = schema_name
        self.one_statement_per_query = one_statement_per_query
        self._ensure_loaded(schema_name)

    def get(self, query_text_id: QueryTextId | str) -> Optional[str]:
        """Return the SQL text for *query_text_id* within this registry's schema.

        Returns ``None`` if the id is not found.

        Parameters
        ----------
        query_text_id:
            The opaque key for the desired query text.

        Returns
        -------
        str or None
            The SQL string, or ``None`` if not found.
        """
        if isinstance(query_text_id, QueryTextId):
            query_text_id = str(query_text_id)
        return _cache.get(self.schema_name, {}).get(query_text_id)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def get_from_schema(
        cls, schema_name: str, query_text_id: QueryTextId | str
    ) -> Optional[str]:
        """Return the SQL text for *query_text_id* within *schema_name*.

        Loads the schema's registry file on first access.  Returns ``None``
        if the schema has no registered texts or the id is not found.

        Parameters
        ----------
        schema_name:
            The schema identifier (e.g. ``"ext_tpcds1000"``).
        query_text_id:
            The opaque key for the desired query text.

        Returns
        -------
        str or None
            The SQL string, or ``None`` if not found.
        """
        if isinstance(query_text_id, QueryTextId):
            query_text_id = str(query_text_id)
        cls._ensure_loaded(schema_name)
        return _cache.get(schema_name, {}).get(query_text_id)

    @classmethod
    def register(cls, schema_name: str, mapping: dict[str, str]) -> None:
        """Register an in-memory ``query_text_id`` → SQL mapping.

        This is useful for testing or for schemas whose texts are generated
        programmatically and should not be persisted to disk.  Any existing
        cached entry for *schema_name* is replaced.

        Parameters
        ----------
        schema_name:
            The schema identifier.
        mapping:
            Dictionary from ``query_text_id`` to SQL string.
        """
        _cache[schema_name] = dict(mapping)

    @classmethod
    def load_schema(
        cls, schema_name: str, one_statement_per_query: bool = True
    ) -> dict[str, str]:
        """Load and return the mapping for *schema_name* from disk.

        Reads
        ``{data_root}/__query_texts/{schema_name}/query_texts.parquet``.

        Parameters
        ----------
        schema_name:
            The schema identifier.
        one_statement_per_query:
            Whether to enforce one statement per query.

        Returns
        -------
        dict[str, str]
            Mapping from ``query_text_id`` to SQL string.

        Raises
        ------
        FileNotFoundError
            If the registry file for the schema does not exist.
        """
        path = cls._registry_path(schema_name)
        if not path.exists():
            raise FileNotFoundError(
                f"No query-text registry found for schema '{schema_name}'. "
                f"Expected: {path}"
            )
        df = pd.read_parquet(path, columns=["query_text_id", "query_text"])
        texts = df["query_text"].astype(str)
        if one_statement_per_query:
            texts = texts.apply(
                lambda t: f"{t.split(';')[0]};" if ";" in t else t
            )
        return dict(zip(df["query_text_id"].astype(str), texts))

    @classmethod
    def save_schema(cls, schema_name: str, mapping: dict[str, str]) -> None:
        """Persist *mapping* to the standard registry path for *schema_name*.

        Creates intermediate directories if they don't exist.  After saving,
        the in-memory cache is updated.

        Parameters
        ----------
        schema_name:
            The schema identifier.
        mapping:
            Dictionary from ``query_text_id`` to SQL string.
        """
        path = cls._registry_path(schema_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            list(mapping.items()), columns=["query_text_id", "query_text"]
        )
        df.to_parquet(path, index=False)
        _cache[schema_name] = dict(mapping)

    @classmethod
    def clear_cache(cls) -> None:
        """Evict all cached schema mappings (primarily useful in tests)."""
        _cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self, schema_name: str) -> None:
        """Populate the cache for *schema_name* from disk if not already done."""
        if schema_name not in _cache:
            try:
                _cache[schema_name] = self.load_schema(
                    schema_name, self.one_statement_per_query
                )
            except FileNotFoundError:
                # Create the summary file from individual query text files if
                # possible, then retry loading.
                self._create_registry_summary_file(schema_name)
                _cache[schema_name] = self.load_schema(
                    schema_name, self.one_statement_per_query
                )

    @classmethod
    def _registry_path(cls, schema_name: str) -> Path:
        return (
            pu.get_data_dir()
            / _REGISTRY_SUBDIR
            / schema_name
            / _REGISTRY_FILENAME
        )

    @classmethod
    def _create_registry_summary_file(cls, schema_name: str) -> None:
        """Helper to create a registry Parquet file from individual
        query text files for a schema.  Expects the files to be in
        ``{data_root}/__query_texts/{schema_name}/`` with names like
        ``{query_text_id}.sql`` containing the SQL text.

        Parameters
        ----------
        schema_name:
            The schema identifier.

        """
        dir_path = pu.get_data_dir() / _REGISTRY_SUBDIR / schema_name
        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"No directory found for schema '{schema_name}' at '{dir_path}'."
            )
        mapping = {}
        for entry in dir_path.iterdir():
            if entry.suffix == ".sql" and entry.is_file():
                query_text_id = entry.stem
                with open(entry, "r") as f:
                    mapping[query_text_id] = f.read()
        cls.save_schema(schema_name, mapping)
