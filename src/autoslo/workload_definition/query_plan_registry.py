"""Per-schema query-plan registry.

Maps ``query_text_id`` values to their parsed query plans on a per-schema basis.
The mapping for each schema is loaded lazily on first access and kept in a
module-level cache so subsequent lookups are O(1).

Storage layout
--------------
``{data_root}/__query_plans/{schema_name}/query_plans.pkl``

The pickle file contains a ``dict[str, Any]`` keyed by ``query_text_id``.
Each value is the parsed plan dict produced by
:func:`~autoslo.query_plans.parse_plan.parse_one_plan`.

Usage
-----
>>> from autoslo.workload_definition.query_plan_registry import QueryPlanRegistry
>>> plan = QueryPlanRegistry.get(QueryTextId("ext_tpcds1000#42#001"))

To populate the registry for testing or new schemas without touching disk,
use :meth:`QueryPlanRegistry.register`:

>>> QueryPlanRegistry.register("my_schema", {"ext_tpcds1000#1#001": {...}})
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Optional

import autoslo.utils.paths as pu
from autoslo.workload_definition.query import QueryTextId

_REGISTRY_SUBDIR = "__query_plans"
_REGISTRY_FILENAME = "query_plans.pkl"

# Module-level lazy cache: schema_name → {query_text_id → plan dict}
_cache: dict[str, dict[str, Any]] = {}


class QueryPlanRegistry:
    """Lazily-loaded, per-schema registry mapping ``query_text_id`` to parsed plans."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, query_text_id: QueryTextId) -> Optional[Any]:
        """Return the parsed plan for *query_text_id*.

        Loads the schema's registry file on first access.  Returns ``None``
        if the schema has no registered plans or the id is not found.

        Parameters
        ----------
        query_text_id:
            The :class:`~autoslo.workload_definition.query.QueryTextId`
            identifying the query.  The schema is derived from
            ``query_text_id.schema_name``.

        Returns
        -------
        dict or None
            The parsed plan dict, or ``None`` if not found.
        """
        cls._ensure_loaded(query_text_id.schema_name)
        return _cache.get(query_text_id.schema_name, {}).get(query_text_id.value)

    @classmethod
    def register(cls, schema_name: str, mapping: dict[str, Any]) -> None:
        """Register an in-memory ``query_text_id`` → plan mapping.

        Any existing cached entry for *schema_name* is replaced.

        Parameters
        ----------
        schema_name:
            The schema identifier.
        mapping:
            Dictionary from ``query_text_id`` to parsed plan dict.
        """
        _cache[schema_name] = dict(mapping)

    @classmethod
    def update(
        cls,
        schema_name: str,
        new_entries: dict[str, Any],
        save: bool = True,
    ) -> None:
        """Merge *new_entries* into the registry for *schema_name*.

        Existing entries are preserved; only keys present in *new_entries*
        are added or overwritten.  If *save* is ``True``, the updated mapping
        is persisted to disk.

        Parameters
        ----------
        schema_name:
            The schema identifier.
        new_entries:
            New ``query_text_id`` → plan pairs to add.
        save:
            Whether to persist the updated mapping to disk.
        """
        cls._ensure_loaded(schema_name)
        _cache.setdefault(schema_name, {}).update(new_entries)
        if save:
            cls.save_schema(schema_name, _cache[schema_name])

    @classmethod
    def load_schema(cls, schema_name: str) -> dict[str, Any]:
        """Load and return the mapping for *schema_name* from disk.

        Parameters
        ----------
        schema_name:
            The schema identifier.

        Returns
        -------
        dict[str, Any]
            Mapping from ``query_text_id`` to parsed plan dict.

        Raises
        ------
        FileNotFoundError
            If the registry file for the schema does not exist.
        """
        path = cls._registry_path(schema_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No query-plan registry found for schema '{schema_name}' "
                f"at '{path}'."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def save_schema(cls, schema_name: str, mapping: dict[str, Any]) -> None:
        """Persist *mapping* to the standard registry path for *schema_name*.

        Creates intermediate directories if they don't exist.  After saving,
        the in-memory cache is updated.

        Parameters
        ----------
        schema_name:
            The schema identifier.
        mapping:
            Dictionary from ``query_text_id`` to parsed plan dict.
        """
        path = cls._registry_path(schema_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(mapping, f)
        _cache[schema_name] = dict(mapping)

    @classmethod
    def clear_cache(cls) -> None:
        """Evict all cached schema mappings (primarily useful in tests)."""
        _cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_loaded(cls, schema_name: str) -> None:
        """Populate the cache for *schema_name* from disk if not already done."""
        if schema_name not in _cache:
            try:
                _cache[schema_name] = cls.load_schema(schema_name)
            except FileNotFoundError:
                # Silently record an empty mapping so we don't repeat the
                # failing disk hit on every subsequent call.
                _cache[schema_name] = {}

    @classmethod
    def _registry_path(cls, schema_name: str) -> str:
        return os.path.join(
            pu.get_data_path(),
            _REGISTRY_SUBDIR,
            schema_name,
            _REGISTRY_FILENAME,
        )
