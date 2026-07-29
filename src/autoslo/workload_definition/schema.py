"""Lightweight descriptor for a database schema used in workloads.

Each schema has a config YAML file at::

    {data_root}/schemas/{schema_name}.yml

Format::

    search_path: ext_tpcds1000
    # additional fields may be added here in the future

Usage::

    from autoslo.workload_definition.schema import Schema
    schema = Schema.load("ext_tpcds1000")
    print(schema.search_path)  # "ext_tpcds1000"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

import autoslo.filesystem.path_utils as pu


@dataclass
class Schema:
    """Configuration descriptor for a database schema.

    Attributes
    ----------
    name:
        The schema identifier (e.g. ``"ext_tpcds1000"``). This is also the
        stem of the YAML config file under ``data/schemas/``.
    search_path:
        The Postgres ``search_path`` to set when opening connections for this
        schema (e.g. ``"ext_tpcds1000"``).
    """

    name: str
    search_path: str

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, schema_name: str) -> "Schema":
        """Load the schema config from ``data/schemas/{schema_name}.yml``.

        Parameters
        ----------
        schema_name:
            The schema identifier.

        Returns
        -------
        Schema

        Raises
        ------
        FileNotFoundError
            If no config file exists for *schema_name*.
        """
        path = cls._config_path(schema_name)
        if not path.exists():
            raise FileNotFoundError(
                f"No schema config found for '{schema_name}'. "
                f"Expected: {path}"
            )
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            name=schema_name,
            search_path=data.get("search_path", schema_name),
        )

    def save(self) -> None:
        """Persist this schema's config to ``data/schemas/{name}.yml``."""
        path = self._config_path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump({"search_path": self.search_path}, f, sort_keys=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _config_path(cls, schema_name: str) -> Path:
        return pu.get_schemas_dir() / f"{schema_name}.yml"
