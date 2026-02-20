"""
slo_resolver.py
---------------
Resolves the effective SLO (in seconds) for a given query, supporting both a
single global SLO and a per-template override dict.

Usage
-----
    from autoslo.blueprint_selection.slo_resolver import SloResolver

    # From a filename stored in config.yml / experiment_meta.json:
    resolver = SloResolver(default_slo_s=10.0, slo_dict_filename="slo_dict.yml")

    # From an already-loaded dict (e.g. inlined in config):
    resolver = SloResolver.from_dict(default_slo_s=10.0, slo_dict={1: 5.0, 2: 8.0})

    # Resolve per query:
    slo = resolver.resolve("3_7")   # returns override for template 3, or default
"""
from __future__ import annotations

import os

import yaml

import autoslo.utils.paths as pu
from autoslo.workload_definition.query import Query


class SloResolver:
    """Resolves the effective SLO for a query given a global default and
    optional per-template overrides.

    Parameters
    ----------
    default_slo_s:
        Fallback SLO used when the query's template has no override.
    slo_dict_filename:
        Filename (not full path) of a YAML file under
        ``data/generation_parameters/`` mapping template IDs (int) to SLOs
        (float).  If *None*, only the global default is used.
    """

    def __init__(
        self,
        default_slo_s: float,
        slo_dict_filename: str | None = None,
    ) -> None:
        self._default = default_slo_s
        self._dict: dict[int, float] = {}
        self._filename: str | None = slo_dict_filename

        if slo_dict_filename:
            path = os.path.join(
                pu.get_data_path(), "generation_parameters", slo_dict_filename
            )
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            self._dict = {int(k): float(v) for k, v in raw.items()}

    # ------------------------------------------------------------------
    # alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        default_slo_s: float,
        slo_dict: dict[int, float],
        slo_dict_filename: str | None = None,
    ) -> "SloResolver":
        """Construct directly from an already-loaded mapping (e.g. inlined in
        a config or experiment_meta.json) without touching the filesystem."""
        inst = cls.__new__(cls)
        inst._default = default_slo_s
        inst._dict = {int(k): float(v) for k, v in slo_dict.items()}
        inst._filename = slo_dict_filename
        return inst

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def resolve(self, tpcds_temp_and_q_idx: str | None) -> float:
        """Return the SLO in seconds for the given query identifier.

        Falls back to *default_slo_s* when the template has no override,
        *tpcds_temp_and_q_idx* is *None*, or its value cannot be parsed
        (e.g. a float NaN from a DataFrame join).
        """
        if tpcds_temp_and_q_idx is None or not self._dict:
            return self._default
        try:
            tid = Query.template_id(tpcds_temp_and_q_idx)
        except (ValueError, AttributeError, TypeError):
            return self._default
        return self._dict.get(tid, self._default)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def default_slo_s(self) -> float:
        return self._default

    @property
    def slo_dict(self) -> dict[int, float]:
        """Returns the per-template override dict (may be empty)."""
        return dict(self._dict)

    @property
    def slo_dict_filename(self) -> str | None:
        return self._filename

    def has_overrides(self) -> bool:
        """True when at least one per-template override is present."""
        return bool(self._dict)
