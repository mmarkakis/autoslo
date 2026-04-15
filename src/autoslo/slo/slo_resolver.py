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
    resolver = SloResolver.from_dict(default_slo_s=10.0, slo_dict={"001": 5.0, "002": 8.0})

    # Resolve per query:
    slo = resolver.resolve("3_7")   # returns override for template 3, or default
"""

from __future__ import annotations

import os

import yaml

import autoslo.utils.paths as pu
from autoslo.workload_definition.query import QueryTextId

import autoslo.utils.config as cfgu


class SloResolver:
    """Resolves the effective SLO for a query given a global default and
    optional per-template overrides.

    Parameters
    ----------
    default_slo_s:
        Fallback SLO used when the query's template has no override.
    slo_dict_filename:
        Filename (not full path) of a YAML file under
        ``data/slos/`` mapping template IDs (zero-padded str) to SLOs
        (float).  If *None*, only the global default is used.
    """

    def __init__(
        self,
        default_slo_s: float,
        slo_dict_filename: str | None = None,
    ) -> None:
        self._default = default_slo_s
        self._dict: dict[str, float] = {}
        self._filename: str | None = slo_dict_filename

        if slo_dict_filename:
            path = os.path.join(pu.get_data_path(), "slos", slo_dict_filename)
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            self._dict = {
                str(k).zfill(3): float(v) for k, v in raw["slo_dict"].items()
            }

    # ------------------------------------------------------------------
    # alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        default_slo_s: float,
        slo_dict: dict[str, float],
        slo_dict_filename: str | None = None,
    ) -> "SloResolver":
        """Construct directly from an already-loaded mapping (e.g. inlined in
        a config or experiment_meta.json) without touching the filesystem."""
        inst = cls.__new__(cls)
        inst._default = default_slo_s
        inst._dict = {str(k).zfill(3): float(v) for k, v in slo_dict.items()}
        inst._filename = slo_dict_filename
        return inst

    @classmethod
    def from_config(cls, config: dict) -> "SloResolver":
        """Construct from a config dict (e.g. the one loaded from config.yml or
        experiment_meta.json).  Looks for keys "default_slo_s" and
        "slo_dict_filename" in the top-level dict and/or under "slo_config"."""
        default_slo_s = config.get("default_slo_s") or cfgu.getd(
            config, "slo_config.default_slo_s", 10.0
        )
        slo_dict_filename = config.get("slo_dict_filename") or cfgu.getd(
            config, "slo_config.slo_dict_filename"
        )
        return cls(
            default_slo_s=default_slo_s, slo_dict_filename=slo_dict_filename
        )

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------

    def resolve(self, query_text_id: "QueryTextId | str | None") -> float:
        """Return the SLO in seconds for the given query identifier.

        Accepts any of:

        * a :class:`QueryTextId` object,
        * a ``"schema#template#index"`` string (from structured logs),
        * a ``"template_index"`` string (e.g. ``"042_001"`` from
          ``tpcds_temp_and_q_idx``),
        * ``None``.

        The lookup key is the **template ID** (int).  All variants of
        the same template share a single SLO.

        Falls back to *default_slo_s* when the template has no override,
        *query_text_id* is *None*, or its value cannot be parsed
        (e.g. a float NaN from a DataFrame join).
        """
        if query_text_id is None or not self._dict:
            return self._default
        try:
            if isinstance(query_text_id, QueryTextId):
                tid = query_text_id.template_id
            elif isinstance(query_text_id, str):
                if "#" in query_text_id:
                    # "ext_tpcds1000#042#001" → template "042"
                    tid = query_text_id.split("#")[1]
                elif "_" in query_text_id:
                    # "042_001" → template "042"
                    tid = query_text_id.split("_")[0]
                else:
                    tid = query_text_id
            else:
                return self._default
        except (ValueError, AttributeError, TypeError, IndexError):
            return self._default
        return self._dict.get(tid, self._default)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def default_slo_s(self) -> float:
        return self._default

    @property
    def slo_dict(self) -> dict[str, float]:
        """Returns the per-template override dict (may be empty)."""
        return dict(self._dict)

    @property
    def slo_dict_filename(self) -> str | None:
        return self._filename

    def has_overrides(self) -> bool:
        """True when at least one per-template override is present."""
        return bool(self._dict)

    # ------------------------------------------------------------------
    # derived resolvers
    # ------------------------------------------------------------------

    def tightened(self, factor: float) -> "SloResolver":
        """Return a copy with all SLOs (default and overrides) scaled by *factor*.

        A *factor* of 0.8 means "pretend SLOs are 80% of real", causing
        the autoscaler to trigger earlier.
        """
        if factor <= 0:
            raise ValueError(f"Tightening factor must be positive, got {factor}")
        return SloResolver.from_dict(
            default_slo_s=self._default * factor,
            slo_dict={k: v * factor for k, v in self._dict.items()},
            slo_dict_filename=self._filename,
        )

