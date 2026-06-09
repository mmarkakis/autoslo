"""
slo_resolver.py
---------------
Resolves the effective SLO (in seconds) for a given query, supporting both a
single global SLO and a per-template override dict.
"""

from __future__ import annotations

import os

import yaml

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import SloResolverConfig
from autoslo.workload_definition.query import QueryTextId


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
    slo_tightening_factor:
        Optional factor to multiply all SLOs by, simulating tighter or looser
        SLOs without needing multiple YAML files.  A factor of 0.8 means
        "pretend SLOs are 80% of real". Applied to both the default and all
        overrides.
    """

    def __init__(
        self, config: SloResolverConfig, slo_tightening_factor: float = 1.0
    ) -> None:
        self._config = config
        self._slo_tightening_factor = slo_tightening_factor
        if slo_tightening_factor <= 0:
            raise ValueError(
                f"Tightening factor must be positive, "
                f"got {slo_tightening_factor}"
            )

        self._default = config.slo_s * slo_tightening_factor
        self._filename: str | None = config.slo_dict_filename
        self._dict: dict[str, float] = {}

        if self._filename:
            path = os.path.join(pu.get_data_path(), "slos", self._filename)
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            self._dict = {
                str(k).zfill(3): float(v) * slo_tightening_factor
                for k, v in raw["slo_dict"].items()
            }

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

    @property
    def slo_tightening_factor(self) -> float:
        return self._slo_tightening_factor

    # ------------------------------------------------------------------
    # derived resolvers
    # ------------------------------------------------------------------

    def tightened(self, slo_tightening_factor: float) -> "SloResolver":
        """Return a copy with all SLOs (default and overrides) scaled by *factor*.

        A *factor* of 0.8 means "pretend SLOs are 80% of real", causing
        the autoscaler to trigger earlier.
        """
        return SloResolver(
            config=self._config,
            slo_tightening_factor=slo_tightening_factor,
        )
