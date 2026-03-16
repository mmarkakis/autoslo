"""
forecast_loader.py
------------------
Load offline forecast distributions and SLO-tightness tables, and expose
them as :class:`~autoslo.routing.cache_risk_scorer.FutureQueryMix` objects
for the cache-aware routing policy.

Two YAML files are consumed:

* **Forecast distribution** (``data/forecast_distributions/{name}.yml``)
  — per ``(day_of_week, hour)`` bin, a probability distribution over
  query templates likely to arrive soon.

* **SLO tightness** (``data/slo_tightness/{name}.yml``)
  — per template, a scalar in [0, 1] measuring how close the isolated
  prediction is to the SLO (1 = very tight).

At construction time the loader also pre-computes per-template table-access
vectors using the :class:`IconqQueryFeaturizer`, so that
:meth:`get_future_query_mix` is allocation-free at routing time.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import yaml

import autoslo.utils.paths as pu
from autoslo.routing.cache_risk_scorer import FutureQueryMix
from autoslo.workload_definition.query import QueryTextId


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _bin_key(day_of_week: int, hour: int) -> tuple[int, int]:
    return (day_of_week, hour)


# ---------------------------------------------------------------------------
# ForecastDistributionLoader
# ---------------------------------------------------------------------------


class ForecastDistributionLoader:
    """Materialise :class:`FutureQueryMix` from offline YAML artefacts.

    Parameters
    ----------
    forecast_distribution_path :
        Absolute or relative path to the forecast distribution YAML.
        If relative, resolved under ``data/forecast_distributions/``.
    slo_tightness_path :
        Absolute or relative path to the SLO-tightness YAML.
        If relative, resolved under ``data/slo_tightness/``.
    iconq_query_featurizer :
        A loaded featurizer used to look up per-template table vectors.
    n_table_dims :
        Number of table dimensions in the featurization (``_n``).
        Used when a template has no cached featurization (zero-vector
        fallback).
    m_operator_dims :
        Number of operator pairs (``_m``) — needed to slice table dims
        from the full featurization vector.
    fallback_tightness :
        Default tightness for templates present in the forecast but missing
        from the tightness table (default 0.5).
    """

    def __init__(
        self,
        forecast_distribution_path: str,
        slo_tightness_path: str,
        iconq_query_featurizer: Any,
        n_table_dims: int,
        m_operator_dims: int,
        fallback_tightness: float = 0.5,
    ) -> None:
        self._featurizer = iconq_query_featurizer
        self._n = n_table_dims
        self._m = m_operator_dims
        self._fallback_tightness = fallback_tightness

        # -- Load YAMLs -----------------------------------------------------
        forecast_path = self._resolve_path(
            forecast_distribution_path, "forecast_distributions"
        )
        tightness_path = self._resolve_path(
            slo_tightness_path, "slo_tightness"
        )
        forecast_cfg = _load_yaml(forecast_path)
        tightness_cfg = _load_yaml(tightness_path)

        self._schema_name: str = forecast_cfg["schema_name"]

        # -- Parse tightness table ------------------------------------------
        self._tightness: dict[str, float] = {}
        for tid, entry in tightness_cfg.get("entries", {}).items():
            self._tightness[str(tid)] = float(entry["tightness"])

        # -- Pre-compute table vectors per template -------------------------
        self._table_vectors: dict[str, np.ndarray] = {}
        self._precompute_table_vectors(forecast_cfg)

        # -- Build per-bin FutureQueryMix -----------------------------------
        self._bins: dict[tuple[int, int], FutureQueryMix] = {}
        self._build_bins(forecast_cfg)

        # -- Default (uniform) mix for bins not in the YAML -----------------
        self._default_mix = self._build_default_mix()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(raw: str, subdirectory: str) -> str:
        if os.path.isabs(raw):
            return raw
        # Try as-is first (could be project-relative).
        if os.path.exists(raw):
            return raw
        # Fall back to data/{subdirectory}/{raw}.
        return os.path.join(pu.get_data_path(), subdirectory, raw)

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _extract_table_vector(self, template_id: str) -> np.ndarray:
        """Get the table-access vector (length N) for a template.

        Tries every query index ``000`` through ``009`` in the featurizer
        cache; returns the first hit.  Falls back to a zero vector.
        """
        for qi in range(10):
            qtid = QueryTextId(
                value=f"{self._schema_name}#{template_id}#{qi:03d}"
            )
            try:
                feat = self._featurizer.featurize_from_query_text_id(qtid)
                return np.array(feat[2 * self._m :], dtype=np.float64)
            except (ValueError, KeyError):
                continue
        return np.zeros(self._n, dtype=np.float64)

    def _precompute_table_vectors(self, forecast_cfg: dict) -> None:
        """Collect all unique template IDs across bins and pre-compute their
        table vectors."""
        all_tids: set[str] = set()
        for bin_entry in forecast_cfg.get("bins", []):
            for t in bin_entry.get("templates", []):
                all_tids.add(str(t["template_id"]))
        for tid in all_tids:
            self._table_vectors[tid] = self._extract_table_vector(tid)

    # ------------------------------------------------------------------
    # Bin construction
    # ------------------------------------------------------------------

    def _build_bins(self, forecast_cfg: dict) -> None:
        for bin_entry in forecast_cfg.get("bins", []):
            dow = int(bin_entry["day_of_week"])
            hour = int(bin_entry["hour"])
            templates = bin_entry.get("templates", [])
            if not templates:
                continue

            tids: list[str] = []
            probs: list[float] = []
            tvecs: list[np.ndarray] = []
            tights: list[float] = []

            for t in templates:
                tid = str(t["template_id"])
                tids.append(tid)
                probs.append(float(t["probability"]))
                tvecs.append(self._table_vectors.get(
                    tid, np.zeros(self._n, dtype=np.float64)
                ))
                tights.append(
                    self._tightness.get(tid, self._fallback_tightness)
                )

            self._bins[_bin_key(dow, hour)] = FutureQueryMix(
                template_ids=tids,
                probabilities=np.array(probs, dtype=np.float64),
                table_vectors=np.array(tvecs, dtype=np.float64),
                slo_tightness=np.array(tights, dtype=np.float64),
            )

    def _build_default_mix(self) -> FutureQueryMix:
        """Uniform mix over all known templates (used when the current
        time-of-week bin has no entry)."""
        all_tids = sorted(self._table_vectors.keys())
        if not all_tids:
            # Degenerate: no templates at all → single zero-vector entry.
            return FutureQueryMix(
                template_ids=["_empty"],
                probabilities=np.array([1.0]),
                table_vectors=np.zeros((1, self._n), dtype=np.float64),
                slo_tightness=np.array([0.0]),
            )
        k = len(all_tids)
        return FutureQueryMix(
            template_ids=all_tids,
            probabilities=np.full(k, 1.0 / k, dtype=np.float64),
            table_vectors=np.array(
                [self._table_vectors[t] for t in all_tids], dtype=np.float64
            ),
            slo_tightness=np.array(
                [self._tightness.get(t, self._fallback_tightness) for t in all_tids],
                dtype=np.float64,
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_future_query_mix(
        self,
        timestamp_s: float | None = None,
        base_datetime: datetime | None = None,
    ) -> FutureQueryMix:
        """Return the :class:`FutureQueryMix` for the time-of-week bin
        corresponding to the given timestamp.

        Parameters
        ----------
        timestamp_s :
            Unix epoch seconds.  Ignored if *base_datetime* is given.
        base_datetime :
            Explicit datetime (must be timezone-aware).  Takes priority
            over *timestamp_s*.

        Returns
        -------
        FutureQueryMix
            Pre-built mix for the matching ``(day_of_week, hour)`` bin,
            or a uniform default if no bin matches.
        """
        if base_datetime is not None:
            dt = base_datetime
        elif timestamp_s is not None:
            dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
        else:
            dt = datetime.now(tz=timezone.utc)

        key = _bin_key(dt.weekday(), dt.hour)
        return self._bins.get(key, self._default_mix)

    @property
    def schema_name(self) -> str:
        return self._schema_name

    @property
    def available_bins(self) -> list[tuple[int, int]]:
        """List of ``(day_of_week, hour)`` bins that have explicit entries."""
        return sorted(self._bins.keys())

    @property
    def all_template_ids(self) -> list[str]:
        """All unique template IDs across every bin."""
        return sorted(self._table_vectors.keys())
