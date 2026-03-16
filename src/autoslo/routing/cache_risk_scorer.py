"""
cache_risk_scorer.py
--------------------
Stateless scorer for the "cache favourableness delta" of a routing decision.

Given a cluster's current cache state and the hypothetical state after
placing a query, this module computes how much harder life becomes for
future queries that are likely to arrive soon — weighted by their
probability and SLO tightness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# FutureQueryMix — describes the expected upcoming workload
# ---------------------------------------------------------------------------


@dataclass
class FutureQueryMix:
    """Probability distribution over upcoming query templates.

    Built offline (from per-hour/day histograms) and looked up at routing
    time by :class:`~autoslo.routing.forecast_loader.ForecastDistributionLoader`.

    All arrays share a common first axis of size *K* (number of templates).

    Attributes
    ----------
    template_ids :
        Human-readable template identifiers (length K).
    probabilities :
        Probability of each template arriving in the next window (K,).
        Should sum to 1 (or close to it).
    table_vectors :
        Per-template table-access vectors (K, N).  Same units as
        ``IconqQueryFeaturization[2*m:]``.
    slo_tightness :
        Per-template SLO tightness in [0, 1] (K,).
        ``isolated_prediction / slo``.
        Values >= 1 mean the query already meets or exceeds its SLO
        in isolation; higher values are penalised more.
    """

    template_ids: list[str]
    probabilities: np.ndarray  # (K,)
    table_vectors: np.ndarray  # (K, N)
    slo_tightness: np.ndarray  # (K,)

    def __post_init__(self) -> None:
        k = len(self.template_ids)
        if self.probabilities.shape != (k,):
            raise ValueError(
                f"probabilities shape {self.probabilities.shape} != ({k},)"
            )
        if self.table_vectors.shape[0] != k:
            raise ValueError(
                f"table_vectors rows {self.table_vectors.shape[0]} != {k}"
            )
        if self.slo_tightness.shape != (k,):
            raise ValueError(
                f"slo_tightness shape {self.slo_tightness.shape} != ({k},)"
            )


# ---------------------------------------------------------------------------
# Cosine similarity (vectorised over the template axis)
# ---------------------------------------------------------------------------


def _cosine_similarities(
    cache_state: np.ndarray, table_vectors: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between *cache_state* (N,) and each row
    of *table_vectors* (K, N).  Returns (K,) in [0, 1]."""
    cache_norm = np.linalg.norm(cache_state)
    if cache_norm == 0:
        return np.zeros(table_vectors.shape[0], dtype=np.float64)

    row_norms = np.linalg.norm(table_vectors, axis=1)
    # Avoid division by zero for templates with no table access.
    safe_norms = np.where(row_norms > 0, row_norms, 1.0)

    dots = table_vectors @ cache_state
    sims = dots / (cache_norm * safe_norms)
    # Templates with zero norms get similarity 0.
    sims = np.where(row_norms > 0, sims, 0.0)
    return np.clip(sims, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CacheRiskScorer
# ---------------------------------------------------------------------------


class CacheRiskScorer:
    """Compute the cache-risk penalty for a routing decision.

    The risk captures how much the routing degrades cache favourableness
    for future queries, weighted by arrival probability and SLO tightness.

    This class is **stateless** — all mutable cache state lives in
    :class:`~autoslo.routing.cluster_cache_state.ClusterCacheState`.
    """

    @staticmethod
    def score_cache_risk(
        current_cache: np.ndarray,
        hypothetical_cache: np.ndarray,
        future: FutureQueryMix,
    ) -> float:
        """Compute the scalar cache-risk penalty.

        For each future template *t*:

        1. ``fav_before_t = cosine_similarity(current_cache, table_vector_t)``
        2. ``fav_after_t  = cosine_similarity(hypothetical_cache, table_vector_t)``
        3. ``delta_t = max(0, fav_before_t − fav_after_t)``
           (only penalise degradation, not improvement)
        4. ``risk_t = delta_t × probability_t × slo_tightness_t``

        Returns ``sum(risk_t)`` over all templates.

        Parameters
        ----------
        current_cache :
            The cluster's current cache-state vector (N,).
        hypothetical_cache :
            The cache state if the candidate query is routed here (N,).
        future :
            Expected future workload mix (templates, probabilities,
            table vectors, SLO tightness).

        Returns
        -------
        float
            Non-negative scalar risk penalty.
        """
        fav_before = _cosine_similarities(current_cache, future.table_vectors)
        fav_after = _cosine_similarities(hypothetical_cache, future.table_vectors)

        degradation = np.maximum(0.0, fav_before - fav_after)  # (K,)
        weighted = degradation * future.probabilities * future.slo_tightness
        return float(weighted.sum())
