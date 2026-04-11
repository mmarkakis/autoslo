"""
cache_aware_policy.py
---------------------
Routing policy that extends :class:`ModelPolicy` with a forward-looking
cache-risk penalty.

For each candidate cluster the policy estimates how routing the current
query would shift the cluster's cache state away from what high-priority
future queries need.  The risk is combined with the standard marginal
SLO-violation score as::

    adjusted_slo = marginal_slo + λ · cache_risk

With ``λ = 0`` the policy is equivalent to :class:`ModelPolicy`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from autoslo.routing.cache_risk_scorer import CacheRiskScorer, FutureQueryMix
from autoslo.routing.cluster_cache_state import (
    ClusterCacheState,
    build_decay_strategy,
)
from autoslo.routing.forecast_loader import ForecastDistributionLoader
from autoslo.routing.model_policy import ModelPolicy
from autoslo.routing.routing_core import RoutingCore, RoutingResult
from autoslo.slo.slo_objective import SloMetric
from autoslo.workload_definition.query import QueryTextId

if TYPE_CHECKING:
    from autoslo.models.iconq_model import IconqModel
    from autoslo.clusters.managed_cluster_pool import ManagedClusterPool

logger = logging.getLogger(__name__)


class CacheAwarePolicy(ModelPolicy):
    """Model-based routing with cache-aware risk adjustment.

    Parameters
    ----------
    iconq_model_id :
        Passed through to :class:`ModelPolicy`.
    default_slo_s :
        Passed through to :class:`ModelPolicy`.
    slo_overrides :
        Passed through to :class:`ModelPolicy`.
    slo_metric :
        Passed through to :class:`ModelPolicy`.
    forecast_distribution_path :
        Path to the forecast-distribution YAML
        (see :mod:`~autoslo.routing.forecast_loader`).
    slo_tightness_path :
        Path to the SLO-tightness YAML.
    cache_risk_lambda :
        Scalar weight for the cache-risk penalty (default 0.0 =
        equivalent to :class:`ModelPolicy`).
    cache_decay_strategy :
        One of ``"exponential"``, ``"sliding_window"``, ``"lru"``.
    cache_decay_params :
        Strategy-specific keyword arguments forwarded to
        :func:`~autoslo.routing.cluster_cache_state.build_decay_strategy`.
    fallback_tightness :
        Default tightness for templates missing from the tightness table.
    """

    def __init__(
        self,
        iconq_model_id: str,
        default_slo_s: float = 10.0,
        slo_overrides: Optional[dict[str, float]] = None,
        slo_metric: SloMetric = SloMetric.RELATIVE,
        *,
        forecast_distribution_path: str,
        slo_tightness_path: str,
        cache_risk_lambda: float = 0.0,
        cache_decay_strategy: str = "exponential",
        cache_decay_params: Optional[dict[str, Any]] = None,
        fallback_tightness: float = 0.5,
        iconq_model: Optional["IconqModel"] = None,
    ) -> None:
        super().__init__(
            iconq_model_id=iconq_model_id,
            default_slo_s=default_slo_s,
            slo_overrides=slo_overrides,
            slo_metric=slo_metric,
            iconq_model=iconq_model,
        )

        self._lambda = cache_risk_lambda
        self._decay_strategy_kind = cache_decay_strategy
        self._decay_params = cache_decay_params or {}

        # Derive dims from the loaded model's featurizer.
        featurizer = self._iconq_model.iconq_query_featurizer
        self._m = featurizer._m
        self._n = featurizer._n

        # Forecast loader (pre-computes table vectors per template).
        self._forecast_loader = ForecastDistributionLoader(
            forecast_distribution_path=forecast_distribution_path,
            slo_tightness_path=slo_tightness_path,
            iconq_query_featurizer=featurizer,
            n_table_dims=self._n,
            m_operator_dims=self._m,
            fallback_tightness=fallback_tightness,
        )

        # Per-cluster cache states — populated lazily in on_attach / route.
        self._cluster_caches: dict[str, ClusterCacheState] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return (
            f"CacheAwarePolicy(iconq_model_id={self._iconq_model_id}, "
            f"lambda={self._lambda})"
        )

    @property
    def forecast_loader(self) -> ForecastDistributionLoader:
        return self._forecast_loader

    # ------------------------------------------------------------------
    # RoutingPolicy hooks
    # ------------------------------------------------------------------

    def on_attach(self, pool: ManagedClusterPool) -> None:
        super().on_attach(pool)
        for cn in pool.cluster_names:
            self._ensure_cache(cn)

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route_with_details(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        pool: ManagedClusterPool,
        exclude_clusters: set[str] | None = None,
        current_latencies: dict[str, float] | None = None,
    ) -> RoutingResult:
        query_id = str(query_id)
        qtid = QueryTextId(value=str(query_text_id))

        eligible = pool.cluster_names
        if exclude_clusters:
            eligible = [cn for cn in eligible if cn not in exclude_clusters]

        # 1) Base scores from RoutingCore (same as ModelPolicy).
        scores, incoming, stage_preds = RoutingCore.score_query_on_clusters(
            iconq_model=self._iconq_model,
            pool=pool,
            query_id=query_id,
            query_text_id=qtid,
            start_time_s=start_time_s,
            slo_resolver=self._slo_resolver,
            slo_metric=self._slo_metric,
            current_latencies=current_latencies or {},
            cluster_names=eligible,
        )

        if not scores:
            logger.warning(
                "No scores produced for query %s; falling back to first "
                "eligible cluster.",
                query_id,
            )
            return RoutingResult(
                cluster_name=eligible[0],
                score=None,
                query=incoming,
            )

        # 2) Extract the incoming query's table-access vector.
        table_vector = self._extract_table_vector(qtid)

        # 3) Get forecast mix for current time.
        future_mix = self._forecast_loader.get_future_query_mix(
            timestamp_s=start_time_s,
        )

        # 4) Augment each score with cache risk.
        for cn, score in scores.items():
            cache = self._ensure_cache(cn)
            current_cs = cache.current_state(start_time_s)
            hypo_cs = cache.hypothetical_state(table_vector, start_time_s)
            risk = CacheRiskScorer.score_cache_risk(
                current_cache=current_cs,
                hypothetical_cache=hypo_cs,
                future=future_mix,
            )
            score.cache_risk = risk
            score.adjusted_slo_violation = (
                score.marginal_slo_violation + self._lambda * risk
            )

        # 5) Pick best using adjusted scores.
        score_list = list(scores.values())
        best = RoutingCore.pick_best(score_list, tolerance=self.TOLERANCE_S)

        # 6) Update cache state for the chosen cluster.
        self._cluster_caches[best.cluster_name].update(
            table_vector, start_time_s
        )

        predicted_latency = (
            best.latencies.get(incoming.query_id, -1.0)
            if best.latencies
            else -1.0
        )

        return RoutingResult(
            cluster_name=best.cluster_name,
            score=best,
            query=incoming,
            stage_prediction_s=stage_preds.get(best.cluster_name, -1.0),
            predicted_latency_s=predicted_latency,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_cache(self, cluster_name: str) -> ClusterCacheState:
        if cluster_name not in self._cluster_caches:
            strategy = build_decay_strategy(
                kind=self._decay_strategy_kind,
                n_tables=self._n,
                params=self._decay_params,
            )
            self._cluster_caches[cluster_name] = ClusterCacheState(
                n_tables=self._n,
                strategy=strategy,
            )
        return self._cluster_caches[cluster_name]

    def _extract_table_vector(self, qtid: QueryTextId) -> np.ndarray:
        """Get the table-access vector (length N) for a query."""
        try:
            feat = self._iconq_model.iconq_query_featurizer.featurize_from_query_text_id(
                qtid
            )
            return np.array(feat[2 * self._m :], dtype=np.float64)
        except (ValueError, KeyError):
            return np.zeros(self._n, dtype=np.float64)
