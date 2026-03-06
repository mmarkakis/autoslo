"""
model_policy.py
---------------
Model-based routing policy using :mod:`autoslo.routing.routing_core`.

This policy replaces the scoring logic that lived inside ``RAutoSLO``.
All mutable bookkeeping (active queries, neighbours, billing windows) is
delegated to a :class:`ClusterStateTracker`; this class owns only the
**IconQ model** and **SLO resolver**.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.routing_core import (
    PlacementScore,
    RoutingCore,
    RoutingResult,
)
from autoslo.routing.routing_policy import RoutingPolicy
from autoslo.workload_definition.query import Query, QueryTextId, SloMetric

if TYPE_CHECKING:
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool

logger = logging.getLogger(__name__)


class ModelPolicy(RoutingPolicy):
    """Routing policy backed by an IconQ model.

    Uses marginal *(SLO-violation, cost)* to pick the best cluster,
    matching the scoring logic of the offline simulator.

    Parameters
    ----------
    iconq_model_id :
        Identifier passed to ``IconqModel.load()``.
    default_slo_s :
        Default latency SLO (seconds) for templates without an override.
    slo_overrides :
        ``{template_id: slo_s}`` dict for per-template SLOs.
    slo_metric :
        Which SLO-violation metric drives the routing optimiser.
        See :class:`~autoslo.workload_definition.query.SloMetric`.
    """

    # SLO-violation tolerance for the lexicographic comparison (seconds).
    TOLERANCE_S = 1e-4

    def __init__(
        self,
        iconq_model_id: str,
        default_slo_s: float = 10.0,
        slo_overrides: Optional[dict[int, float]] = None,
        slo_metric: SloMetric = SloMetric.RELATIVE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Model ---------------------------------------------------------------
        self._iconq_model_id = iconq_model_id
        self._iconq_model = IconqModel.load(model_id=iconq_model_id)

        # SLO -----------------------------------------------------------------
        self._slo_resolver = SloResolver.from_dict(
            default_slo_s=default_slo_s,
            slo_dict=slo_overrides or {},
        )
        self._slo_metric = slo_metric

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def slo_resolver(self) -> SloResolver:
        """Expose the resolver for autoscaling policies."""
        return self._slo_resolver

    @property
    def iconq_model(self) -> IconqModel:
        return self._iconq_model

    @property
    def name(self) -> str:
        return f"ModelPolicy(iconq_model_id={self._iconq_model_id})"

    # ------------------------------------------------------------------
    # RoutingPolicy hooks
    # ------------------------------------------------------------------

    def on_attach(self, pool: ManagedClusterPool) -> None:
        """Inject an RPU lookup into the interaction featuriser."""
        self._iconq_model.iconq_interaction_featurizer.set_rpu_lookup(
            pool.get_rpu
        )

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def select_cluster(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        pool: ManagedClusterPool,
        exclude_clusters: set[str] | None = None,
    ) -> str:
        return self.route_with_details(
            query_id=query_id,
            query_text_id=query_text_id,
            start_time_s=start_time_s,
            pool=pool,
            exclude_clusters=exclude_clusters,
        ).cluster_name

    def route_with_details(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        pool: ManagedClusterPool,
        exclude_clusters: set[str] | None = None,
    ) -> RoutingResult:
        query_id = str(query_id)
        qtid = QueryTextId(value=str(query_text_id))

        eligible = pool.cluster_names
        if exclude_clusters:
            eligible = [cn for cn in eligible if cn not in exclude_clusters]

        scores, incoming, stage_preds = RoutingCore.score_query_on_clusters(
            iconq_model=self._iconq_model,
            pool=pool,
            query_id=query_id,
            query_text_id=qtid,
            start_time_s=start_time_s,
            slo_resolver=self._slo_resolver,
            slo_metric=self._slo_metric,
            cluster_names=eligible,
        )

        if not scores:
            logger.warning(
                "No scores produced for query %s; falling back to first "
                "eligible cluster.",
                query_id,
            )
            incoming.cluster_name = eligible[0]
            return RoutingResult(
                cluster_name=eligible[0],
                score=None,
                tracking_query=incoming,
            )

        score_list = list(scores.values())
        best = RoutingCore.pick_best(score_list, tolerance=self.TOLERANCE_S)

        # Finalise the incoming query for the chosen cluster.
        incoming.cluster_name = best.cluster_name
        incoming.stage_latency_prediction_s = stage_preds.get(best.cluster_name, -1)
        if best.latencies:
            incoming.latency_s = best.latencies.get(
                incoming.query_id, incoming.latency_s
            )

        return RoutingResult(
            cluster_name=best.cluster_name,
            score=best,
            tracking_query=incoming,
        )

    # ------------------------------------------------------------------
    # Tracking-query builder
    # ------------------------------------------------------------------

    def build_tracking_query(
        self,
        query_id: str,
        cluster_name: str,
        query_text_id: str,
        start_time_s: float,
        cluster_rpu: int = 0,
    ) -> Query:
        """Build a fully-featurised :class:`Query` for the state tracker."""
        query_id = str(query_id)
        qtid = QueryTextId(value=str(query_text_id))

        featurization = (
            self._iconq_model.iconq_query_featurizer
            .featurize_from_query_text_id(qtid)
        )
        stage_pred = (
            self._iconq_model.stage_model
            .predict_from_query_text_id(
                {query_id: qtid}, cluster_rpu
            )[query_id]
            .overall_mean_s()
        )
        return Query(
            query_id=query_id,
            query_text_id=qtid,
            rel_start_time_s=start_time_s,
            cluster_name=cluster_name,
            featurization=featurization,
            stage_latency_prediction_s=stage_pred,
            latency_s=stage_pred,
        )
