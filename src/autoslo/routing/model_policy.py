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
    ClusterSnapshot,
    PlacementScore,
    RoutingCore,
    RoutingResult,
)
from autoslo.routing.routing_policy import RoutingPolicy
from autoslo.workload_definition.query import Query, QueryTextId, SloMetric
from autoslo.utils.structured_log import emit_structured, LOGGER_NAME

if TYPE_CHECKING:
    from autoslo.routing.managed_cluster_pool import ManagedClusterPool

logger = logging.getLogger(__name__)
_has_structured = lambda: bool(logging.getLogger(LOGGER_NAME).handlers)


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
        current_latencies: dict[str, float] | None = None,
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
            current_latencies=current_latencies or {},
            cluster_names=eligible,
        )

        # Create structured log records for the scores.
        if _has_structured():
            for cluster_name, score in scores.items():
                record = {
                    "timestamp": start_time_s,
                    "source": self.name,
                    "event_type": "routing_score",
                    "query_id": query_id,
                    "cluster_name": score.cluster_name,
                    "end_time_s": (
                        start_time_s
                        + (
                            score.latencies.get(query_id)
                            if score.latencies
                            else -1
                        )
                    ),
                    "marginal_slo_violation": score.marginal_slo_violation,
                    "marginal_cost": score.marginal_cost,
                }
                emit_structured(record)

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

        score_list = list(scores.values())
        best = RoutingCore.pick_best(score_list, tolerance=self.TOLERANCE_S)

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

        featurization = self._iconq_model.iconq_query_featurizer.featurize_from_query_text_id(
            qtid
        )
        stage_pred = self._iconq_model.stage_model.predict_from_query_text_id(
            {query_id: qtid}, cluster_rpu
        )[query_id].overall_mean_s()
        return Query(
            query_id=query_id,
            query_text_id=qtid,
            rel_start_time_s=start_time_s,
            featurization=featurization,
            stage_predictions_per_rpu={cluster_rpu: stage_pred},
        )

    # ------------------------------------------------------------------
    # Counterfactual scoring
    # ------------------------------------------------------------------

    def score_counterfactual(
        self,
        query: Query,
        arrival_time_s: float,
        snapshots: dict[str, ClusterSnapshot],
        cluster_rpus: dict[str, int],
        current_latencies: dict[str, float],
    ) -> tuple[str, dict[str, float]] | None:
        """Score *query* against virtual cluster snapshots using the full
        IconQ scoring pipeline.

        This mirrors :meth:`RoutingCore.score_query_on_clusters` but works
        against caller-provided virtual state rather than a live
        :class:`ManagedClusterPool`.  Used by
        :class:`~autoslo.capacity.headroom_policy.HeadroomPolicy` for
        counterfactual RPU selection.

        The RPU lookup on the interaction featuriser is temporarily
        overridden to cover hypothetical cluster names that don't exist
        in the real pool.
        """
        from autoslo.nn.concurrent_query_dataset import (  # noqa: PLC0415
            ConcurrentQueryDataset,
        )

        if not snapshots:
            return None

        featurizer = self._iconq_model.iconq_interaction_featurizer
        orig_lookup = featurizer._rpu_lookup

        # Override RPU lookup so virtual / hypothetical clusters resolve.
        featurizer.set_rpu_lookup(lambda cn: cluster_rpus[cn])
        try:
            return self._score_counterfactual_impl(
                query=query,
                arrival_time_s=arrival_time_s,
                snapshots=snapshots,
                cluster_rpus=cluster_rpus,
                current_latencies=current_latencies,
            )
        finally:
            featurizer._rpu_lookup = orig_lookup

    def _score_counterfactual_impl(
        self,
        query: Query,
        arrival_time_s: float,
        snapshots: dict[str, ClusterSnapshot],
        cluster_rpus: dict[str, int],
        current_latencies: dict[str, float],
    ) -> tuple[str, dict[str, float]] | None:
        """Inner implementation of counterfactual scoring (RPU lookup
        already set)."""
        from autoslo.nn.concurrent_query_dataset import (  # noqa: PLC0415
            ConcurrentQueryDataset,
        )

        # -- Build neighbour maps (all-vs-all per cluster) ----------------
        cluster_to_base_to_neighbors: dict[str, dict[Query, list[Query]]] = {}
        for cn, snap in snapshots.items():
            all_queries = list(snap.active_queries) + [query]
            # Use the *same* list object so ConcurrentQueryDataset can
            # hit the fast all-vs-all featurisation path.
            cluster_to_base_to_neighbors[cn] = {
                q: all_queries for q in all_queries
            }

        # -- Before-state per cluster -------------------------------------
        before: dict[str, tuple[float, float]] = {}
        for cn, snap in snapshots.items():
            before[cn] = RoutingCore.compute_before_state(
                snapshot=snap,
                current_time_s=arrival_time_s,
                slo_resolver=self._slo_resolver,
                slo_metric=self._slo_metric,
                latencies=current_latencies,
            )

        # -- Batched model predictions ------------------------------------
        dataset = ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=(
                self._iconq_model.iconq_interaction_featurizer
            ),
            cluster_to_base_to_neighbors=cluster_to_base_to_neighbors,
        )
        all_predictions = self._iconq_model.predict_from_dataset(dataset)

        # -- Score each cluster -------------------------------------------
        scores: list[PlacementScore] = []
        for cn, predictions in all_predictions.items():
            if cn not in before or cn not in snapshots:
                continue
            bc, bv = before[cn]
            score = RoutingCore.score_placement(
                query=query,
                snapshot=snapshots[cn],
                predictions=predictions,
                current_time_s=arrival_time_s,
                slo_resolver=self._slo_resolver,
                slo_metric=self._slo_metric,
                before_cost=bc,
                before_slo_violation=bv,
                current_latencies=current_latencies,
            )
            scores.append(score)

        if not scores:
            return None

        best = RoutingCore.pick_best(scores, tolerance=self.TOLERANCE_S)
        return (best.cluster_name, best.latencies)
