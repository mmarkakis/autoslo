"""
model_policy.py
---------------
Model-based routing policy using :mod:`autoslo.routing.routing_core`.

This policy replaces the scoring logic that lived inside ``RAutoSLO``.
All mutable bookkeeping (active queries, neighbours, billing windows) is
delegated to a :class:`ClusterStateTracker`; this class owns only the
**IconQ model**, **SLO resolver**, and the rolling **routing window**
(which is policy-specific state, not per-cluster bookkeeping).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Optional

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.routing.routing_core import (
    ClusterSnapshot,
    PlacementScore,
    RoutingCore,
)
from autoslo.routing.routing_policy import RoutingPolicy
from autoslo.workload_definition.query import Query

if TYPE_CHECKING:
    from autoslo.routing.cluster_state_tracker import ClusterStateTracker

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
    optimize_by_amount :
        If *True*, violations are measured in seconds of overshoot;
        if *False*, violations are binary (0/1) per query.
    on_capacity_pressure :
        Optional no-arg callback invoked when *every* eligible cluster
        would incur an SLO violation for the incoming query.
    routing_window_s :
        Duration (seconds) of the rolling routing-decision window
        retained for counterfactual RPU selection.
    """

    # SLO-violation tolerance for the lexicographic comparison (seconds).
    TOLERANCE_S = 1e-4

    def __init__(
        self,
        iconq_model_id: str,
        default_slo_s: float = 10.0,
        slo_overrides: Optional[dict[int, float]] = None,
        optimize_by_amount: bool = True,
        on_capacity_pressure: Optional[Callable[[], None]] = None,
        routing_window_s: float = 120.0,
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
        self._optimize_by_amount = optimize_by_amount

        # Capacity pressure ---------------------------------------------------
        self._on_capacity_pressure = on_capacity_pressure

        # Rolling routing window ----------------------------------------------
        self._routing_window_s = routing_window_s
        self._routing_window: deque[tuple[Query, ClusterSnapshot | None]] = (
            deque()
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def slo_resolver(self) -> SloResolver:
        """Expose the resolver for the capacity controller."""
        return self._slo_resolver

    @property
    def iconq_model(self) -> IconqModel:
        return self._iconq_model

    # ------------------------------------------------------------------
    # RoutingPolicy hooks
    # ------------------------------------------------------------------

    def on_attach(self, state_tracker: ClusterStateTracker) -> None:
        """Inject an RPU lookup into the interaction featuriser."""
        self._iconq_model.iconq_interaction_featurizer.set_rpu_lookup(
            state_tracker.get_rpu
        )

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def select_cluster(
        self,
        query_id: str,
        query_text_id: str,
        start_time_s: float,
        state_tracker: ClusterStateTracker,
    ) -> str:
        query_id = str(query_id)
        query_text_id = str(query_text_id)

        # Build featurisation for the incoming query.
        featurization = (
            self._iconq_model.iconq_query_featurizer
            .featurize_from_tpcds_temp_and_q_idx(query_text_id)
        )

        incoming = Query(
            query_id=query_id,
            query_text_id=query_text_id,
            rel_start_time_s=start_time_s,
            featurization=featurization,
            latency_s=-1,
        )

        # Snapshot current state from the tracker.
        snapshots, run_to_base_to_neighbors = (
            state_tracker.build_routing_context(incoming)
        )
        eligible = state_tracker.cluster_names

        # -- Stage-model prediction (per cluster) ----------------------------
        stage_latency_predictions: dict[str, float] = {}
        for cn in eligible:
            incoming.cluster_name = cn
            stage_pred = (
                self._iconq_model.stage_model
                .predict_from_tpcds_temp_and_q_idx(
                    {query_id: query_text_id}, cn
                )[query_id]
                .overall_mean_s()
            )
            incoming.stage_latency_prediction_s = stage_pred
            stage_latency_predictions[cn] = stage_pred

        # -- Before-state per cluster ----------------------------------------
        before: dict[str, tuple[float, float]] = {}
        for cn, snap in snapshots.items():
            incoming.cluster_name = cn
            incoming.stage_latency_prediction_s = stage_latency_predictions[cn]
            before[cn] = RoutingCore.compute_before_state(
                snapshot=snap,
                current_time_s=start_time_s,
                slo_resolver=self._slo_resolver,
                optimize_by_amount=self._optimize_by_amount,
            )

        # -- Batched model prediction across all clusters --------------------
        dataset = ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=(
                self._iconq_model.iconq_interaction_featurizer
            ),
            run_to_base_to_neighbors=run_to_base_to_neighbors,
        )
        all_predictions = self._iconq_model.predict_from_dataset(dataset)

        # -- Score each cluster ----------------------------------------------
        scores: list[PlacementScore] = []
        for cn, predictions in all_predictions.items():
            bc, bv = before[cn]
            score = RoutingCore.score_placement(
                query=incoming,
                snapshot=snapshots[cn],
                predictions=predictions,
                current_time_s=start_time_s,
                slo_resolver=self._slo_resolver,
                optimize_by_amount=self._optimize_by_amount,
                before_cost=bc,
                before_slo_violation=bv,
            )
            scores.append(score)

        if not scores:
            logger.warning(
                "No scores produced for query %s; falling back to first "
                "eligible cluster.",
                query_id,
            )
            return eligible[0]

        best = RoutingCore.pick_best(scores, tolerance=self.TOLERANCE_S)

        # Emit capacity-pressure signal if every cluster would violate SLO.
        if all(s.marginal_slo_violation > 0 for s in scores):
            logger.info(
                "Capacity pressure: all %d clusters have positive marginal "
                "SLO violation for query %s.",
                len(scores),
                query_id,
            )
            if self._on_capacity_pressure is not None:
                try:
                    self._on_capacity_pressure()
                except Exception:
                    logger.exception("capacity_pressure callback failed")

        # Maintain rolling routing window.
        best_snapshot = snapshots.get(best.cluster_name)
        if best.latencies:
            incoming.latency_s = best.latencies.get(
                incoming.query_id, incoming.latency_s
            )
        self._routing_window.append((incoming, best_snapshot))
        cutoff = start_time_s - self._routing_window_s
        while (
            self._routing_window
            and self._routing_window[0][0].rel_start_time_s < cutoff
        ):
            self._routing_window.popleft()

        return best.cluster_name

    # ------------------------------------------------------------------
    # Tracking-query builder
    # ------------------------------------------------------------------

    def build_tracking_query(
        self,
        query_id: str,
        cluster_name: str,
        query_text_id: str,
        start_time_s: float,
    ) -> Query:
        """Build a fully-featurised :class:`Query` for the state tracker."""
        query_id = str(query_id)
        query_text_id = str(query_text_id)

        featurization = (
            self._iconq_model.iconq_query_featurizer
            .featurize_from_tpcds_temp_and_q_idx(query_text_id)
        )
        stage_pred = (
            self._iconq_model.stage_model
            .predict_from_tpcds_temp_and_q_idx(
                {query_id: query_text_id}, cluster_name
            )[query_id]
            .overall_mean_s()
        )
        return Query(
            query_id=query_id,
            query_text_id=query_text_id,
            rel_start_time_s=start_time_s,
            cluster_name=cluster_name,
            featurization=featurization,
            stage_latency_prediction_s=stage_pred,
            latency_s=stage_pred,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_routing_window(
        self,
    ) -> list[tuple[Query, ClusterSnapshot | None]]:
        """Return a copy of the recent routing-decision window."""
        return list(self._routing_window)

    def compute_slo_headroom(
        self, state_tracker: ClusterStateTracker
    ) -> float:
        """Compute the minimum SLO headroom across all active queries."""
        all_active: list[Query] = []
        for qs in state_tracker.get_all_active_queries().values():
            all_active.extend(qs)
        return RoutingCore.compute_slo_headroom(
            all_active, self._slo_resolver
        )
