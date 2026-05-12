import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from autoslo.clusters.actions import ScalingAction, SpinUpAction, TearDownAction
from autoslo.clusters.autoscaling_policy import AutoscalingPolicy
from autoslo.clusters.cluster import (
    Cluster,
    ClusterState,
    ClusterView,
    cluster_cost_until_drained,
)
from autoslo.config.component_configs import AutoscalerConfig
from autoslo.filesystem.logging import emit_structured
from autoslo.filesystem.structured_events import BaseStructuredEvent, EventType
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter, QueryRouterConfig
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)


class Autoscaler:
    """
    Coordinator that dispatches events to an autoscaling policy and
    executes the returned actions via callbacks.
    """

    def __init__(
        self,
        slo_resolver: SloResolver,
        slo_objective: SloObjective,
        iconq_model: IconqModel,
        cluster_cache_state_dim: int,
        query_router_config: QueryRouterConfig,
        autoscaler_config: AutoscalerConfig,
        out_dir: str | Path,
    ) -> None:
        self._slo_resolver = slo_resolver
        self._slo_objective = slo_objective
        self._allowed_rpu_sizes = sorted(autoscaler_config.allowed_rpu_sizes)
        self._iconq_model = iconq_model
        self._query_router_config = query_router_config
        self._out_dir = out_dir
        self._min_cluster_lifetime_s = autoscaler_config.min_cluster_lifetime_s
        self._idle_time_before_tear_down_s = (
            autoscaler_config.idle_time_before_tear_down_s
        )
        self._observation_window_s = autoscaler_config.observation_window_s
        self._min_observations_to_act = (
            autoscaler_config.min_observations_to_act
        )
        self._slo_tightening_factor = autoscaler_config.slo_tightening_factor
        self._cluster_cache_state_dim = cluster_cache_state_dim
        self._autoscaling_policy = AutoscalingPolicy(
            autoscaler_config.autoscaling_policy
        )
        self._trigger_slo_resolver = (
            slo_resolver.tightened(autoscaler_config.slo_tightening_factor)
            if autoscaler_config.slo_tightening_factor != 1.0
            else slo_resolver
        )

        # Internal mutable state (guarded by _lock)
        self._lock = threading.Lock()
        self._window_start_time_s: Optional[float] = None
        self._snapshot_at_window_start: Optional[dict[str, ClusterView]] = None
        self._window_queries: list[Query] = []
        self._spin_up_disabled: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def slo_resolver(self) -> SloResolver:
        return self._slo_resolver

    @property
    def slo_objective(self) -> SloObjective:
        return self._slo_objective

    @property
    def allowed_rpu_sizes(self) -> list[int]:
        return self._allowed_rpu_sizes

    @property
    def iconq_model(self) -> IconqModel:
        return self._iconq_model

    @property
    def min_cluster_lifetime_s(self) -> float:
        return self._min_cluster_lifetime_s

    @property
    def idle_time_before_tear_down_s(self) -> float:
        return self._idle_time_before_tear_down_s

    @property
    def observation_window_s(self) -> float:
        return self._observation_window_s

    @property
    def min_observations_to_act(self) -> int:
        return self._min_observations_to_act

    @property
    def slo_tightening_factor(self) -> float:
        return self._slo_tightening_factor

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _reset_window(
        self,
        window_start_time: float,
        snapshot: dict[str, ClusterView],
    ) -> None:
        """Clear the routing window and all associated state."""
        self._window_start_time_s = window_start_time
        self._snapshot_at_window_start = snapshot
        self._window_queries = []

    def disable_spin_up(self) -> None:
        """Permanently disable spin-up recommendations.

        Called when a spin-up is rejected because the max cluster budget is
        exhausted, so future windows no longer waste cycles considering it.
        """
        with self._lock:
            self._spin_up_disabled = True

    def inform(
        self,
        rel_time_s: float,
        current_query: Query,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> list[ScalingAction]:
        with self._lock:
            return self._inform_locked(
                rel_time_s,
                current_query,
                pool_snapshot_with_current_query,
            )

    def _inform_locked(
        self,
        rel_time_s: float,
        current_query: Query,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> list[ScalingAction]:

        actions: list[ScalingAction] = []

        # Start new window if needed.
        if (self._window_start_time_s is None) or (
            (rel_time_s - self._window_start_time_s)
            > self._observation_window_s
        ):
            self._reset_window(
                rel_time_s,
                pool_snapshot_with_current_query,
            )
            return actions

        # Add the current query to the existing window.
        self._window_queries.append(current_query)
        if len(self._window_queries) < self._min_observations_to_act:
            return actions

        # Determine whether to take any spinup actions.
        spin_up_actions = self.consider_spin_up(
            rel_time_s,
            pool_snapshot_with_current_query,
        )
        actions.extend(spin_up_actions)

        # Determine whether to take any teardown actions.
        tear_down_actions = self.consider_teardown(
            rel_time_s, pool_snapshot_with_current_query
        )
        actions.extend(tear_down_actions)

        # If acting, reset the window so that future decisions are based on
        # post-action evidence.
        if len(actions) > 0:
            self._reset_window(
                rel_time_s,
                pool_snapshot_with_current_query,
            )

        return actions

    def consider_spin_up(
        self,
        rel_time_s: float,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> list[ScalingAction]:
        """
        Recommend spinning up a cluster if both are true:
        1. The window has at least ``min_observations_to_act`` queries.
        2. The SloObjective is being violated on the current snapshot.
        3. There are no other ongoing spinups.

        Use the window to determine the size of the window to spin up.
        """

        # Determine if we are disallowed from spinning up.
        if self._autoscaling_policy == AutoscalingPolicy.NOOP:
            return []
        if len(self.allowed_rpu_sizes) == 0 or self._spin_up_disabled:
            return []

        # Determine if we have enough observations to act.
        if len(self._window_queries) < self._min_observations_to_act:
            return []

        # Determine if there are ongoing spinups.
        for cluster in pool_snapshot_with_current_query.values():
            if cluster.state == ClusterState.PENDING:
                return []

        # Determine if the (possibly tightened) SLO objective is met.
        lat_and_slos = []
        for cluster_name, cluster in pool_snapshot_with_current_query.items():
            for q in cluster.active_queries:
                pred_lat = cluster.predicted_latencies[q.query_id]
                slo = self._trigger_slo_resolver.resolve(q.query_text_id)
                lat_and_slos.append(LatencySlo(pred_lat, slo))
        slo_metric_value = self._slo_objective.slo_metric.aggregate_batch(
            lat_and_slos
        )
        slo_is_met = self._slo_objective.is_met_from_aggregated(
            slo_metric_value
        )
        if slo_is_met:
            return []

        # Find the best size to spin up.
        best_rpu = self._select_rpu(rel_time_s)
        action = SpinUpAction(
            reason=(
                f"num_queries_in_window={len(self._window_queries)}, "
                f"slo_metric={self._slo_objective.slo_metric}, "
                f"slo_metric_value={slo_metric_value:.4f}, "
                f"slo_threshold={self._slo_objective.slo_threshold:.4f}, "
                f"slo_tightening_factor={self._slo_tightening_factor}"
            ),
            rpu=best_rpu,
        )
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.SPIN_UP_DECISION,
                source="Autoscaler",
                details={
                    "rpu": best_rpu,
                    "reason": action.reason,
                    "autoscaling_policy": self._autoscaling_policy.value,
                },
            )
        )
        returned_actions: list[ScalingAction] = [action]

        if (
            self._autoscaling_policy
            == AutoscalingPolicy.REPLACE_WITH_SINGLE_BEST
        ):
            replace_teardowns = [
                TearDownAction(
                    reason="replace policy",
                    cluster_name=cluster_name,
                )
                for cluster_name, cluster_view in pool_snapshot_with_current_query.items()
                if cluster_view.state == ClusterState.READY
            ]
            returned_actions.extend(replace_teardowns)

        return returned_actions

    def consider_teardown(
        self,
        rel_time_s: float,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> list[TearDownAction]:
        """
        Recommend tearing down any cluster(s) that have:
        1. Been idle for at least ``idle_time_before_tear_down_s`` seconds, and
        2. Exceeded the minimum cluster lifetime of ``min_cluster_lifetime_s``.
        """

        tear_down_actions: list[TearDownAction] = []

        for cluster_name, cluster in pool_snapshot_with_current_query.items():
            if (cluster.state != ClusterState.READY) or (
                len(cluster.active_query_ids) > 0
            ):
                continue

            idle_time_s = (
                rel_time_s - cluster.most_recent_query_completion_rel_time_s
            )
            lifetime_s = rel_time_s - cluster.creation_time_s

            if (idle_time_s >= self._idle_time_before_tear_down_s) and (
                lifetime_s >= self._min_cluster_lifetime_s
            ):
                action = TearDownAction(
                    reason=(
                        f"creation_time: {cluster.creation_time_s:.0f}s, "
                        f"most_recent_query_completion_time: "
                        f"{cluster.most_recent_query_completion_rel_time_s:.0f}s, "
                        f"current_time: {rel_time_s:.0f}s, "
                    ),
                    cluster_name=cluster_name,
                )
                tear_down_actions.append(action)

                emit_structured(
                    BaseStructuredEvent(
                        rel_time_s=rel_time_s,
                        event_type=EventType.TEAR_DOWN_DECISION,
                        source="Autoscaler",
                        cluster_name=cluster_name,
                        details={"reason": action.reason},
                    )
                )

        return tear_down_actions

    def _select_rpu(self, rel_time_s: float) -> int:
        """
        Select the RPU size for a new cluster based on the current window.
        """
        if self._autoscaling_policy == AutoscalingPolicy.DUPLICATE_LARGEST:
            assert self._snapshot_at_window_start is not None
            ready_rpus = [
                cluster_view.rpu
                for cluster_view in self._snapshot_at_window_start.values()
                if cluster_view.state == ClusterState.READY
            ]
            return (
                max(ready_rpus) if ready_rpus else max(self._allowed_rpu_sizes)
            )

        viol_and_costs: list[ViolationCost] = []

        for rpu in self._allowed_rpu_sizes:
            slo_viol_and_cost = self._counterfactual_replay(rpu, rel_time_s)
            viol_and_costs.append(slo_viol_and_cost)

            hyp_cluster_name = f"autoslo-{rpu}-hypothetical"
            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=rel_time_s,
                    event_type=EventType.RPU_COUNTERFACTUAL,
                    source="Autoscaler",
                    cluster_name=hyp_cluster_name,
                    details={
                        "rpu": rpu,
                        "slo_violation": slo_viol_and_cost.violation,
                        "cost": slo_viol_and_cost.cost,
                        "slo_threshold": self._slo_objective.slo_threshold,
                    },
                )
            )

        best_local_idx = self._slo_objective.idx_of_best(viol_and_costs)
        best_rpu = self._allowed_rpu_sizes[best_local_idx]
        best_viol_and_cost = viol_and_costs[best_local_idx]

        best_hyp_cluster_name = f"autoslo-{best_rpu}-hypothetical"
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.RPU_SELECTION,
                source="Autoscaler",
                cluster_name=best_hyp_cluster_name,
                details={
                    "rpu": best_rpu,
                    "slo_violation": best_viol_and_cost.violation,
                    "cost": best_viol_and_cost.cost,
                    "slo_threshold": self._slo_objective.slo_threshold,
                },
            )
        )

        return best_rpu

    def _counterfactual_replay(
        self,
        candidate_rpu: int,
        rel_time_s: float,
    ) -> ViolationCost:
        """Replay the routing window with a hypothetical new cluster of
        *candidate_rpu* and return the aggregate SLO-violation metric and cost.
        """
        assert self._snapshot_at_window_start is not None
        assert self._window_start_time_s is not None

        # Build a fully mutable local snapshot for replay from the frozen views.
        # REPLACE policy evaluates each candidate RPU in isolation (no existing
        # clusters), since it models replacing the entire pool with one cluster.
        if (
            self._autoscaling_policy
            == AutoscalingPolicy.REPLACE_WITH_SINGLE_BEST
        ):
            local_cluster_pool: dict[str, Cluster] = {}
        else:
            local_cluster_pool = {
                cluster_name: cluster_view.to_cluster()
                for cluster_name, cluster_view in self._snapshot_at_window_start.items()
            }
        hyp_cluster_name = f"autoslo-{candidate_rpu}-hypothetical"
        local_cluster_pool[hyp_cluster_name] = Cluster(
            creation_time_s=self._window_start_time_s,
            rpu=candidate_rpu,
            name=hyp_cluster_name,
            cache_state=np.zeros(self._cluster_cache_state_dim, dtype=float),
        )
        router = QueryRouter(
            slo_resolver=self._slo_resolver,
            slo_objective=self._slo_objective,
            query_router_config=self._query_router_config,
            iconq_model=self._iconq_model,
            out_dir=self._out_dir,
        )

        # Sequential replay.
        lat_and_slos = []
        for query in self._window_queries:
            time_s = query.rel_start_time_s

            # Expire any finished queries.
            for cluster_name, cluster in local_cluster_pool.items():
                qs_and_latencies = cluster.finish_queries_until(
                    rel_time_s=time_s,
                )
                if len(qs_and_latencies) > 0:
                    for q, latency_s in qs_and_latencies:
                        slo = self._slo_resolver.resolve(q.query_text_id)
                        lat_and_slos.append(LatencySlo(latency_s, slo))

            # Route and update state for the incoming query.
            # Wrap as ClusterViews for the router
            snapshot_for_routing = {
                cluster_name: ClusterView(cluster)
                for cluster_name, cluster in local_cluster_pool.items()
            }
            (
                selected_cluster_name,
                new_predicted_latencies_for_cluster,
                new_cache_state,
            ) = router.route_query(
                query=query,
                snapshot=snapshot_for_routing,
                rel_time_s=time_s,
            )
            local_cluster_pool[selected_cluster_name].add_query(
                query,
                new_predicted_latencies_for_cluster,
                new_cache_state,
            )

        # Also add the lat and slo from any queries still active at the end of
        # the window, and compute cost directly from the replay cluster state
        # using the same model as QueryRouter.
        total_cost = 0.0
        for cluster_name, cluster in local_cluster_pool.items():
            active_pairs = router._collect_cluster_pairs(
                queries=cluster.active_queries,
                predicted_latencies=cluster.predicted_latencies,
            )
            cluster_cost = cluster_cost_until_drained(
                queries=cluster.active_queries,
                predicted_latencies=cluster.predicted_latencies,
                past_billing_intervals=cluster.past_billing_intervals,
                billing_window_start_s=cluster.billing_window_start_s,
                cost_per_second=cluster.cost_per_second,
                current_rel_time_s=rel_time_s,
            )

            lat_and_slos.extend(active_pairs)
            total_cost += cluster_cost

        aggregate = self._slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        return ViolationCost(aggregate, total_cost)
