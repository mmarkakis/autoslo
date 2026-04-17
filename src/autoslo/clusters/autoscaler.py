import logging
import threading
from typing import Optional

from intervaltree import Interval  # type: ignore[import]

from autoslo.clusters.actions import ScalingAction, SpinUpAction, TearDownAction
from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter, QueryRouterPolicy
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.billing import Billing
from autoslo.utils.logging import emit_structured
from autoslo.utils.structured_events import (
    RpuCounterfactualEvent,
    RpuSelectionEvent,
    TearDownDecisionEvent,
)
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)

MAX_CLUSTERS: int = 10


class Autoscaler:
    """
    Coordinator that dispatches events to an autoscaling policy and
    executes the returned actions via callbacks.
    """

    def __init__(
        self,
        slo_resolver: SloResolver,
        slo_objective: SloObjective,
        allowed_rpu_sizes: list[int],
        iconq_model: IconqModel,
        min_cluster_lifetime_s: float = 1200.0,
        idle_time_before_tear_down_s: float = 300.0,  # TODO: should clusters note start of idle period?
        observation_window_s: float = 120.0,
        min_observations_to_act: int = 5,
        routing_policy: QueryRouterPolicy = QueryRouterPolicy.USE_ICONQ_MODEL,
        slo_tightening_factor: float = 1.0,
        max_clusters: int = MAX_CLUSTERS,
    ) -> None:
        self._slo_resolver = slo_resolver
        self._slo_objective = slo_objective
        self._allowed_rpu_sizes = sorted(allowed_rpu_sizes)
        self._iconq_model = iconq_model
        self._routing_policy = routing_policy
        self._min_cluster_lifetime_s = min_cluster_lifetime_s
        self._idle_time_before_tear_down_s = idle_time_before_tear_down_s
        self._observation_window_s = observation_window_s
        self._min_observations_to_act = min_observations_to_act
        self._slo_tightening_factor = slo_tightening_factor
        self._max_clusters = max_clusters
        self._trigger_slo_resolver = (
            slo_resolver.tightened(slo_tightening_factor)
            if slo_tightening_factor != 1.0
            else slo_resolver
        )

        # Internal mutable state (guarded by _lock)
        self._lock = threading.Lock()
        self._window_start_time_s: Optional[float] = None
        self._snapshot_at_window_start: Optional[dict[str, Cluster]] = None
        self._window_queries: list[Query] = []

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
        snapshot: dict[str, Cluster],
    ) -> None:
        """Clear the routing window and all associated state."""
        self._window_start_time_s = window_start_time
        self._snapshot_at_window_start = snapshot
        self._window_queries = []

    def inform(
        self,
        rel_time_s: float,
        current_query: Query,
        pool_snapshot_with_current_query: dict[str, Cluster],
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
        pool_snapshot_with_current_query: dict[str, Cluster],
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

        # Store rel_time_s for use in structured event emissions.
        self._latest_rel_time_s = rel_time_s

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
        pool_snapshot_with_current_query: dict[str, Cluster],
    ) -> list[SpinUpAction]:
        """
        Recommend spinning up a cluster if both are true:
        1. The window has at least ``min_observations_to_act`` queries.
        2. The SloObjective is being violated on the current snapshot.
        3. There are no other ongoing spinups.

        Use the window to determine the size of the window to spin up.
        """

        # Determine if we are disallowed from spinning up.
        if len(self.allowed_rpu_sizes) == 0:
            return []

        # Determine if we have enough observations to act.
        if len(self._window_queries) < self._min_observations_to_act:
            return []

        # Determine if there are ongoing spinups.
        for cluster in pool_snapshot_with_current_query.values():
            if cluster.state == ClusterState.PENDING:
                return []

        # Guard: do not exceed the maximum cluster count.
        if len(pool_snapshot_with_current_query) >= self._max_clusters:
            logger.warning(
                "Skipping spin-up: pool already has %d clusters (max %d).",
                len(pool_snapshot_with_current_query),
                self._max_clusters,
            )
            return []

        # Determine if the (possibly tightened) SLO objective is met.
        lat_and_slos = []
        for cluster_name, cluster in pool_snapshot_with_current_query.items():
            for q in cluster.active_queries:
                pred_lat = cluster.predicted_latencies[q.query_id]
                slo = self._trigger_slo_resolver.resolve(q.query_text_id)
                lat_and_slos.append((pred_lat, slo))
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
        return [action]

    def consider_teardown(
        self,
        rel_time_s: float,
        pool_snapshot_with_current_query: dict[str, Cluster],
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
                    TearDownDecisionEvent(
                        rel_time_s=self._latest_rel_time_s,
                        source="Autoscaler",
                        cluster_name=cluster_name,
                        rpu=cluster.rpu,
                        detail=action.reason,
                    )
                )

        return tear_down_actions

    def _select_rpu(self, rel_time_s: float) -> int:
        """
        Select the RPU size for a new cluster based on the current window.
        """

        best_rpu: int = 4  # placeholder
        best_viol_and_cost: tuple[float, float] = (float("inf"), float("inf"))

        for rpu in self._allowed_rpu_sizes:
            slo_viol_and_cost = self._counterfactual_replay(rpu, rel_time_s)

            if self._slo_objective.is_b_better(
                best_viol_and_cost, slo_viol_and_cost
            ):
                best_rpu = rpu
                best_viol_and_cost = slo_viol_and_cost

            hyp_cluster_name = f"autoslo-{rpu}-hypothetical"
            emit_structured(
                RpuCounterfactualEvent(
                    rel_time_s=self._latest_rel_time_s,
                    source="Autoscaler",
                    cluster_name=hyp_cluster_name,
                    rpu=rpu,
                    slo_violation=slo_viol_and_cost[0],
                    cost=slo_viol_and_cost[1],
                    slo_threshold=self._slo_objective.slo_threshold,
                )
            )

        best_hyp_cluster_name = f"autoslo-{best_rpu}-hypothetical"
        emit_structured(
            RpuSelectionEvent(
                rel_time_s=self._latest_rel_time_s,
                source="Autoscaler",
                cluster_name=best_hyp_cluster_name,
                rpu=best_rpu,
                slo_violation=best_viol_and_cost[0],
                cost=best_viol_and_cost[1],
                slo_threshold=self._slo_objective.slo_threshold,
            )
        )

        return best_rpu

    def _counterfactual_replay(
        self,
        candidate_rpu: int,
        rel_time_s: float,
    ) -> tuple[float, float]:
        """Replay the routing window with a hypothetical new cluster of
        *candidate_rpu* and return the aggregate SLO-violation metric and cost.
        """
        assert self._snapshot_at_window_start is not None

        # Set up.
        local_snapshot = {
            cluster_name: cluster.clone()
            for cluster_name, cluster in self._snapshot_at_window_start.items()
        }
        hyp_cluster_name = f"autoslo-{candidate_rpu}-hypothetical"
        local_snapshot[hyp_cluster_name] = Cluster(
            creation_time_s=rel_time_s,
            rpu=candidate_rpu,
            name=hyp_cluster_name,
        )
        billed_intervals: dict[str, list[Interval]] = {}
        for cluster_name, cluster in local_snapshot.items():
            billed_intervals[cluster_name] = []
            if cluster.billing_window_start_s is not None:
                billed_intervals[cluster_name].append(
                    Interval(
                        begin=cluster.billing_window_start_s, end=rel_time_s
                    )
                )
        router = QueryRouter(
            slo_resolver=self._slo_resolver,
            slo_objective=self._slo_objective,
            routing_policy=self._routing_policy,
        )

        # Sequential replay.
        lat_and_slos = []
        for query in self._window_queries:
            time_s = query.rel_start_time_s

            # Expire any finished queries.
            for cluster_name, cluster in local_snapshot.items():
                qs_and_latencies = cluster.finish_queries_until(
                    rel_time_s=time_s,
                )
                if len(qs_and_latencies) > 0:
                    for q, latency_s in qs_and_latencies:
                        slo = self._slo_resolver.resolve(q.query_text_id)
                        lat_and_slos.append((latency_s, slo))

                        billed_intervals[cluster_name].append(
                            Interval(begin=q.rel_start_time_s, end=latency_s)
                        )

            # Route and update state for the incoming query.
            snapshot_for_routing = {
                cluster_name: cluster.clone()
                for cluster_name, cluster in local_snapshot.items()
            }
            selected_cluster_name, new_predicted_latencies_for_cluster = (
                router.route_query(
                    query=query,
                    clusters=snapshot_for_routing,
                    iconq_model=self._iconq_model,
                    rel_time_s=time_s,
                )
            )
            local_snapshot[selected_cluster_name].add_query(query)
            local_snapshot[selected_cluster_name].predicted_latencies = (
                new_predicted_latencies_for_cluster
            )

        # Also add the lat and slo from any queries still active at the end of
        # the window.
        for cluster_name, cluster in local_snapshot.items():
            for q in cluster.active_queries:
                pred_lat = cluster.predicted_latencies[q.query_id]
                slo = self._slo_resolver.resolve(q.query_text_id)
                lat_and_slos.append((pred_lat, slo))

        aggregate = self._slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        total_cost = sum(
            cluster.cost_per_second
            * Billing.billed_s(billed_intervals.get(cluster_name, []))
            for cluster_name, cluster in local_snapshot.items()
        )
        return aggregate, total_cost
