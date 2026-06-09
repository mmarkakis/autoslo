import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
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
from autoslo.config.component_configs import AutoscalerConfig, ProvisionerConfig
from autoslo.filesystem.structured_events import BaseStructuredEvent, EventType
from autoslo.filesystem.structured_log import emit_structured
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter, QueryRouterConfig
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CounterfactualReplayStats:
    candidate_rpu: int
    simulated_queries: int
    replay_copies: int
    finished_after_ready: int
    organic_done: bool


@dataclass(frozen=True)
class SelectRpuStats:
    total_simulated_queries: int
    per_candidate: dict[int, CounterfactualReplayStats]


class Autoscaler:
    """
    Coordinator that dispatches events to an autoscaling policy and
    executes the returned actions via callbacks.
    """

    def __init__(
        self,
        slo_resolver: SloResolver,
        slo_objective: SloObjective,
        provisioner_config: ProvisionerConfig,
        query_router_config: QueryRouterConfig,
        autoscaler_config: AutoscalerConfig,
        out_dir: str | Path,
        iconq_model: Optional[IconqModel] = None,
        force_one_decision_after_query_count: Optional[int] = None,
    ) -> None:
        self._slo_resolver = slo_resolver
        self._slo_objective = slo_objective
        self._allowed_rpu_sizes = sorted(autoscaler_config.allowed_rpu_sizes)
        self._iconq_model = (
            iconq_model
            if iconq_model is not None
            else IconqModel.load(query_router_config.iconq_model_id)
        )
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
        self._cluster_cache_state_dim = (
            provisioner_config.cluster_cache_state_dim
        )
        self._autoscaling_policy = AutoscalingPolicy(
            autoscaler_config.autoscaling_policy
        )
        self._trigger_slo_resolver = (
            slo_resolver.tightened(autoscaler_config.slo_tightening_factor)
            if autoscaler_config.slo_tightening_factor != 1.0
            else slo_resolver
        )
        self._spin_up_delay_s = provisioner_config.spin_up_delay_s
        self._min_finished_queries_in_counterfactual = (
            autoscaler_config.min_finished_queries_in_counterfactual
        )

        # Internal mutable state (guarded by _lock)
        self._lock = threading.Lock()
        self._trailing_queries: deque[Query] = deque()
        self._num_queries_since_last_cluster_ready: int = 0
        self._known_ready_cluster_names: frozenset[str] = frozenset()
        self._spin_up_disabled: bool = False
        # True from the moment a SpinUpAction is emitted until the new cluster
        # first appears as READY in the pool.  Prevents duplicate spin-up
        # recommendations during the provisioning window (which can be minutes
        # in live mode, during which no PENDING cluster is visible in the pool).
        self._spin_up_in_flight: bool = False

        # Forced mode (set when force_one_decision_after_query_count is
        # provided).
        self._force_one_decision_after_query_count: Optional[int] = (
            force_one_decision_after_query_count
        )
        self._inform_count: int = 0
        self._last_select_rpu_stats = SelectRpuStats(
            total_simulated_queries=0,
            per_candidate={},
        )

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

    @property
    def forced_decision_mode(self) -> bool:
        return self._force_one_decision_after_query_count is not None

    @property
    def last_select_rpu_stats(self) -> SelectRpuStats:
        return self._last_select_rpu_stats

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def disable_spin_up(self) -> None:
        """Permanently disable spin-up recommendations.

        Called when a spin-up is rejected because the max cluster budget is
        exhausted, so future windows no longer waste cycles considering it.
        """
        with self._lock:
            self._spin_up_disabled = True
            self._spin_up_in_flight = False

    def clear_spin_up_in_flight(self) -> None:
        """Clear the in-flight spin-up flag without disabling future spin-ups.

        Called by the runner when a provisioning attempt fails with an
        exception, so that the autoscaler can attempt another spin-up on the
        next eligible query rather than being blocked permanently.
        """
        with self._lock:
            self._spin_up_in_flight = False

    def inform(
        self,
        rel_time_s: float,
        current_query: Query,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> list[ScalingAction]:
        with self._lock:
            self._inform_count += 1
            if self._autoscaling_policy == AutoscalingPolicy.NOOP:
                return []

            # Detect new-READY clusters and reset the post-spinup observation
            # counter.  Also clear _spin_up_in_flight: the cluster we were
            # waiting for has arrived.
            current_ready = frozenset(
                name
                for name, cluster in pool_snapshot_with_current_query.items()
                if cluster.state == ClusterState.READY
            )
            if current_ready - self._known_ready_cluster_names:
                self._num_queries_since_last_cluster_ready = 0
                self._known_ready_cluster_names = current_ready
                self._spin_up_in_flight = False
            self._num_queries_since_last_cluster_ready += 1

            # Maintain the trailing window.
            self._trailing_queries.append(current_query)
            cutoff_s = rel_time_s - self._observation_window_s
            while (
                self._trailing_queries
                and self._trailing_queries[0].rel_start_time_s < cutoff_s
            ):
                self._trailing_queries.popleft()

            actions: list[ScalingAction] = []

            # Determine whether to take any spinup actions.
            spin_up_actions = self.consider_spin_up(
                rel_time_s, pool_snapshot_with_current_query
            )
            actions.extend(spin_up_actions)

            # Determine whether to take any teardown actions.
            tear_down_actions = self.consider_teardown(
                rel_time_s, pool_snapshot_with_current_query
            )
            actions.extend(tear_down_actions)
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
        3. No spin-up is currently in flight.

        Use the window to determine the size of the cluster to spin up.
        """

        # Determine if we are disallowed from spinning up.
        if self._autoscaling_policy == AutoscalingPolicy.NOOP:
            return []
        if len(self.allowed_rpu_sizes) == 0 or self._spin_up_disabled:
            return []

        # Block if a spin-up is already in flight (not yet reflected in pool).
        # This covers the live-runner case where the cluster goes directly from
        # not-in-pool to READY without ever appearing as PENDING, making the
        # pool-level PENDING check ineffective.
        if self._spin_up_in_flight:
            return []

        if self.forced_decision_mode:
            ### IN FORCED MODE ###
            if self._inform_count != self._force_one_decision_after_query_count:
                return []

            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=rel_time_s,
                    event_type=EventType.FORCED_DECISION_POINT,
                    source="Autoscaler",
                    details={
                        "force_one_decision_after_query_count": (
                            self._force_one_decision_after_query_count
                        )
                    },
                )
            )
            reason = (
                f"forced decision point, "
                f"force_one_decision_after_query_count="
                f"{self._force_one_decision_after_query_count}, "
                f"autoscaling_policy={self._autoscaling_policy.value}"
            )
        else:
            ### NOT IN FORCED MODE: CHECK CONDITIONS ###

            # Determine if we have enough observations to act.
            if (
                self._num_queries_since_last_cluster_ready
                < self._min_observations_to_act
            ):
                return []

            # Determine if the (possibly tightened) SLO objective is met.
            lat_and_slos = []
            for (
                cluster_name,
                cluster,
            ) in pool_snapshot_with_current_query.items():
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

            reason = (
                f"num_queries_since_last_cluster_ready="
                f"{self._num_queries_since_last_cluster_ready}, "
                f"slo_metric={self._slo_objective.slo_metric}, "
                f"slo_metric_value={slo_metric_value:.4f}, "
                f"slo_threshold={self._slo_objective.slo_threshold:.4f}, "
                f"slo_tightening_factor={self._slo_tightening_factor}"
            )

        # Find the best size to spin up.

        best_rpu = self._select_rpu(
            rel_time_s,
            pool_snapshot_with_current_query,
        )
        deferred_teardowns: tuple[str, ...] = ()
        if (
            self._autoscaling_policy
            == AutoscalingPolicy.REPLACE_WITH_SINGLE_BEST_FORWARD
        ):
            deferred_teardowns = tuple(
                cluster_name
                for cluster_name, cluster_view in pool_snapshot_with_current_query.items()
                if cluster_view.state == ClusterState.READY
            )

        action = SpinUpAction(
            reason=reason,
            rpu=best_rpu,
            deferred_teardowns=deferred_teardowns,
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
        self._spin_up_in_flight = True
        return [action]

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

        if self.forced_decision_mode:
            ### IN FORCED MODE: DO NOT TEAR DOWN ANYTHING (TO AVOID INTERFERING
            # WITH THE FORCED SPIN-UP DECISION) ###
            return tear_down_actions

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

    def _select_rpu(
        self,
        rel_time_s: float,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> int:
        """
        Select the RPU size for a new cluster based on the current window.
        """
        if self._autoscaling_policy == AutoscalingPolicy.DUPLICATE_LARGEST:
            ready_rpus = [
                cluster_view.rpu
                for cluster_view in pool_snapshot_with_current_query.values()
                if cluster_view.state == ClusterState.READY
            ]
            return (
                max(ready_rpus) if ready_rpus else max(self._allowed_rpu_sizes)
            )

        viol_and_costs: list[ViolationCost] = []
        replay_stats_by_rpu: dict[int, CounterfactualReplayStats] = {}
        total_simulated_queries = 0

        for rpu in self._allowed_rpu_sizes:
            slo_viol_and_cost, replay_stats = self._counterfactual_replay(
                rpu, rel_time_s, pool_snapshot_with_current_query
            )
            viol_and_costs.append(slo_viol_and_cost)
            replay_stats_by_rpu[rpu] = replay_stats
            total_simulated_queries += replay_stats.simulated_queries

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
        self._last_select_rpu_stats = SelectRpuStats(
            total_simulated_queries=total_simulated_queries,
            per_candidate=replay_stats_by_rpu,
        )

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
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> tuple[ViolationCost, CounterfactualReplayStats]:
        """Replay the trailing window with a hypothetical new cluster of
        *candidate_rpu*.

        Two-phase approach:

        1. **Organic phase**: replay window copies with continuous arrivals
           until ``_min_finished_queries_in_counterfactual`` queries that
           both *started* and *finished* after ``hyp_ready_time_s`` have
           completed organically.  A safety cap of ``max_copies`` prevents
           runaway in degenerate inputs.

        2. **Drain phase**: stop accepting new queries; compute cost from the
           current active-query state; drain remaining in-flight queries via
           ``finish_queries_until``; count drain completions (same
           started-after-ready gate) toward the violation metric.

        Only queries whose arrival time is >= ``hyp_ready_time_s`` are counted
        toward violation in either phase.  Returns
        ``ViolationCost(float('inf'), cost)`` if no such completions occur.
        """
        replay_start_rel_time_s = rel_time_s
        local_cluster_pool: dict[str, Cluster] = {
            name: view.to_cluster()
            for name, view in pool_snapshot_with_current_query.items()
        }
        hyp_cluster_name = f"autoslo-{candidate_rpu}-hypothetical"
        local_cluster_pool[hyp_cluster_name] = Cluster(
            creation_time_s=replay_start_rel_time_s,
            rpu=candidate_rpu,
            name=hyp_cluster_name,
            cache_state=np.zeros(self._cluster_cache_state_dim, dtype=float),
            state=ClusterState.PENDING,
        )
        router = QueryRouter(
            slo_resolver=self._slo_resolver,
            slo_objective=self._slo_objective,
            query_router_config=self._query_router_config,
            iconq_model=self._iconq_model,
            out_dir=self._out_dir,
            source_for_log_records="Autoscaler.QueryRouter",
        )

        # Sequential replay.
        lat_and_slos: list[LatencySlo] = []
        hyp_ready_time_s: float | None = None
        finished_after_ready: int = 0
        simulated_queries: int = 0
        replay_copies_run: int = 0

        actual_observed_window_length_s = (
            rel_time_s - self._trailing_queries[0].rel_start_time_s
        )
        base_copies = math.ceil(
            (self._spin_up_delay_s + self._observation_window_s)
            / actual_observed_window_length_s
        )
        max_copies = max(base_copies * 10, 200)

        # --- Organic phase ---
        # Replay window copies with continuous arrivals until enough
        # post-readiness completions have been collected.
        organic_done = False
        last_time_s = replay_start_rel_time_s
        for copy_idx in range(max_copies):
            replay_copies_run = copy_idx + 1
            for query in self._trailing_queries:
                time_s = (
                    query.rel_start_time_s
                    - self._trailing_queries[0].rel_start_time_s
                    + rel_time_s
                    + copy_idx * actual_observed_window_length_s
                )
                last_time_s = time_s
                routed_query = query.copy_with_new_info(
                    f"fwd-{copy_idx}:", time_s
                )

                # Phase transition: activate hyp cluster once spin-up delay
                # has elapsed.
                if (
                    local_cluster_pool[hyp_cluster_name].state
                    == ClusterState.PENDING
                ) and (time_s >= rel_time_s + self._spin_up_delay_s):
                    local_cluster_pool[hyp_cluster_name].update_state(
                        ClusterState.READY
                    )
                    hyp_ready_time_s = time_s
                    if (
                        self._autoscaling_policy
                        == AutoscalingPolicy.REPLACE_WITH_SINGLE_BEST_FORWARD
                    ):
                        for name, c in local_cluster_pool.items():
                            if (
                                name != hyp_cluster_name
                                and c.state == ClusterState.READY
                            ):
                                c.update_state(ClusterState.DRAINING)

                # Expire finished queries. Only count completions for queries
                # that arrived after the hyp cluster became ready.
                for cluster in local_cluster_pool.values():
                    for q, latency_s in cluster.finish_queries_until(
                        rel_time_s=time_s,
                    ):
                        if (
                            hyp_ready_time_s is not None
                            and q.rel_start_time_s >= hyp_ready_time_s
                        ):
                            slo = self._slo_resolver.resolve(q.query_text_id)
                            lat_and_slos.append(LatencySlo(latency_s, slo))
                            finished_after_ready += 1

                if (
                    finished_after_ready
                    >= self._min_finished_queries_in_counterfactual
                ):
                    organic_done = True
                    break

                # Route and update state for the incoming query.
                # Wrap as ClusterViews for the router
                snapshot_for_routing = {
                    cluster_name: ClusterView(cluster)
                    for cluster_name, cluster in local_cluster_pool.items()
                    if cluster.state == ClusterState.READY
                }
                (
                    selected_cluster_name,
                    new_predicted_latencies_for_cluster,
                    new_cache_state,
                ) = router.route_query(
                    query=routed_query,
                    snapshot=snapshot_for_routing,
                    rel_time_s=time_s,
                )
                local_cluster_pool[selected_cluster_name].add_query(
                    routed_query,
                    new_predicted_latencies_for_cluster,
                    new_cache_state,
                )
                simulated_queries += 1

            if organic_done:
                break

        # --- Drain phase ---
        # Also add the lat and slo from any queries still active at the end of
        # the window, and compute cost directly from the replay cluster state
        # using the same model as QueryRouter.
        total_cost = 0.0
        for cluster in local_cluster_pool.values():
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
                current_rel_time_s=last_time_s,
            )

            lat_and_slos.extend(active_pairs)
            total_cost += cluster_cost

        aggregate = self._slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        return ViolationCost(aggregate, total_cost), CounterfactualReplayStats(
            candidate_rpu=candidate_rpu,
            simulated_queries=simulated_queries,
            replay_copies=replay_copies_run,
            finished_after_ready=finished_after_ready,
            organic_done=organic_done,
        )
