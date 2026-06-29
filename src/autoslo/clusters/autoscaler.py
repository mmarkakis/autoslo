import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from autoslo.clusters.actions import ScalingAction, SpinUpAction, TearDownAction
from autoslo.clusters.autoscaling_policy import AutoscalingPolicy
from autoslo.clusters.autoscaling_trigger_policy import AutoscalingTriggerPolicy
from autoslo.clusters.cluster import (
    Cluster,
    ClusterState,
    ClusterView,
    cluster_cost_until_drained,
)
from autoslo.config.component_configs import AutoscalerConfig, ProvisionerConfig
from autoslo.filesystem.structured_events import (
    BaseStructuredEvent,
    EventType,
    QueryRelatedEvent,
)
from autoslo.filesystem.structured_log import emit_structured
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import (
    QueryRouter,
    QueryRouterConfig,
    QueryRouterState,
)
from autoslo.slo.slo_metric import LatencySlo
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoscalerCompletionRecord:
    """
    Internal record of a completed query, for the OBSERVED_VIOLATIONS trigger.
    """

    completion_rel_time_s: float
    arrival_rel_time_s: float
    latency_slo: LatencySlo


@dataclass(frozen=True)
class ReplayEndCheckpoint:
    replay_type: str
    arrivals_processed: int
    pool_snapshot: dict[str, ClusterView]
    query_router_state: QueryRouterState
    next_copy_idx: int = 0
    next_query_idx: int = 0


@dataclass(frozen=True)
class SelectRpuStats:
    pre_spinup_arrivals_processed: int
    post_spinup_arrivals_processed: dict[Optional[int], int]


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
        self._trigger_slo_objective: SloObjective = (
            SloObjective(autoscaler_config.trigger_slo_objective_config)
            if autoscaler_config.trigger_slo_objective_config is not None
            else slo_objective
        )
        self._allowed_rpu_sizes = sorted(autoscaler_config.allowed_rpu_sizes)
        self._iconq_model = (
            iconq_model
            if iconq_model is not None
            else IconqModel.load(
                query_router_config.iconq_model_id, inference_mode=True
            )
        )
        self._query_router_config = query_router_config
        self._out_dir = out_dir
        self._min_cluster_lifetime_s = autoscaler_config.min_cluster_lifetime_s
        self._idle_time_before_tear_down_s = (
            autoscaler_config.idle_time_before_tear_down_s
        )
        self._observation_window_s = autoscaler_config.observation_window_s
        self._slo_tightening_factor = autoscaler_config.slo_tightening_factor
        self._cluster_cache_state_dim = (
            provisioner_config.cluster_cache_state_dim
        )
        self._autoscaling_policy = AutoscalingPolicy(
            autoscaler_config.autoscaling_policy
        )
        self._autoscaling_trigger_policy = AutoscalingTriggerPolicy(
            autoscaler_config.autoscaling_trigger_policy
        )
        self._queue_length_for_trigger_policy: int = (
            autoscaler_config.queue_length_for_trigger_policy
        )
        self._trigger_slo_resolver = slo_resolver.tightened(
            autoscaler_config.slo_tightening_factor
        )
        self._spin_up_delay_s = provisioner_config.spin_up_delay_s
        self._min_finished_queries_in_counterfactual = (
            autoscaler_config.min_finished_queries_in_counterfactual
        )
        self._max_replay_copies = autoscaler_config.max_replay_copies
        self._min_observations_to_act: int = (
            autoscaler_config.min_observations_to_act
        )

        # Internal mutable state (guarded by _lock)
        self._lock = threading.Lock()
        self._trailing_queries: deque[Query] = deque()
        self._trailing_completions: deque[AutoscalerCompletionRecord] = deque()
        self._most_recent_cluster_ready_rel_time_s: float = 0.0
        self._known_ready_cluster_names: frozenset[str] = frozenset()
        self._spin_up_disabled: bool = False
        # True from the moment a SpinUpAction is emitted until the new cluster
        # first appears as READY in the pool.  Prevents duplicate spin-up
        # recommendations during the provisioning window (which can be minutes
        # in live mode, during which no PENDING cluster is visible in the pool).
        self._spin_up_in_flight: bool = False
        # Cooldown timestamp after a "do nothing" decision. Blocks re-evaluation
        # for spin_up_delay_s seconds to prevent the trigger from immediately
        # re-firing when the counterfactual found no useful spinup.
        self._do_nothing_cooldown_until_rel_time_s: float = 0.0

        # Forced mode (set when force_one_decision_after_query_count is
        # provided).
        self._force_one_decision_after_query_count: Optional[int] = (
            force_one_decision_after_query_count
        )
        self._inform_count: int = 0

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
    def slo_tightening_factor(self) -> float:
        return self._slo_tightening_factor

    @property
    def forced_decision_mode(self) -> bool:
        return self._force_one_decision_after_query_count is not None

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

            # Detect new-READY clusters and record when the most recent one
            # became ready. Also clear _spin_up_in_flight: the cluster we were
            # waiting for has arrived.
            current_ready = frozenset(
                name
                for name, cluster in pool_snapshot_with_current_query.items()
                if cluster.state == ClusterState.READY
            )
            if current_ready - self._known_ready_cluster_names:
                self._most_recent_cluster_ready_rel_time_s = rel_time_s
                self._known_ready_cluster_names = current_ready
                self._spin_up_in_flight = False

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
        Recommend spinning up a cluster when the active trigger policy fires.

        Gate 0  – AutoscalingPolicy.NOOP check (backward compatibility).
        Gate 0a – AutoscalingTriggerPolicy.NOOP check.
        Gate 1  – Budget guards (no allowed RPUs / disabled / in-flight).
        Gate 2  – Forced-decision mode (bypass normal trigger checks).
        Gate 3  – Trigger-policy-specific condition check.
        """

        # Gate 0: legacy NOOP guard (backward compatibility with configs that
        # set autoscaling_policy: noop).
        if self._autoscaling_policy == AutoscalingPolicy.NOOP:
            return []
        # Gate 0a: new trigger-policy NOOP check.
        if self._autoscaling_trigger_policy == AutoscalingTriggerPolicy.NOOP:
            return []
        # Gate 1: budget guards.
        if len(self.allowed_rpu_sizes) == 0 or self._spin_up_disabled:
            return []

        # Block if a spin-up is already in flight (not yet reflected in pool).
        # This covers the live-runner case where the cluster goes directly from
        # not-in-pool to READY without ever appearing as PENDING, making the
        # pool-level PENDING check ineffective.
        if self._spin_up_in_flight:
            return []

        # Gate 1b: do-nothing cooldown.
        if rel_time_s < self._do_nothing_cooldown_until_rel_time_s:
            return []

        if self.forced_decision_mode:
            ### IN FORCED MODE: Gate 2 ###
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
            ### NOT IN FORCED MODE: Gate 3 ###
            # Gate 3: trigger-policy-specific condition.  Each policy applies
            # its own post-spinup observation guard internally.
            triggered, reason = self._evaluate_trigger(
                rel_time_s, pool_snapshot_with_current_query
            )
            if not triggered:
                return []

        # Find the best size to spin up.

        best_rpu, _ = self._select_rpu(
            rel_time_s,
            pool_snapshot_with_current_query,
        )
        if best_rpu is None:
            # The counterfactual found no RPU size that beats doing nothing.
            # Apply a cooldown identical to the spin-up delay so the trigger
            # cannot immediately re-fire, then return without spinning up.
            self._do_nothing_cooldown_until_rel_time_s = (
                rel_time_s + self._spin_up_delay_s
            )
            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=rel_time_s,
                    event_type=EventType.SPIN_UP_DECISION,
                    source="Autoscaler",
                    details={
                        "rpu": None,
                        "reason": reason,
                        "autoscaling_policy": self._autoscaling_policy.value,
                        "autoscaling_trigger_policy": (
                            self._autoscaling_trigger_policy.value
                        ),
                    },
                )
            )
            return []
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
                    "autoscaling_trigger_policy": (
                        self._autoscaling_trigger_policy.value
                    ),
                },
            )
        )
        self._spin_up_in_flight = True
        return [action]

    # ------------------------------------------------------------------
    # Trigger-policy helpers
    # ------------------------------------------------------------------

    def _evaluate_trigger(
        self,
        rel_time_s: float,
        pool_snapshot: dict[str, ClusterView],
    ) -> tuple[bool, str]:
        """
        Dispatch to the active trigger policy's evaluation method.

        Returns ``(triggered, reason)`` where *reason* is a human-readable
        string describing why the trigger fired (used in the SpinUpAction
        and the structured log).  *reason* is empty when not triggered.
        """
        if (
            self._autoscaling_trigger_policy
            == AutoscalingTriggerPolicy.PREDICTED_VIOLATIONS
        ):
            return self._check_predicted_violations(rel_time_s, pool_snapshot)
        if (
            self._autoscaling_trigger_policy
            == AutoscalingTriggerPolicy.QUEUE_DEPTH
        ):
            return self._check_queue_depth(pool_snapshot)
        if (
            self._autoscaling_trigger_policy
            == AutoscalingTriggerPolicy.OBSERVED_VIOLATIONS
        ):
            return self._check_observed_violations(rel_time_s)
        if (
            self._autoscaling_trigger_policy
            == AutoscalingTriggerPolicy.COMBINED_VIOLATIONS
        ):
            return self._check_combined_violations(rel_time_s, pool_snapshot)
        # NOOP is handled before _evaluate_trigger is ever called.
        return False, ""

    def _check_predicted_violations(
        self,
        rel_time_s: float,
        pool_snapshot: dict[str, ClusterView],
    ) -> tuple[bool, str]:
        """Trigger when the (tightened) SLO objective is violated over
        currently active queries (predicted latency only), restricted to
        queries that arrived after the most recent cluster became ready."""
        cutoff_s = rel_time_s - self._observation_window_s
        ready_since = self._most_recent_cluster_ready_rel_time_s
        lat_and_slos: list[LatencySlo] = []
        for cluster_name, cluster in pool_snapshot.items():
            for q in cluster.active_queries:
                if q.rel_start_time_s < ready_since:
                    continue
                pred_lat = cluster.predicted_latencies[q.query_id]
                slo = self._trigger_slo_resolver.resolve(q.query_text_id)
                lat_and_slos.append(LatencySlo(pred_lat, slo))
        if not lat_and_slos:
            return False, ""
        n_completions_since_ready = sum(
            1
            for rec in self._trailing_completions
            if rec.arrival_rel_time_s >= ready_since
        )
        total_arrivals_since_ready = (
            len(lat_and_slos) + n_completions_since_ready
        )
        if (
            self._min_observations_to_act > 0
            and total_arrivals_since_ready < self._min_observations_to_act
        ):
            return False, ""
        slo_metric_value = (
            self._trigger_slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        )
        slo_is_met = self._trigger_slo_objective.is_met_from_aggregated(
            slo_metric_value
        )
        if slo_is_met:
            return False, ""
        reason = (
            f"observation_window_start_s={cutoff_s:.4f}, "
            f"most_recent_cluster_ready_rel_time_s={ready_since:.4f}, "
            f"post_spinup_active_queries={len(lat_and_slos)}, "
            f"total_arrivals_since_ready={total_arrivals_since_ready}, "
            f"trigger_slo_metric={self._trigger_slo_objective.slo_metric}, "
            f"slo_metric_value={slo_metric_value:.4f}, "
            f"trigger_slo_threshold="
            f"{self._trigger_slo_objective.slo_threshold:.4f}, "
            f"slo_tightening_factor={self._slo_tightening_factor}"
        )
        return True, reason

    def _check_combined_violations(
        self,
        rel_time_s: float,
        pool_snapshot: dict[str, ClusterView],
    ) -> tuple[bool, str]:
        """Trigger when the (tightened) SLO objective is violated over the
        union of active queries (predicted latency) and completed queries
        within the observation window (actual latency), both restricted to
        queries that arrived after the most recent cluster became ready."""
        cutoff_s = rel_time_s - self._observation_window_s
        ready_since = self._most_recent_cluster_ready_rel_time_s

        # Active queries with predicted latencies.
        active_lat_and_slos: list[LatencySlo] = []
        for cluster_name, cluster in pool_snapshot.items():
            for q in cluster.active_queries:
                if q.rel_start_time_s < ready_since:
                    continue
                pred_lat = cluster.predicted_latencies[q.query_id]
                slo = self._trigger_slo_resolver.resolve(q.query_text_id)
                active_lat_and_slos.append(LatencySlo(pred_lat, slo))

        # Completed queries within the observation window with actual latencies.
        completed_lat_and_slos: list[LatencySlo] = [
            rec.latency_slo
            for rec in self._trailing_completions
            if (rec.completion_rel_time_s >= cutoff_s)
            and (rec.arrival_rel_time_s >= ready_since)
        ]

        lat_and_slos = active_lat_and_slos + completed_lat_and_slos
        if not lat_and_slos:
            return False, ""
        n_completions_since_ready = sum(
            1
            for rec in self._trailing_completions
            if rec.arrival_rel_time_s >= ready_since
        )
        total_arrivals_since_ready = (
            len(active_lat_and_slos) + n_completions_since_ready
        )
        if (
            self._min_observations_to_act > 0
            and total_arrivals_since_ready < self._min_observations_to_act
        ):
            return False, ""
        slo_metric_value = (
            self._trigger_slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        )
        slo_is_met = self._trigger_slo_objective.is_met_from_aggregated(
            slo_metric_value
        )
        if slo_is_met:
            return False, ""
        reason = (
            f"observation_window_start_s={cutoff_s:.4f}, "
            f"most_recent_cluster_ready_rel_time_s={ready_since:.4f}, "
            f"post_spinup_active_queries={len(active_lat_and_slos)}, "
            f"post_spinup_completions_in_window={len(completed_lat_and_slos)}, "
            f"total_observations={len(lat_and_slos)}, "
            f"total_arrivals_since_ready={total_arrivals_since_ready}, "
            f"trigger_slo_metric={self._trigger_slo_objective.slo_metric}, "
            f"slo_metric_value={slo_metric_value:.4f}, "
            f"trigger_slo_threshold="
            f"{self._trigger_slo_objective.slo_threshold:.4f}, "
            f"slo_tightening_factor={self._slo_tightening_factor}"
        )
        return True, reason

    def _check_queue_depth(
        self,
        pool_snapshot: dict[str, ClusterView],
    ) -> tuple[bool, str]:
        """Trigger when the number of active queries that arrived after the
        most recent cluster became ready reaches *queue_length_for_trigger_policy*.
        """
        ready_since = self._most_recent_cluster_ready_rel_time_s
        post_spinup_active = sum(
            sum(
                1
                for q in cluster.active_queries
                if q.rel_start_time_s >= ready_since
            )
            for cluster in pool_snapshot.values()
            if cluster.state == ClusterState.READY
        )
        if post_spinup_active < self._queue_length_for_trigger_policy:
            return False, ""
        reason = (
            f"queue_depth_trigger: "
            f"post_spinup_active_queries={post_spinup_active}, "
            f"queue_length_for_trigger_policy="
            f"{self._queue_length_for_trigger_policy}"
        )
        return True, reason

    def _check_observed_violations(
        self,
        rel_time_s: float,
    ) -> tuple[bool, str]:
        """Trigger when the trigger SLO objective is violated over completed
        queries in the observation window that arrived after the most recent
        cluster became ready."""
        cutoff_s = rel_time_s - self._observation_window_s
        ready_since = self._most_recent_cluster_ready_rel_time_s
        lat_and_slos = [
            rec.latency_slo
            for rec in self._trailing_completions
            if (rec.completion_rel_time_s >= cutoff_s)
            and (rec.arrival_rel_time_s >= ready_since)
        ]
        if not lat_and_slos:
            return False, ""
        total_completions_since_ready = sum(
            1
            for rec in self._trailing_completions
            if rec.arrival_rel_time_s >= ready_since
        )
        if (
            self._min_observations_to_act > 0
            and total_completions_since_ready < self._min_observations_to_act
        ):
            return False, ""
        slo_metric_value = (
            self._trigger_slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        )
        slo_is_met = self._trigger_slo_objective.is_met_from_aggregated(
            slo_metric_value
        )
        if slo_is_met:
            return False, ""
        reason = (
            f"observed_violations_trigger: "
            f"slo_metric={self._trigger_slo_objective.slo_metric}, "
            f"slo_metric_value={slo_metric_value:.4f}, "
            f"trigger_slo_threshold={self._trigger_slo_objective.slo_threshold:.4f}, "
            f"post_spinup_completions_in_window={len(lat_and_slos)}, "
            f"total_completions_since_ready={total_completions_since_ready}, "
            f"observation_window_s={self._observation_window_s}"
        )
        return True, reason

    def record_completion(
        self,
        query: Query,
        latency_s: Optional[float],
        rel_time_s: float,
    ) -> None:
        """Notify the autoscaler that *query* has completed.

        *latency_s* is ``None`` for failed queries, which are treated as SLO
        violations.

        This is only meaningful for the ``OBSERVED_VIOLATIONS``,
        ``PREDICTED_VIOLATIONS``, and ``COMBINED_VIOLATIONS`` trigger policies;
        for all other policies the method returns immediately.

        Must be called from the same serialised context as :meth:`inform`
        (i.e. from the autoscaler background thread in the runner, or from the
        simulator's single-threaded event loop).
        """
        if self._autoscaling_trigger_policy in (
            AutoscalingTriggerPolicy.NOOP,
            AutoscalingTriggerPolicy.QUEUE_DEPTH,
        ):
            return

        with self._lock:
            effective_latency_s = (
                latency_s if latency_s is not None else float("inf")
            )
            slo = self._trigger_slo_resolver.resolve(query.query_text_id)
            self._trailing_completions.append(
                AutoscalerCompletionRecord(
                    completion_rel_time_s=rel_time_s,
                    arrival_rel_time_s=query.rel_start_time_s,
                    latency_slo=LatencySlo(effective_latency_s, slo),
                )
            )
            # Prune records older than the observation window.
            cutoff_s = rel_time_s - self._observation_window_s
            while self._trailing_completions and (
                self._trailing_completions[0].completion_rel_time_s < cutoff_s
            ):
                self._trailing_completions.popleft()

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
    ) -> tuple[Optional[int], SelectRpuStats]:
        """
        Select the RPU size for a new cluster based on the current window.

        Returns ``None`` as the RPU when the no-spinup baseline is at least as
        good as every candidate, signalling that no cluster should be added.
        """
        if self._autoscaling_policy == AutoscalingPolicy.DUPLICATE_LARGEST:
            ready_rpus = [
                cluster_view.rpu
                for cluster_view in pool_snapshot_with_current_query.values()
                if cluster_view.state == ClusterState.READY
            ]
            return (
                max(ready_rpus) if ready_rpus else max(self._allowed_rpu_sizes)
            ), SelectRpuStats(
                pre_spinup_arrivals_processed=0,
                post_spinup_arrivals_processed={
                    rpu: 0 for rpu in self._allowed_rpu_sizes
                },
            )

        viol_and_costs: list[ViolationCost] = []
        post_spinup_replay_end_checkpoints: dict[
            Optional[int], ReplayEndCheckpoint
        ] = {}

        # Run the spin-up-delay portion of the replay once; it is identical
        # for every RPU candidate because the hypothetical cluster is PENDING
        # (excluded from routing) throughout that window.
        query_router = QueryRouter(
            slo_resolver=self._slo_resolver,
            slo_objective=self._slo_objective,
            query_router_config=self._query_router_config,
            iconq_model=self._iconq_model,
            out_dir=self._out_dir,
            source_for_log_records="Autoscaler.QueryRouter",
        )
        initial_checkpoint = ReplayEndCheckpoint(
            replay_type="initial",
            arrivals_processed=0,
            pool_snapshot=pool_snapshot_with_current_query,
            query_router_state=query_router.get_state(),
            next_copy_idx=0,
            next_query_idx=0,
        )
        pre_spinup_replay_end_checkpoint, _ = (
            self._partial_counterfactual_replay(
                query_router=query_router,
                overall_replay_start_rel_time_s=rel_time_s,
                prev_replay_end_checkpoint=initial_checkpoint,
                candidate_rpu=None,
                is_post_spinup=False,
            )
        )

        # Run the no-spinup baseline: same post-spinup window, no new cluster.
        # Placed first in the comparison so that ties resolve to "do nothing".
        baseline_checkpoint, baseline_viol_and_cost = (
            self._partial_counterfactual_replay(
                query_router=query_router,
                overall_replay_start_rel_time_s=rel_time_s,
                prev_replay_end_checkpoint=pre_spinup_replay_end_checkpoint,
                candidate_rpu=None,
                is_post_spinup=True,
            )
        )
        post_spinup_replay_end_checkpoints[None] = baseline_checkpoint
        viol_and_costs.append(baseline_viol_and_cost)
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.RPU_COUNTERFACTUAL,
                source="Autoscaler",
                cluster_name="no-spinup-baseline",
                details={
                    "rpu": None,
                    "slo_violation": baseline_viol_and_cost.violation,
                    "cost": baseline_viol_and_cost.cost,
                    "slo_threshold": self._slo_objective.slo_threshold,
                },
            )
        )

        # Now hypothesize each size.
        for rpu in self._allowed_rpu_sizes:
            post_spinup_replay_end_checkpoint, slo_viol_and_cost = (
                self._partial_counterfactual_replay(
                    query_router=query_router,
                    overall_replay_start_rel_time_s=rel_time_s,
                    prev_replay_end_checkpoint=pre_spinup_replay_end_checkpoint,
                    candidate_rpu=rpu,
                    is_post_spinup=True,
                )
            )
            post_spinup_replay_end_checkpoints[rpu] = (
                post_spinup_replay_end_checkpoint
            )
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

        # Baseline is first so idx_of_best returns it on any tie (prefer
        # doing nothing when no candidate is strictly better).
        best_local_idx = self._slo_objective.idx_of_best(viol_and_costs)
        best_viol_and_cost = viol_and_costs[best_local_idx]

        common_stats = SelectRpuStats(
            pre_spinup_arrivals_processed=(
                pre_spinup_replay_end_checkpoint.arrivals_processed
            ),
            post_spinup_arrivals_processed={
                rpu: post_spinup_replay_end_checkpoints[rpu].arrivals_processed
                for rpu in {self._allowed_rpu_sizes}.union({None})
            },
        )

        if best_local_idx == 0:
            # Baseline is (tied) best — signal "do nothing" to the caller.
            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=rel_time_s,
                    event_type=EventType.RPU_SELECTION,
                    source="Autoscaler",
                    cluster_name="no-spinup-baseline",
                    details={
                        "rpu": None,
                        "slo_violation": baseline_viol_and_cost.violation,
                        "cost": baseline_viol_and_cost.cost,
                        "slo_threshold": self._slo_objective.slo_threshold,
                    },
                )
            )
            return None, common_stats

        # Something beats the baseline.  Among tied non-baseline candidates,
        # pick the largest RPU: when uncertain about size, go conservative.
        best_rpu = max(
            rpu
            for i, rpu in enumerate(self._allowed_rpu_sizes)
            if self._slo_objective.cmp(viol_and_costs[i], best_viol_and_cost)
            == 0
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
        return best_rpu, common_stats

    def _partial_counterfactual_replay(
        self,
        query_router: QueryRouter,
        overall_replay_start_rel_time_s: float,
        prev_replay_end_checkpoint: ReplayEndCheckpoint,
        candidate_rpu: Optional[int] = None,
        is_post_spinup: bool = False,
    ) -> tuple[ReplayEndCheckpoint, Optional[ViolationCost]]:
        """Replay part of the trailing window with a hypothetical new cluster of
        *candidate_rpu*. Allows for independent replay before/after the new
        cluster is available, for efficiency.

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

        # Restore pool and router from the checkpoint.
        local_cluster_pool: dict[str, Cluster] = {
            name: view.to_cluster()
            for name, view in prev_replay_end_checkpoint.pool_snapshot.items()
        }
        query_router.set_state(prev_replay_end_checkpoint.query_router_state)

        # Initialize tracking variables.
        cluster_ready_time_s = (
            overall_replay_start_rel_time_s + self._spin_up_delay_s
        )
        queries_list = list(self._trailing_queries)
        arrivals_processed = 0
        last_seen_rel_start_time_s = overall_replay_start_rel_time_s
        lat_and_slos: list[LatencySlo] = []
        finished_after_ready = 0
        organic_done = False

        # Add the hypothetical cluster for real post-spinup phases only
        # (not the no-spinup baseline).
        if is_post_spinup and (candidate_rpu is not None):
            hyp_cluster_name = f"autoslo-{candidate_rpu}-hypothetical"
            local_cluster_pool[hyp_cluster_name] = Cluster(
                creation_time_s=overall_replay_start_rel_time_s,
                rpu=candidate_rpu,
                name=hyp_cluster_name,
                cache_state=np.zeros(
                    self._cluster_cache_state_dim, dtype=float
                ),
                state=ClusterState.READY,
            )

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

        # Process organic phase arrivals.
        for copy_idx in range(
            prev_replay_end_checkpoint.next_copy_idx,
            self._max_replay_copies,
        ):
            # For the first copy, begin from the query that triggered spinup;
            # subsequent copies replay the full window.
            start_q_idx = (
                prev_replay_end_checkpoint.next_query_idx
                if copy_idx == prev_replay_end_checkpoint.next_copy_idx
                else 0
            )
            for query_idx, query in enumerate(
                queries_list[start_q_idx:], start=start_q_idx
            ):
                rel_start_time_s = (
                    query.rel_start_time_s
                    + (copy_idx + 1) * self._observation_window_s
                )
                last_seen_rel_start_time_s = rel_start_time_s
                routed_query = query.copy_with_new_info(
                    f"fwd-{copy_idx}:", rel_start_time_s
                )

                # If pre-spinup and reached the cluster ready time, return.
                if (
                    not is_post_spinup
                    and rel_start_time_s >= cluster_ready_time_s
                ):
                    # This query is the first one at/after the spin-up
                    # deadline.  Snapshot the pool state and return; the
                    # post-spinup phase will process this query and all
                    # subsequent ones.
                    return (
                        ReplayEndCheckpoint(
                            replay_type="pre_spinup",
                            arrivals_processed=arrivals_processed,
                            pool_snapshot={
                                name: ClusterView.from_cluster(c)
                                for name, c in local_cluster_pool.items()
                            },
                            query_router_state=query_router.get_state(),
                            next_copy_idx=copy_idx,
                            next_query_idx=query_idx,
                        ),
                        None,
                    )

                # Expire finished queries. Only count completions for queries
                # that arrived after the hyp cluster became ready.
                for cluster_name, cluster in local_cluster_pool.items():
                    for q, latency_s in cluster.finish_queries_until(
                        rel_time_s=rel_start_time_s,
                    ):
                        started_after_ready = (
                            q.rel_start_time_s >= cluster_ready_time_s
                        )
                        emit_structured(
                            QueryRelatedEvent(
                                rel_time_s=rel_start_time_s,
                                event_type=EventType.SIM_QUERY_COMPLETION,
                                source="Autoscaler",
                                cluster_name=cluster_name,
                                query_id=q.query_id,
                                query_text_id=q.query_text_id,
                                details={
                                    "latency_s": latency_s,
                                    "phase": (
                                        "post_spinup"
                                        if is_post_spinup
                                        else "pre_spinup"
                                    ),
                                    "started_after_ready": started_after_ready,
                                    "candidate_rpu": candidate_rpu,
                                },
                            )
                        )
                        if started_after_ready:
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
                emit_structured(
                    QueryRelatedEvent(
                        rel_time_s=rel_start_time_s,
                        event_type=EventType.SIM_QUERY_ARRIVAL,
                        source="Autoscaler",
                        query_id=routed_query.query_id,
                        query_text_id=routed_query.query_text_id,
                        details={
                            "copy_idx": copy_idx,
                            "phase": (
                                "post_spinup"
                                if is_post_spinup
                                else "pre_spinup"
                            ),
                            "candidate_rpu": candidate_rpu,
                        },
                    )
                )
                snapshot_for_routing = {
                    cluster_name: ClusterView.from_cluster(cluster)
                    for cluster_name, cluster in local_cluster_pool.items()
                    if cluster.state == ClusterState.READY
                }
                (
                    selected_cluster_name,
                    new_predicted_latencies_for_cluster,
                    new_cache_state,
                    new_lstm_states_for_cluster,
                ) = query_router.route_query(
                    query=routed_query,
                    snapshot=snapshot_for_routing,
                    rel_time_s=rel_start_time_s,
                )
                local_cluster_pool[selected_cluster_name].add_query(
                    routed_query,
                    new_predicted_latencies_for_cluster,
                    new_cache_state,
                    new_lstm_states_for_cluster,
                )
                arrivals_processed += 1

            if organic_done:
                break

        # --- Drain phase ---
        # If we got here pre-spinup, raise error:
        if not is_post_spinup:
            raise RuntimeError(
                f"Replay exhausted {self._max_replay_copies} window copies "
                f"without reaching cluster_ready_time_s="
                f"{cluster_ready_time_s:.3f} s. Max time in replay was "
                f"{last_seen_rel_start_time_s:.3f} s, which is "
                f"{last_seen_rel_start_time_s - overall_replay_start_rel_time_s:.3f} s "
                f"after the start of the replay but "
                f"{cluster_ready_time_s - last_seen_rel_start_time_s:.3f} s "
                f"before the cluster ready time."
            )

        # Also add the lat and slo from any queries still active at the end of
        # the window, and compute cost directly from the replay cluster state
        # using the same model as QueryRouter.
        total_cost = 0.0
        for cluster in local_cluster_pool.values():
            active_pairs = query_router._collect_cluster_pairs(
                queries=cluster.active_queries,
                predicted_latencies=cluster.predicted_latencies,
            )
            cluster_cost = cluster_cost_until_drained(
                queries=cluster.active_queries,
                predicted_latencies=cluster.predicted_latencies,
                billing_accumulator=cluster.billing_accumulator,
                billing_window_start_s=cluster.billing_window_start_s,
                cost_per_second=cluster.cost_per_second,
                current_rel_time_s=last_seen_rel_start_time_s,
            )

            lat_and_slos.extend(active_pairs)
            total_cost += cluster_cost

        aggregate = self._slo_objective.slo_metric.aggregate_batch(lat_and_slos)
        return ReplayEndCheckpoint(
            replay_type="post_spinup",
            arrivals_processed=arrivals_processed,
            pool_snapshot={},  # For efficiency.
            query_router_state=query_router.get_state(),
        ), ViolationCost(aggregate, total_cost)
