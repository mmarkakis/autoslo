import heapq
import logging
from typing import Callable, Optional, Protocol

from autoslo.clusters.actions import ScalingAction
from autoslo.clusters.cluster import ClusterView
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.filesystem.structured_events import EventType, QueryRelatedEvent
from autoslo.filesystem.structured_log import emit_structured
from autoslo.routing.query_router import QueryRouter
from autoslo.workload_definition.query import Query
from autoslo.workload_execution.simulator_event import (
    SimulatorEvent,
    SimulatorEventType,
)

logger = logging.getLogger(__name__)


class _AutoscalerLike(Protocol):
    """Structural interface required by route_and_update_bookkeeping.

    Only inform() is called; the concrete Autoscaler class satisfies this
    protocol, as does any proxy that intercepts inform() calls.
    """

    def inform(
        self,
        rel_time_s: float,
        current_query: Query,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> list[ScalingAction]: ...

class NoOpAutoscaler:
    """An autoscaler that performs no actions, for testing purposes."""

    def inform(
        self,
        rel_time_s: float,
        current_query: Query,
        pool_snapshot_with_current_query: dict[str, ClusterView],
    ) -> list[ScalingAction]:
        return []
    

def route_and_update_bookkeeping(
    source: str,
    rel_time_s_getter: Callable[[], float],
    pool: ManagedClusterPool,
    router: QueryRouter,
    query: Query,
    autoscaler: _AutoscalerLike = NoOpAutoscaler(),
    simulator_pending_events_heap: Optional[list[SimulatorEvent]] = None,
) -> tuple[str, list[ScalingAction]]:

    #  ── Route the query ────────────────────────────────────
    route_start_rel_s = rel_time_s_getter()
    snapshot: dict[str, ClusterView] = pool.snapshot(only_ready=True)

    (
        selected_cluster_name,
        new_predicted_latencies_on_selected,
        new_cluster_cache_state,
        new_lstm_states,
    ) = router.route_query(
        query=query,
        snapshot=snapshot,
        rel_time_s=route_start_rel_s,
    )
    self_latency_s = new_predicted_latencies_on_selected[query.query_id]

    route_end_rel_s = rel_time_s_getter()
    emit_structured(
        QueryRelatedEvent(
            rel_time_s=route_end_rel_s,
            event_type=EventType.QUERY_ROUTED,
            source=source,
            cluster_name=selected_cluster_name,
            details={"latency_s": self_latency_s},
            query_id=query.query_id,
            query_text_id=query.query_text_id,
        )
    )
    if simulator_pending_events_heap is not None:
        heapq.heappush(
            simulator_pending_events_heap,
            SimulatorEvent(
                rel_time_s=query.rel_start_time_s + self_latency_s,
                event_type=SimulatorEventType.QUERY_COMPLETION,
                details={
                    "query_id": query.query_id,
                    "cluster_name": selected_cluster_name,
                    "latency_s": self_latency_s,
                    "query_text_id": query.query_text_id,
                },
            ),
        )

    #  ── Update existing latencies ────────────────────────────────────
    old_latencies = snapshot[selected_cluster_name].predicted_latencies
    for affected_query in snapshot[selected_cluster_name].queries.values():
        old_latency_s = old_latencies.get(affected_query.query_id, None)
        updated_latency_s = new_predicted_latencies_on_selected[
            affected_query.query_id
        ]

        if (old_latency_s is not None) and (
            abs(updated_latency_s - old_latency_s) < 1e-3
        ):
            # No change in latency prediction for this query, so skip the
            # update.
            continue

        emit_structured(
            QueryRelatedEvent(
                rel_time_s=route_end_rel_s,
                event_type=EventType.LATENCY_UPDATE,
                source=source,
                cluster_name=selected_cluster_name,
                details={
                    "old_latency_s": old_latency_s,
                    "latency_s": updated_latency_s,
                },
                query_id=affected_query.query_id,
                query_text_id=affected_query.query_text_id,
            )
        )

        if simulator_pending_events_heap is not None:
            heapq.heappush(
                simulator_pending_events_heap,
                SimulatorEvent(
                    rel_time_s=(
                        affected_query.rel_start_time_s + updated_latency_s
                    ),
                    event_type=SimulatorEventType.QUERY_COMPLETION,
                    details={
                        "query_id": affected_query.query_id,
                        "cluster_name": selected_cluster_name,
                        "latency_s": updated_latency_s,
                        "query_text_id": affected_query.query_text_id,
                    },
                ),
            )

    #  ── Notify pool and autoscaler ────────────────────────────────────
    pool.on_query_start(
        query=query,
        cluster_name=selected_cluster_name,
        new_predicted_latencies_on_selected=new_predicted_latencies_on_selected,
        new_cluster_cache_state=new_cluster_cache_state,
        new_lstm_states=new_lstm_states,
    )
    post_snapshot = pool.snapshot(only_ready=False)
    actions: list[ScalingAction] = []
    try:
        actions = autoscaler.inform(
            rel_time_s=rel_time_s_getter(),
            current_query=query,
            pool_snapshot_with_current_query=post_snapshot,
        )
    except Exception:
        logger.exception(
            "Autoscaler failed after routing query %s; "
            "continuing without scaling actions.",
            query.query_id,
        )

    return selected_cluster_name, actions
