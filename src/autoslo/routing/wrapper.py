import heapq
import logging
from typing import Callable, Optional

from autoslo.clusters.actions import ScalingAction, SpinUpAction, TearDownAction
from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter
from autoslo.utils.logging import emit_structured
from autoslo.utils.structured_events import EventType, QueryRelatedEvent
from autoslo.workload_definition.query import Query
from autoslo.workload_execution.simulator_event import (
    SimulatorEvent,
    SimulatorEventType,
)

logger = logging.getLogger(__name__)


def route_and_update_bookkeeping(
    source: str,
    rel_time_s_getter: Callable[[], float],
    pool: ManagedClusterPool,
    router: QueryRouter,
    query: Query,
    iconq_model: IconqModel,
    autoscaler: Autoscaler,
    on_spin_up: Callable[[SpinUpAction], None],
    write_text_log: bool = False,
    simulator_pending_events_heap: Optional[list[SimulatorEvent]] = None,
) -> str:

    #  ── Route the query ────────────────────────────────────
    route_start_rel_s = rel_time_s_getter()
    snapshot = pool.snapshot(only_ready=True)
    old_predicted_latencies = {
        cluster_name: dict(cluster.predicted_latencies)
        for cluster_name, cluster in snapshot.items()
    }

    selected_cluster_name, new_predicted_latencies_on_selected = (
        router.route_query(
            query=query,
            clusters=snapshot,
            iconq_model=iconq_model,
            rel_time_s=route_start_rel_s,
        )
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
                rel_time_s=route_end_rel_s + self_latency_s,
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
    old_latencies = old_predicted_latencies.get(selected_cluster_name, {})
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
                    rel_time_s=route_end_rel_s + updated_latency_s,
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
    )
    post_snapshot = pool.snapshot(only_ready=False)
    try:
        autoscaler_suggested_actions: list[ScalingAction] = autoscaler.inform(
            rel_time_s=rel_time_s_getter(),
            current_query=query,
            pool_snapshot_with_current_query=post_snapshot,
        )
        for action in autoscaler_suggested_actions:
            if isinstance(action, SpinUpAction):
                on_spin_up(action)
            elif isinstance(action, TearDownAction):
                pool.request_tear_down(action, rel_time_s_getter())
            elif write_text_log:
                logger.warning(
                    f"Unknown autoscaling action type: {type(action)}"
                )
    except Exception:
        logger.exception(
            "Autoscaler failed after routing query %s; "
            "continuing without scaling actions.",
            query.query_id,
        )

    return selected_cluster_name
