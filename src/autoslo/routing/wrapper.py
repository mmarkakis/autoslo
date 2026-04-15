import heapq
import logging
from typing import Callable, Optional

from autoslo.clusters.actions import ScalingAction, SpinUpAction, TearDownAction
from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter
from autoslo.utils.logging import LOGGER_NAME, emit_structured
from autoslo.workload_definition.query import Query
from autoslo.workload_execution.simulator_event import (
    SimulatorEvent,
    SimulatorEventType,
)

logger = logging.getLogger(__name__)
_has_structured = lambda: bool(logging.getLogger(LOGGER_NAME).handlers)


def route_and_update_bookkeeping(
    source: str,
    current_time_getter: Callable[[], float],
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
    route_start_ts = current_time_getter()
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
            current_time_s=route_start_ts,
        )
    )
    self_latency_s = new_predicted_latencies_on_selected[query.query_id]
    route_end_ts = current_time_getter()

    if _has_structured():
        emit_structured(
            {
                "timestamp": route_end_ts,
                "event_type": "query_routed",
                "query_id": query.query_id,
                "query_text_id": query.query_text_id.value,
                "cluster_name": selected_cluster_name,
                "old_latency_s": None,
                "raw_model_latency_s": None,
                "latency_s": self_latency_s,
                "end_time_s": route_end_ts + self_latency_s,
                "source": source,
            }
        )

    #  ── Update existing latencies ────────────────────────────────────
    old_predicted_latencies_on_selected = old_predicted_latencies.get(
        selected_cluster_name, {}
    )
    for qid, latency_s in new_predicted_latencies_on_selected.items():
        old_latency_s = old_predicted_latencies_on_selected.get(qid, None)

        if (old_latency_s is not None) and (
            abs(latency_s - old_latency_s) < 1e-3
        ):
            # No change in latency prediction for this query, so skip the
            # update.
            continue

        completion_time_s = route_end_ts + latency_s
        if _has_structured():
            emit_structured(
                {
                    "timestamp": route_end_ts,
                    "event_type": "latency_update",
                    "source": source,
                    "query_id": qid,
                    "cluster_name": selected_cluster_name,
                    "old_latency_s": old_latency_s,
                    "latency_s": latency_s,
                    "end_time_s": completion_time_s,
                }
            )

        if simulator_pending_events_heap is not None:
            heapq.heappush(
                simulator_pending_events_heap,
                SimulatorEvent(
                    rel_time_s=completion_time_s,
                    event_type=SimulatorEventType.QUERY_COMPLETION,
                    details={
                        "query_id": qid,
                        "cluster_name": selected_cluster_name,
                        "latency_s": latency_s,
                        "query_text_id": query.query_text_id,
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
            current_time_s=current_time_getter(),
            current_query=query,
            pool_snapshot_with_current_query=post_snapshot,
        )
        for action in autoscaler_suggested_actions:
            if isinstance(action, SpinUpAction):
                on_spin_up(action)
            elif isinstance(action, TearDownAction):
                pool.request_tear_down(action, current_time_getter())
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
