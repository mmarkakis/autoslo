from collections import defaultdict
from typing import Callable, Optional

from intervaltree import Interval, IntervalTree  # type: ignore[import]

from autoslo.clusters.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.workload_definition.query import ClusterAwareQueryId, Query
from autoslo.workload_execution.trace import Trace


def build_dataset_from_trace(
    trace: Trace,
    iconq_model: IconqModel,
    use_log_runtime: bool = False,
    use_client_side_latencies: bool = False,
    use_fixed_window_radius_s: Optional[float] = None,
    use_fixed_window_max_neighbors_per_side: Optional[int] = None,
    ignore_aborted_queries: bool = False,
) -> ConcurrentQueryDataset:
    """Build a ConcurrentQueryDataset from a Trace for IconqModel training.

    When use_client_side_latencies=True, timing windows, latency targets, and
    is_lower_bound are sourced from the structured log.  Raises ValueError if
    no structured_log.parquet is present for the run.
    """
    cluster_aware_query_ids = trace.cluster_aware_query_ids
    query_text_ids = trace.query_text_ids

    query_featurizer = iconq_model.iconq_query_featurizer
    interaction_featurizer = iconq_model.iconq_interaction_featurizer
    stage_model = iconq_model.stage_model

    if not cluster_aware_query_ids:
        return ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=interaction_featurizer,
            cluster_to_base_to_neighbors={},
        )

    if use_client_side_latencies:
        arrival_s = trace.client_side_arrival_times_s
        completion_s = trace.client_side_completion_times_s
        latencies = trace.client_side_latencies_s
        query_success = trace.structured_log.query_success()  # type: ignore[union-attr]
        reference_s = float(arrival_s.min())

        def _start(qid: ClusterAwareQueryId) -> float:
            return float(arrival_s[qid]) - reference_s

        def _end(qid: ClusterAwareQueryId) -> float:
            return float(completion_s[qid]) - reference_s

        def _latency(qid: ClusterAwareQueryId) -> float:
            return float(latencies[qid])

        def _is_lb(qid: ClusterAwareQueryId) -> bool:
            return not bool(query_success.get(qid.query_id, False))

    else:
        arrival_times = trace.arrival_times()
        completion_times = trace.completion_times()
        was_aborted = trace.was_aborted()
        reference_ts = min(
            arrival_times[qid].timestamp() for qid in cluster_aware_query_ids
        )

        def _start(qid: ClusterAwareQueryId) -> float:  # type: ignore[misc]
            return arrival_times[qid].timestamp() - reference_ts

        def _end(qid: ClusterAwareQueryId) -> float:  # type: ignore[misc]
            return completion_times[qid].timestamp() - reference_ts

        def _latency(qid: ClusterAwareQueryId) -> float:  # type: ignore[misc]
            return (completion_times[qid] - arrival_times[qid]).total_seconds()

        def _is_lb(qid: ClusterAwareQueryId) -> bool:  # type: ignore[misc]
            return bool(was_aborted[qid])

    # Build one IntervalTree per cluster. Each interval's data payload is the
    # fully-featurized Query object, so _find_neighbors can return Query
    # objects directly without a second lookup.
    interval_trees: dict[str, IntervalTree] = defaultdict(IntervalTree)

    for cluster_aware_query_id in cluster_aware_query_ids:
        cluster_name = cluster_aware_query_id.cluster_name
        query_text_id = query_text_ids[cluster_aware_query_id]
        start_s = _start(cluster_aware_query_id)
        end_s = _end(cluster_aware_query_id)
        query = Query(
            query_id=cluster_aware_query_id.query_id,
            query_text_id=query_text_id,
            rel_start_time_s=start_s,
            featurization=query_featurizer.featurize_from_query_text_id(
                query_text_id
            ),
            # Pre-compute stage-model predictions for every allowed RPU size so
            # build_from_query_groups can look up the right one per cluster.
            stage_predictions_per_rpu={
                rpu: float(
                    stage_model.predict_from_query_text_id(
                        {cluster_aware_query_id: query_text_id}, cluster_rpu=rpu
                    )[cluster_aware_query_id].overall_mean_s()
                )
                for rpu in Cluster.ALL_ALLOWED_RPU_SIZES
            },
        )
        interval_trees[cluster_name].add(Interval(start_s, end_s, query))

    cluster_to_base_to_neighbors = _find_neighbors(
        interval_trees,
        use_fixed_window_radius_s,
        use_fixed_window_max_neighbors_per_side,
        _is_lb if ignore_aborted_queries else lambda qid: False,
    )

    targets: dict[ClusterAwareQueryId, float] = {}
    is_lower_bound: dict[ClusterAwareQueryId, bool] = {}
    for qid in cluster_aware_query_ids:
        is_lb = _is_lb(qid)
        if ignore_aborted_queries and is_lb:
            continue
        targets[qid] = _latency(qid)
        is_lower_bound[qid] = is_lb

    return ConcurrentQueryDataset.build_from_query_groups(
        iconq_interaction_featurizer=interaction_featurizer,
        cluster_to_base_to_neighbors=cluster_to_base_to_neighbors,
        targets=targets,
        is_lower_bound=is_lower_bound,
        use_log_runtime=use_log_runtime,
    )


def _find_neighbors(
    interval_trees: dict[str, IntervalTree],
    use_fixed_window_radius_s: Optional[float],
    use_fixed_window_max_neighbors_per_side: Optional[int],
    ignore_as_base: Callable[[ClusterAwareQueryId], bool],
) -> dict[str, dict[Query, list[Query]]]:
    """For each cluster, map every query to its ordered list of neighbors.

    Two neighbor strategies:
    - Overlap-based (use_fixed_window_radius_s is None): neighbors are queries
      whose execution interval overlaps with the base query's interval.
    - Fixed-window: neighbors are queries whose *start time* falls within
      ±use_fixed_window_radius_s of the base query's start time, optionally
      capped to use_fixed_window_max_neighbors_per_side on each side.

    Queries for which ignore_as_base returns True are excluded from the
    base-query set but are still kept as neighbors of other queries.
    """
    result: dict[str, dict[Query, list[Query]]] = {}

    for cluster_name, tree in interval_trees.items():
        base_to_neighbors: dict[Query, list[Query]] = {}

        for iv in sorted(tree, key=lambda x: x.begin):
            neighbor_ivs: list[Interval] = []

            if ignore_as_base(
                ClusterAwareQueryId.make(cluster_name, iv.data.query_id)
            ):
                continue

            if use_fixed_window_radius_s is None:
                # Overlap-based: any interval that shares time with [begin, end).
                neighbor_ivs = sorted(
                    [b for b in tree.overlap(iv.begin, iv.end) if b != iv],
                    key=lambda b: (b.begin, b.end),
                )
            else:
                # Fixed-window: split into before/after by start-time proximity.
                neighbors_before = sorted(
                    [
                        b
                        for b in tree
                        if b.begin < iv.begin
                        and b.begin >= iv.begin - use_fixed_window_radius_s
                    ],
                    key=lambda b: (b.begin, b.end),
                )
                neighbors_after = sorted(
                    [
                        b
                        for b in tree
                        if b.begin > iv.begin
                        and b.begin <= iv.begin + use_fixed_window_radius_s
                    ],
                    key=lambda b: (b.begin, b.end),
                )
                if use_fixed_window_max_neighbors_per_side is not None:
                    # Keep the N closest on each side.
                    neighbors_before = neighbors_before[
                        -use_fixed_window_max_neighbors_per_side:
                    ]
                    neighbors_after = neighbors_after[
                        :use_fixed_window_max_neighbors_per_side
                    ]
                neighbor_ivs = neighbors_before + neighbors_after

            base_to_neighbors[iv.data] = [b.data for b in neighbor_ivs]

        result[cluster_name] = base_to_neighbors

    return result
