import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from intervaltree import Interval, IntervalTree  # type: ignore[import]
from matplotlib.patches import Rectangle

from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.nn.concurrent_query_dataset import (
    ConcurrentQueryDataset,
    QueryInfo,
)
from autoslo.utils.billing import Billing
from autoslo.workload_execution.trace import Trace


@dataclass
class QueryMove:
    """
    Represents a move of a query from one cluster to another.
    """

    query_id: str
    seq_num: int
    from_cluster_name: str
    to_cluster_name: str

    def inverse(self) -> "QueryMove":
        return QueryMove(
            query_id=self.query_id,
            seq_num=self.seq_num,
            from_cluster_name=self.to_cluster_name,
            to_cluster_name=self.from_cluster_name,
        )

    def pretty_print(self) -> str:
        return (
            f"{Trace.redshift_query_id_from_query_id(self.query_id)} "
            f"(Seq Num {self.seq_num}): "
            f"{self.from_cluster_name} -> "
            f"{self.to_cluster_name}"
        )


class QueryTimeline:
    """Represents a timestamped schedule of query submissions."""

    ONE_MICROSECOND = 1e-6

    def __init__(
        self,
        iconq_model: IconqModel,
    ) -> None:
        """
        Initializes the QueryTimeline.

        Parameters:
            iconq_model: The IconqModel to use for predictions.

        """
        self._iconq_model = iconq_model
        self._iconq_query_featurizer = self._iconq_model.iconq_query_featurizer
        self._iconq_interaction_featurizer = (
            self._iconq_model.iconq_interaction_featurizer
        )
        self._stage_model = self._iconq_model.stage_model
        self._interval_trees: dict[str, IntervalTree] = defaultdict(
            IntervalTree
        )
        self._query_id_to_cluster_interval: dict[str, tuple[str, Interval]] = {}

    def initialize_from_trace(
        self,
        trace: Trace,
    ) -> None:
        """
        Initialize the timeline from a Trace.

        Parameters:
            trace: The Trace containing the query submission events.
        """

        tpcds_temp_and_q_idxs = trace.tpcds_temp_and_q_idxs
        start_times = trace.arrival_times()
        end_times = trace.completion_times()
        query_ids = trace.query_ids
        seq_nums = trace.seq_nums

        reference_timestamp = min(
            start_times[query_id].timestamp() for query_id in query_ids
        )

        intervals_to_add: dict[str, list[Interval]] = defaultdict(list)

        ordered_cluster_names_per_rpu = Cluster.ordered_cluster_names_per_rpu()

        for query_id in query_ids:
            observed_cluster_name = trace.cluster_name_from_query_id(query_id)
            temp_and_q_idx = tpcds_temp_and_q_idxs[query_id]
            seq_num = seq_nums[query_id]

            interval = Interval(
                begin=start_times[query_id].timestamp() - reference_timestamp,
                end=end_times[query_id].timestamp() - reference_timestamp,
                data={
                    "query_id": query_id,
                    "tpcds_temp_and_q_idx": temp_and_q_idx,
                    "seq_num": int(seq_num),
                    "featurization": self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                        temp_and_q_idx
                    ),
                    "stage_model_predictions_per_rpu": {
                        rpu: (
                            float(
                                self._stage_model.predict_from_tpcds_temp_and_q_idx(
                                    {query_id: temp_and_q_idx}, cluster_names[0]
                                )[
                                    query_id
                                ].overall_mean_s()
                            )
                        )
                        for rpu, cluster_names in ordered_cluster_names_per_rpu.items()
                    },
                },
            )

            self._query_id_to_cluster_interval[query_id] = (
                observed_cluster_name,
                interval,
            )
            intervals_to_add[observed_cluster_name].append(interval)

        for cluster_name, intervals in intervals_to_add.items():
            self._interval_trees[cluster_name].update(intervals)

    @property
    def query_ids(self) -> list[str]:
        """
        Get the list of query IDs in the timeline.

        Returns:
            A list of query IDs.
        """
        return list(self._query_id_to_cluster_interval.keys())

    @property
    def num_queries(self) -> int:
        """
        Get the number of queries in the timeline.

        Returns:
            The number of queries.
        """
        return len(self._query_id_to_cluster_interval)

    @property
    def active_clusters(self) -> list[str]:
        """
        Get the list of active cluster names in the timeline.

        Returns:
            A list of active cluster names.
        """
        return list(self._interval_trees.keys())

    def query_ids_to_cluster_names(self) -> dict[str, str]:
        """
        Get a mapping from query IDs to their corresponding cluster names.

        Returns:
            A dictionary mapping query IDs to cluster names.
        """
        return {
            query_id: cluster_name
            for query_id, (
                cluster_name,
                _,
            ) in self._query_id_to_cluster_interval.items()
        }

    def seq_num_to_cluster_name(self) -> dict[int, str]:
        """
        Get a mapping from query sequence numbers to their corresponding
        cluster names.

        Returns:
            A dictionary mapping query sequence numbers to cluster names.
        """
        return {
            interval.data["seq_num"]: cluster_name
            for query_id, (
                cluster_name,
                interval,
            ) in self._query_id_to_cluster_interval.items()
        }

    def interval_for_query_id(self, query_id: str) -> Interval:
        """
        Get the Interval for a given query ID.

        Parameters:
            query_id: The ID of the query.

        Returns:
            The Interval corresponding to the query ID.
        """
        _, interval = self._query_id_to_cluster_interval[query_id]
        return interval

    def overlap(self, query_id_a: str, query_id_b: str) -> bool:
        """
        Check if the execution of two queries overlaps on the same cluster.

        Parameters:
            query_id_a: The ID of the first query.
            query_id_b: The ID of the second query.

        Returns:
            True if the queries overlap on the same cluster, False otherwise.
        """
        cluster_name_a, interval_a = self._query_id_to_cluster_interval[
            query_id_a
        ]
        cluster_name_b, interval_b = self._query_id_to_cluster_interval[
            query_id_b
        ]
        if cluster_name_a != cluster_name_b:
            return False

        return interval_a.overlaps(interval_b)

    def predict_all(
        self,
        use_stage_for_isolated_queries: bool = False,
    ) -> dict[str, ModelPrediction]:
        """
        Predicts the runtimes for the queries in the given QueryTimeline,
        taking into account their overlaps, unless they are ignored.

        Parameters:
            query_timeline: The QueryTimeline to predict runtimes for.
            use_stage_for_isolated_queries: Whether to use the StageModel for
                queries that do not overlap with any other queries.

        Returns:
            A dictionary mapping query IDs to their predicted ModelPrediction.
        """

        dataset = self.get_dataset(
            use_log_runtime=self._iconq_model.trained_on_log_runtime,
            use_fixed_window_radius_s=self._iconq_model._init_config.use_fixed_window_radius_s,
            use_fixed_window_max_neighbors_per_side=self._iconq_model._init_config.use_fixed_window_max_neighbors_per_side,
        )
        return self._iconq_model.predict_from_dataset(
            dataset,
            use_stage_for_isolated_queries=use_stage_for_isolated_queries,
        )

    def add_query(
        self,
        cluster_name: str,
        start_time_s: float,
        end_time_s: float,
        query_id: str,
        seq_num: int,
        tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx,
    ) -> None:
        """
        Add a query to the timeline.

        Parameters:
            cluster_name: The name of the cluster where the query is executed.
            start_time_s: The start time of the query (in seconds).
            end_time_s: The end time of the query (in seconds).
            query_id: The ID of the query.
            seq_num: The sequence number of the query.
            tpcds_temp_and_q_idx: The TPC-DS template and query index tuple.
        """

        ordered_cluster_names_per_rpu = Cluster.ordered_cluster_names_per_rpu()

        interval = Interval(
            begin=start_time_s,
            end=end_time_s,
            data={
                "query_id": query_id,
                "tpcds_temp_and_q_idx": tpcds_temp_and_q_idx,
                "seq_num": int(seq_num),
                "featurization": self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                    tpcds_temp_and_q_idx
                ),
                "stage_model_predictions_per_rpu": {
                    rpu: (
                        float(
                            self._stage_model.predict_from_tpcds_temp_and_q_idx(
                                {query_id: tpcds_temp_and_q_idx},
                                cluster_names[0],
                            )[query_id].overall_mean_s()
                        )
                    )
                    for rpu, cluster_names in ordered_cluster_names_per_rpu.items()
                },
            },
        )

        self._query_id_to_cluster_interval[query_id] = (
            cluster_name,
            interval,
        )
        self._interval_trees[cluster_name].add(interval)

    def queries_in_window(
        self,
        cluster_name: str,
        start_time_s: float = -float("inf"),
        end_time_s: float = float("inf"),
        skip_neighbors: bool = False,
        use_fixed_window_radius_s: Optional[float] = None,
        use_fixed_window_max_neighbors_per_side: Optional[int] = None,
    ) -> dict[Interval, list[Interval]]:
        """
        For each query that overlaps with the given time window on the specified
        cluster, yield a mapping from the query Interval to a list of its
        neighboring query Intervals.

        "Neighboring" queries can be defined in two ways:
        - If use_fixed_window_radius_s is None, neighboring queries are those
            that overlap in time with the base query, based on the latencies
            currently stored in the timeline.
        - If use_fixed_window_radius_s is provided, neighboring queries are
            those that start within the fixed time radius (in seconds) around
            the base query's start time, regardless of whether they overlap.
            Optionally, the number of neighbors on each side of the base query
            can be limited, using use_fixed_window_max_neighbors_per_side.

        Parameters:
            cluster_name: The name of the cluster to consider.
            start_time_s: The start time of the window (in seconds).
            end_time_s: The end time of the window (in seconds).
            skip_neighbors: Whether to skip finding overlapping neighbors, i.e.
                just return an empty list for each query.
            use_fixed_window_radius_s: Optional fixed time radius (in seconds)
                around the base query's start time to consider neighbors.
            use_fixed_window_max_neighbors_per_side: Optional maximum number of
                neighbors to include on each side of the base query when using
                a fixed window radius.
        Returns:
            A dictionary mapping each query Interval to its list of neighboring
                query Intervals. The neighboring intervals will be sorted by
                start time.
        """
        tree = self._interval_trees[cluster_name]

        overlap_res = (
            tree.overlap(start_time_s, end_time_s)
            if end_time_s > start_time_s
            else tree.at(start_time_s)
        )

        intervals: dict[Interval, list[Interval]] = {
            x: [] for x in sorted(overlap_res, key=lambda iv: iv.begin)
        }

        if skip_neighbors:
            return intervals

        if use_fixed_window_radius_s is None:
            for a in intervals.keys():
                intervals[a] = sorted(
                    [b for b in tree.overlap(a.begin, a.end) if b != a],
                    key=lambda iv: iv.begin,
                )
            return intervals

        for a in intervals.keys():
            neighbors_before = sorted(
                [
                    b
                    for b in tree
                    if b.begin < a.begin
                    and b.begin >= a.begin - use_fixed_window_radius_s
                ],
                key=lambda iv: iv.begin,
            )
            neighbors_after = sorted(
                [
                    b
                    for b in tree
                    if b.begin > a.begin
                    and b.begin <= a.begin + use_fixed_window_radius_s
                ],
                key=lambda iv: iv.begin,
            )
            if use_fixed_window_max_neighbors_per_side is not None:
                neighbors_before = neighbors_before[
                    -use_fixed_window_max_neighbors_per_side:
                ]
                neighbors_after = neighbors_after[
                    :use_fixed_window_max_neighbors_per_side
                ]
            intervals[a] = neighbors_before + neighbors_after
        return intervals

    @staticmethod
    def _pretty_print_queries_in_window(
        query_overlap_mapping: dict[Interval, list[Interval]],
    ) -> str:
        result_d = {}
        for base_iv, overlapping_ivs in query_overlap_mapping.items():
            result_d[base_iv.data["seq_num"]] = (
                f"Query {Trace.redshift_query_id_from_query_id(base_iv.data['query_id'])} "
                f"(Seq Num {base_iv.data['seq_num']}) "
                f"({base_iv.begin:.4f}s - {base_iv.end:.4f}s) overlaps with: ["
            )
            for neighbor_iv in overlapping_ivs:
                result_d[base_iv.data["seq_num"]] += (
                    f"  - Query {Trace.redshift_query_id_from_query_id(neighbor_iv.data['query_id'])} "
                    f"(Seq Num {neighbor_iv.data['seq_num']}) "
                    f"({neighbor_iv.begin:.4f}s - {neighbor_iv.end:.4f}s)\n"
                )
            result_d[base_iv.data["seq_num"]] += "]\n"
        result = ""
        for seq_num in sorted(result_d.keys()):
            result += result_d[seq_num]
        return result

    def get_dataset(
        self,
        start_time_s: float = -float("inf"),
        end_time_s: float = float("inf"),
        use_log_runtime: bool = True,
        run_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        use_fixed_window_radius_s: Optional[float] = None,
        use_fixed_window_max_neighbors_per_side: Optional[int] = None,
    ) -> ConcurrentQueryDataset:
        """
        Get a ConcurrentQueryDataset representing the timeline.

        Parameters:
            start_time_s: The start time of the window (in seconds).
            end_time_s: The end time of the window (in seconds).
            use_log_runtime: Whether to use the log of the runtime as the
                target variable (log1p), or the runtime itself.
            run_id: The run ID associated with this dataset, if any.
            cluster_name: If provided, only include queries from this cluster.
            use_fixed_window_radius_s: If provided, include neighbors within
                this time radius (in seconds) around each query, regardless of
                whether they overlap.
            use_fixed_window_max_neighbors_per_side: If provided, and
                use_fixed_window_radius_s is also provided, limit the number of
                neighbors on each side of the query (before and after) to this
                number.


        Returns:
            A ConcurrentQueryDataset representing the timeline.
        """
        x = []
        y = []
        pinch_points = []
        query_ids = []
        tpcds_temp_and_q_idx = []

        included_clusters = (
            [cluster_name] if cluster_name is not None else self.active_clusters
        )

        for cluster_name_local in included_clusters:

            cluster_rpu = Cluster.from_config(cluster_name_local).rpu

            query_neighbor_mapping = self.queries_in_window(
                cluster_name=cluster_name_local,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                skip_neighbors=False,
                use_fixed_window_radius_s=use_fixed_window_radius_s,
                use_fixed_window_max_neighbors_per_side=(
                    use_fixed_window_max_neighbors_per_side
                ),
            )

            for base_iv, neighbor_ivs in query_neighbor_mapping.items():

                interaction_featurizations: dict[
                    float,
                    IconqInteractionFeaturizer.IconqInteractionFeaturization,
                ] = {}

                # Add oneself to the interaction featurizations. This helps with
                # queries that do not have any neighbors.
                interaction_featurizations[base_iv.begin] = (
                    self._iconq_interaction_featurizer.featurize_from_vectors(
                        cluster_name=cluster_name_local,
                        qa_features=base_iv.data["featurization"],
                        qa_start_time_s=base_iv.begin,
                        qa_latency_prediction=base_iv.data[
                            "stage_model_predictions_per_rpu"
                        ][cluster_rpu],
                        qb_features=base_iv.data["featurization"],
                        qb_start_time_s=base_iv.begin,
                        qb_latency_prediction=base_iv.data[
                            "stage_model_predictions_per_rpu"
                        ][cluster_rpu],
                    )
                )

                # Collect the interaction featurizations with neighboring nodes.
                for neighbor_iv in neighbor_ivs:

                    interaction_featurizations[neighbor_iv.begin] = (
                        self._iconq_interaction_featurizer.featurize_from_vectors(
                            cluster_name=cluster_name_local,
                            qa_features=base_iv.data["featurization"],
                            qa_start_time_s=base_iv.begin,
                            qa_latency_prediction=base_iv.data[
                                "stage_model_predictions_per_rpu"
                            ][cluster_rpu],
                            qb_features=neighbor_iv.data["featurization"],
                            qb_start_time_s=neighbor_iv.begin,
                            qb_latency_prediction=neighbor_iv.data[
                                "stage_model_predictions_per_rpu"
                            ][cluster_rpu],
                        )
                    )
                neighbor_sort_order = sorted(interaction_featurizations.keys())

                # Update the tensors.
                x.append(
                    torch.stack(
                        [
                            torch.tensor(
                                interaction_featurizations[
                                    neighbor_start_time_s
                                ],
                                dtype=torch.float32,
                            )
                            for neighbor_start_time_s in neighbor_sort_order
                        ]
                    )
                )
                latency = base_iv.end - base_iv.begin
                y.append(latency if not use_log_runtime else np.log1p(latency))
                pinch_points.append(neighbor_sort_order.index(base_iv.begin))
                query_ids.append(base_iv.data["query_id"])
                tpcds_temp_and_q_idx.append(
                    base_iv.data["tpcds_temp_and_q_idx"]
                )

        # Transform lists into tensors.
        x_tensorized = x
        pinch_points_tensorized = torch.tensor(pinch_points, dtype=torch.int8)
        y_tensorized = torch.tensor(y, dtype=torch.float32)

        # Intialize the run ids.
        if run_id is not None:
            run_ids = [run_id for _ in range(len(query_ids))]
        else:
            run_ids = ["unknown" for _ in range(len(query_ids))]

        dataset = ConcurrentQueryDataset(
            x=x_tensorized,
            pinch_points=pinch_points_tensorized,
            y=y_tensorized,
            query_ids=query_ids,
            tpcds_temp_and_q_idx=tpcds_temp_and_q_idx,
            run_ids=run_ids,
        )

        return dataset

    @staticmethod
    def _maybe_log(message: str, verbose: bool):
        if verbose:
            logging.info(message)

    def apply_move(
        self,
        move: QueryMove,
        injected_latencies: Optional[dict[str, float]] = None,
        verbose: bool = False,
        use_stage_for_isolated_queries: bool = False,
    ) -> tuple[QueryMove, dict[str, float]]:
        """
        Apply a query move, and return appropriate information to invert it.

        Parameters:
            move: The QueryMove to apply.
            injected_latencies: Optional mapping from query IDs to their new
                latencies (in seconds) to update after the move. If not provided,
                updated latencies will be predicted using the IconqModel.
            verbose: Whether to log verbose output.
            use_stage_for_isolated_queries: Whether to use the StageModel for
                isolated queries when predicting latencies.

        Returns:
            inverse_move: The inverse of the applied move.
            old_latencies: A mapping from query IDs to their old
                latencies (in seconds).
        """
        # Bookkeeping for return values.
        inverse_move = move.inverse()
        old_latencies = {}

        self._maybe_log(f"Applying move: {move.pretty_print()}", verbose)
        # Move to the new cluster and update the latency to be the defualt from
        # Stage.
        self._move_to_cluster(
            new_cluster_name=move.to_cluster_name, query_id=move.query_id
        )

        # Maybe we have injected latencies to use directly.
        if injected_latencies is not None:
            log_message = f"Updated latencies after move (injected): \n"

            for q_id, new_latency_s in injected_latencies.items():
                old_latency_s = self.update_latency(
                    query_id=q_id,
                    latency_s=new_latency_s,
                )
                log_message += (
                    f"Query {Trace.redshift_query_id_from_query_id(q_id)}"
                    f" (Seq Num {self._query_id_to_cluster_interval[q_id][1].data['seq_num']}): "
                    f"{old_latency_s:.4f}s -> {new_latency_s:.4f}s, "
                )
            self._maybe_log(log_message, verbose)
            return inverse_move, old_latencies

        self._maybe_log(
            "No injected latencies provided, predicting updated latencies using IconqModel.",
            verbose,
        )

        stage_pred_on_new_cluster = self._query_id_to_cluster_interval[
            move.query_id
        ][1].data["stage_model_predictions_per_rpu"][
            Cluster.rpu_for_cluster_name(move.to_cluster_name)
        ]

        old_latencies[move.query_id] = self.update_latency(
            query_id=move.query_id,
            latency_s=stage_pred_on_new_cluster,
        )
        self._maybe_log(
            (
                f"Immediately after move, we updated latency of moved query "
                f"{move.query_id} (Seq Num {move.seq_num}) from "
                f"{old_latencies[move.query_id]:.3f}s to "
                f"{stage_pred_on_new_cluster:.3f}s"
            ),
            verbose,
        )

        # First update latencies on the new cluster for the moved query.
        interval = self.interval_for_query_id(query_id=move.query_id)
        self._maybe_log(
            f"Predicting latencies for interval ({interval.begin}, {interval.end}) on cluster {move.to_cluster_name} after move.",
            verbose,
        )
        log_message = f"Updated latencies after move: \n"

        # Do the queries active before the incoming one.
        dataset = self.get_dataset(
            start_time_s=interval.begin,
            end_time_s=interval.begin,
            use_log_runtime=self._iconq_model._trained_on_log_runtime,
            cluster_name=move.to_cluster_name,
            use_fixed_window_radius_s=self._iconq_model._init_config.use_fixed_window_radius_s,
            use_fixed_window_max_neighbors_per_side=self._iconq_model._init_config.use_fixed_window_max_neighbors_per_side,
        )
        predictions = self._iconq_model.predict_from_dataset(
            dataset=dataset,
            use_stage_for_isolated_queries=use_stage_for_isolated_queries,
        )
        self._maybe_log(
            f"Predictions: {self._pretty_print_predictions(predictions)}",
            verbose,
        )
        for q_id, prediction in predictions.items():
            if q_id not in old_latencies:
                predicted_latency_s = prediction.overall_mean_s()

                old_latency_s = self.update_latency(
                    query_id=q_id,
                    latency_s=predicted_latency_s,
                    only_update_if_increased=True,  # Can only make past queries slower
                )
                new_latency_s = max(old_latency_s, predicted_latency_s)
                log_message += (
                    f"Query {Trace.redshift_query_id_from_query_id(q_id)}"
                    f" (Seq Num {self._query_id_to_cluster_interval[q_id][1].data['seq_num']}): "
                    f"{old_latency_s:.4f}s -> {new_latency_s:.4f}s, "
                )

                old_latencies[q_id] = old_latency_s

        # Do the queries active after the incoming one. For ease, we just
        # do all the overlapping ones again and then don't update the ones we
        # already did.
        dataset = self.get_dataset(
            start_time_s=interval.begin,
            end_time_s=interval.end,
            use_log_runtime=self._iconq_model._trained_on_log_runtime,
            cluster_name=move.to_cluster_name,
            use_fixed_window_radius_s=self._iconq_model._init_config.use_fixed_window_radius_s,
            use_fixed_window_max_neighbors_per_side=self._iconq_model._init_config.use_fixed_window_max_neighbors_per_side,
        )
        predictions = self._iconq_model.predict_from_dataset(
            dataset=dataset,
            use_stage_for_isolated_queries=use_stage_for_isolated_queries,
        )
        self._maybe_log(
            f"Predictions: {self._pretty_print_predictions(predictions)}",
            verbose,
        )
        for q_id, prediction in predictions.items():
            if q_id not in old_latencies:
                predicted_latency_s = prediction.overall_mean_s()

                old_latency_s = self.update_latency(
                    query_id=q_id,
                    latency_s=predicted_latency_s,
                    only_update_if_increased=False,  # These can go either way
                )
                new_latency_s = predicted_latency_s
                log_message += (
                    f"Query {Trace.redshift_query_id_from_query_id(q_id)}"
                    f" (Seq Num {self._query_id_to_cluster_interval[q_id][1].data['seq_num']}): "
                    f"{old_latency_s:.4f}s -> {new_latency_s:.4f}s, "
                )
                old_latencies[q_id] = old_latency_s

        # Now update latencies on the old cluster for any affected queries.
        original_end_time = interval.begin + old_latencies[move.query_id]
        self._maybe_log(
            f"Predicting latencies for interval ({interval.begin}, {original_end_time}) on cluster {move.from_cluster_name} after move.",
            verbose,
        )
        dataset = self.get_dataset(
            start_time_s=interval.begin,
            end_time_s=original_end_time,
            use_log_runtime=self._iconq_model._trained_on_log_runtime,
            cluster_name=move.from_cluster_name,
            use_fixed_window_radius_s=self._iconq_model._init_config.use_fixed_window_radius_s,
            use_fixed_window_max_neighbors_per_side=self._iconq_model._init_config.use_fixed_window_max_neighbors_per_side,
        )
        if not dataset.query_ids or len(dataset.query_ids) == 0:
            self._maybe_log(
                f"No relevant queries on cluster {move.from_cluster_name} after move.",
                verbose,
            )
        else:
            predictions = self._iconq_model.predict_from_dataset(
                dataset=dataset,
                use_stage_for_isolated_queries=use_stage_for_isolated_queries,
            )
            self._maybe_log(
                f"Predictions: {self._pretty_print_predictions(predictions)}",
                verbose,
            )
            for q_id, prediction in predictions.items():

                if q_id not in old_latencies:
                    predicted_latency_s = prediction.overall_mean_s()

                    old_latency_s = self.update_latency(
                        query_id=q_id,
                        latency_s=predicted_latency_s,
                        only_update_if_increased=False,  # These can go either way
                    )
                    new_latency_s = predicted_latency_s
                    log_message += (
                        f"Query {Trace.redshift_query_id_from_query_id(q_id)}"
                        f" (Seq Num {self._query_id_to_cluster_interval[q_id][1].data['seq_num']}): "
                        f"{old_latency_s:.4f}s -> {new_latency_s:.4f}s, "
                    )
                    old_latencies[q_id] = old_latency_s

        self._maybe_log(log_message, verbose)
        return inverse_move, old_latencies

    def _pretty_print_predictions(
        self, predictions: dict[str, ModelPrediction]
    ) -> str:
        result_d = {}

        for query_id, prediction in predictions.items():
            seq_num = self._query_id_to_cluster_interval[query_id][1].data[
                "seq_num"
            ]
            result_d[seq_num] = (
                f"Query {Trace.redshift_query_id_from_query_id(query_id)} "
                f"(Seq Num {seq_num}): "
                f"{prediction.overall_mean_s():.4f}s, "
            )

        result = ""
        for seq_num in sorted(result_d.keys()):
            result += result_d[seq_num]
        return result

    def update_latency(
        self,
        query_id: str,
        latency_s: float,
        only_update_if_increased: bool = False,
    ) -> float:
        """
        Update the latency of a query in the timeline.

        Parameters:
            query_id: The ID of the query to update.
            latency_s: The new latency of the query (in seconds).
            only_update_if_increased: If True, only update the latency if the
                new latency is greater than the old latency.

        Returns:
            The old latency of the query (in seconds).

        Raises:
            ValueError: If the query ID does not exist in the timeline.
        """
        if query_id not in self.query_ids:
            raise ValueError(f"Query ID {query_id} does not exist in timeline.")
        if latency_s < 0:
            raise ValueError("Latency must be non-negative.")

        cluster_name, old_interval = self._query_id_to_cluster_interval[
            query_id
        ]
        old_latency_s = old_interval.end - old_interval.begin

        if only_update_if_increased and latency_s <= old_latency_s:
            return old_latency_s

        new_interval = Interval(
            begin=old_interval.begin,
            end=old_interval.begin + latency_s,
            data=old_interval.data,
        )

        self._interval_trees[cluster_name].remove(old_interval)
        self._interval_trees[cluster_name].add(new_interval)
        self._query_id_to_cluster_interval[query_id] = (
            cluster_name,
            new_interval,
        )

        return old_latency_s

    def _move_to_cluster(
        self,
        new_cluster_name: str,
        query_id: str,
    ) -> None:
        """
        Move a query to a different cluster.

        Parameters:
            new_cluster_name: The name of the new cluster.
            query_id: The ID of the query to move.
        """
        old_cluster_name, interval = self._query_id_to_cluster_interval[
            query_id
        ]
        self._interval_trees[old_cluster_name].remove(interval)
        if len(self._interval_trees[old_cluster_name]) == 0:
            del self._interval_trees[old_cluster_name]
        self._interval_trees[new_cluster_name].add(interval)
        self._query_id_to_cluster_interval[query_id] = (
            new_cluster_name,
            interval,
        )

    def slo_violation_rate(
        self,
        slo_s: float | dict[str, float],
    ) -> float:
        """
        Calculate the SLO violation rate for the timeline.

        Parameters:
            slo_s: The SLO threshold (in seconds), or a mapping from query ids
                to SLO thresholds.

        Returns:
            The SLO violation rate as a float between 0 and 1.
        """
        total_queries = 0
        violating_queries = 0

        for query_id, (
            _,
            interval,
        ) in self._query_id_to_cluster_interval.items():
            total_queries += 1
            latency = interval.end - interval.begin
            if isinstance(slo_s, dict):
                slo_for_query = slo_s[query_id]
            else:
                slo_for_query = slo_s
            if latency > slo_for_query:
                violating_queries += 1

        if total_queries == 0:
            return 0.0

        return violating_queries / total_queries

    def slo_violation_amount(
        self,
        slo_s: float | dict[str, float],
    ) -> float:
        """
        Calculate the cumulative SLO violation amount for the timeline.

        Parameters:
            slo_s: The SLO threshold (in seconds), or a mapping from query ids
                to SLO thresholds.

        Returns:
            The cumulative SLO violation amount in seconds.
        """
        total_violation_amount = 0.0

        for query_id, (
            _,
            interval,
        ) in self._query_id_to_cluster_interval.items():
            latency = interval.end - interval.begin
            if isinstance(slo_s, dict):
                slo_for_query = slo_s[query_id]
            else:
                slo_for_query = slo_s
            violation_amount = latency - slo_for_query
            if violation_amount > 0:
                total_violation_amount += violation_amount

        return total_violation_amount

    def find_intervals_by_slo_adherence(
        self,
        slo_s: float | dict[str, float],
        look_for_slo_violations: bool = True,
        weigh_by_distance: bool = True,
        k: int = 5,
    ) -> list[tuple[str, Interval]]:
        """
        Compare each query latency to its SLO. Across clusters, return the top
        k cluster-interval pairs with the maximum total distance from the SLO,
        either in the violation or the slack direction.

        Parameters:
            slo_s: The SLO threshold (in seconds), or a mapping from query ids
                to SLO thresholds.
            look_for_slo_violations: Whether to look for intervals with maximum
                SLO violations (latency > SLO). If False, will look for
                intervals with maximum SLO slack (latency < SLO).
            weigh_by_distance: Whether to weigh queries by their distance from
                the SLO (violation or slack), or just count them equally.
            k: The maximum number of top intervals to return.

        Returns:
            A list of pairs of cluster name and Interval, in decreasing order
            of total distance from the SLO.
        """

        result: list[tuple[float, str, Interval]] = []

        for cluster_name in self.active_clusters:
            tree = self._interval_trees[cluster_name]
            events: list[tuple[float, int]] = []

            for iv in tree:
                latency = iv.end - iv.begin
                if isinstance(slo_s, dict):
                    slo_for_query = slo_s[iv.data["query_id"]]
                else:
                    slo_for_query = slo_s
                violation_amount = latency - slo_for_query

                is_violation = violation_amount > 0
                if look_for_slo_violations != is_violation:
                    continue

                contribution = abs(violation_amount) if weigh_by_distance else 1
                events.append((iv.begin, +contribution))
                events.append((iv.end, -contribution))

            events.sort()

            score_to_interval: dict[float, Interval] = {}
            previous_score = 0.0
            previous_time = None

            for time, change in events:
                if previous_time is not None:
                    score_to_interval[previous_score] = Interval(
                        begin=previous_time, end=time
                    )
                previous_score += change
                previous_time = time

            sorted_scores = sorted(score_to_interval.keys(), reverse=True)
            for score in sorted_scores[:k]:
                result.append((score, cluster_name, score_to_interval[score]))

        result.sort(key=lambda x: x[0], reverse=True)

        return [
            (cluster_name, interval) for _, cluster_name, interval in result[:k]
        ]

    def total_cost(self) -> float:
        """
        Calculate the total dollar cost of the timeline, considering billing
        thresholds and granularities.

        Returns:
            The total cost of the timeline in dollars.
        """

        total_cost = 0.0
        for cluster_name in self.active_clusters:
            total_cost += self.cost_for_cluster(cluster_name)
        return total_cost

    def cost_per_cluster(
        self,
    ) -> dict[str, float]:
        """
        Calculate the total dollar cost per cluster in the timeline,
        considering billing thresholds and granularities.

        Returns:
            A dictionary mapping cluster names to their total cost in dollars.
        """
        return {
            cluster_name: self.cost_for_cluster(cluster_name)
            for cluster_name in self.active_clusters
        }

    def cost_for_cluster(
        self,
        cluster_name: str,
    ) -> float:
        """
        Calculate the total dollar cost of a specific cluster in the timeline,
        considering billing thresholds and granularities.

        Parameters:
            cluster_name: The name of the cluster.

        Returns:
            The total cost of the cluster in dollars.
        """
        if cluster_name not in self.active_clusters:
            return 0.0

        query_intervals = list(self._interval_trees[cluster_name])
        cluster_billed_s = Billing.billed_s(query_intervals=query_intervals)
        cost_per_second = Cluster.from_config(cluster_name).cost_per_second
        cluster_cost = cluster_billed_s * cost_per_second
        return cluster_cost

    def draw_gantt_chart(
        self,
        slo_s: float | dict[str, float],
        path: Optional[str] = None,
    ) -> None:

        # Simple Gantt chart of query assignments over time. Have a horizontal
        # "lane" per cluster, and plot each query as a line segment. Make sure
        # that line segments are offset vertically per cluster so that they don't
        # overlap. Have the cluster IDs be strings on the y axis, not numbers, and
        # include the number of their rpus.
        fig, ax = plt.subplots(figsize=(12, 6))
        y_ticks = []
        y_labels = []
        y_pos = 0

        # Before plotting, compute for each cluster how much vertical space its
        # lane needs, so that the queries will not visually overlap but we also
        # won't take up excessive vertical space. That is, make sure to reuse
        # vertical space within a cluster as much as possible.
        min_time = float("inf")
        max_time = -float("inf")

        for cluster_name in self.active_clusters:
            tree = self._interval_trees[cluster_name]
            sorted_intervals = sorted(
                [(iv.begin, iv.end, iv.data["query_id"]) for iv in tree]
            )
            min_time = min(min_time, sorted_intervals[0][0])
            max_time = max(max_time, sorted_intervals[-1][1])

            # Greedily assign intervals to lanes
            lanes: list[list[tuple[float, float, str]]] = []
            for interval in sorted_intervals:
                placed = False
                for lane in lanes:
                    if all(
                        not (
                            interval[0] < existing[1]
                            and interval[1] > existing[0]
                        )
                        for existing in lane
                    ):
                        lane.append(interval)
                        placed = True
                        break
                if not placed:
                    lanes.append([interval])

            # Plot the queries. For each query, color any part of it that violates the SLO red.
            for lane_idx, lane in enumerate(lanes):
                for s, e, query_id in lane:
                    rel_s = s - min_time
                    rel_e = e - min_time

                    slo_rel_e = rel_s + (
                        slo_s if isinstance(slo_s, float) else slo_s[query_id]
                    )
                    if rel_e > slo_rel_e:
                        # Plot violation part in red.
                        ax.add_patch(
                            Rectangle(
                                (rel_s, y_pos + lane_idx - 0.4),
                                slo_rel_e - rel_s,
                                0.8,
                                facecolor="green",
                                alpha=0.6,
                            )
                        )
                        ax.add_patch(
                            Rectangle(
                                (slo_rel_e, y_pos + lane_idx - 0.4),
                                rel_e - slo_rel_e,
                                0.8,
                                facecolor="red",
                                alpha=1,
                            )
                        )

                    else:
                        # Plot entire query in blue.
                        ax.add_patch(
                            Rectangle(
                                (rel_s, y_pos + lane_idx - 0.4),
                                rel_e - rel_s,
                                0.8,
                                edgecolor="black",
                                facecolor="green",
                                alpha=0.6,
                            )
                        )
                    # Outline
                    ax.add_patch(
                        Rectangle(
                            (rel_s, y_pos + lane_idx - 0.4),
                            rel_e - rel_s,
                            0.8,
                            edgecolor="black",
                            facecolor="none",
                            alpha=1,
                        )
                    )

            # Label the cluster on the y axis
            y_ticks.append(y_pos + (len(lanes) - 1) / 2)
            y_labels.append(
                f"{cluster_name} "
                f"(RPU {Cluster.rpu_for_cluster_name(cluster_name)})"
            )
            y_pos += len(lanes) + 1  # add space between clusters

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time since start (s)")
        ax.set_xlim(left=-1, right=max_time - min_time + 1)
        ax.set_ylim(bottom=-1, top=y_pos)
        ax.set_title("Cluster Query Assignments Over Time")
        ax.grid(True)

        if path is not None:
            plt.savefig(path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
