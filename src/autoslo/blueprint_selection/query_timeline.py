from math import isclose
from typing import Optional, defaultdict

import numpy as np
import torch
from intervaltree import Interval, IntervalTree

from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.stage_model import StageModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.workload_execution.trace import Trace


class QueryTimeline:
    """Represents a timestamped schedule of query submissions."""

    ONE_MICROSECOND = 1e-6

    def __init__(
        self,
        iconq_query_featurizer: IconqQueryFeaturizer,
        iconq_interaction_featurizer: IconqInteractionFeaturizer,
    ) -> None:
        """
        Initializes the QueryTimeline.

        Parameters:
            iconq_query_featurizer: The IconqQueryFeaturizer to use for
                featurizing the queries.
            iconq_interaction_featurizer: The IconqInteractionFeaturizer to use
                for featurizing interactions between queries.

        """
        self._iconq_query_featurizer = iconq_query_featurizer
        self._iconq_interaction_featurizer = iconq_interaction_featurizer
        self._interval_trees: dict[str, IntervalTree] = defaultdict(
            IntervalTree
        )
        self._query_id_to_cluster_interval: dict[str, tuple[str, Interval]] = {}

    def initialize_from_trace(
        self, trace: Trace, stage_model: Optional[StageModel] = None
    ) -> None:
        """
        Initialize the timeline from a Trace.

        Parameters:
            trace: The Trace containing the query submission events.
            stage_model: The optional model to use for predicting single-query
                latencies.
        """

        tpcds_temp_and_q_idxs = trace.tpcds_temp_and_q_idxs
        start_times = trace.arrival_times()
        end_times = trace.completion_times()
        query_ids = trace.query_ids

        intervals_to_add: dict[str, list[Interval]] = defaultdict(list)

        for query_id in query_ids:
            cluster_name = trace.cluster_name_from_query_id(query_id)
            temp_and_q_idx = tpcds_temp_and_q_idxs[query_id]

            interval = Interval(
                begin=start_times[query_id].timestamp(),
                end=end_times[query_id].timestamp(),
                data={
                    "query_id": query_id,
                    "tpcds_temp_and_q_idx": temp_and_q_idx,
                    "featurization": self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                        temp_and_q_idx
                    ),
                    "stage_model_prediction": (
                        stage_model.predict_from_tpcds_temp_and_q_idx(
                            {query_id: temp_and_q_idx}
                        )[query_id]
                        if stage_model is not None
                        else 0.0
                    ),
                },
            )

            self._query_id_to_cluster_interval[query_id] = (
                cluster_name,
                interval,
            )
            intervals_to_add[cluster_name].append(interval)

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
    def active_clusters(self) -> list[str]:
        """
        Get the list of active cluster names in the timeline.

        Returns:
            A list of active cluster names.
        """
        return list(self._interval_trees.keys())

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

    def add_query(
        self,
        cluster_name: str,
        start_time_s: float,
        end_time_s: float,
        query_id: str,
        tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx,
        stage_model_prediction: float,
    ) -> None:
        """
        Add a query to the timeline.

        Parameters:
            cluster_name: The name of the cluster where the query is executed.
            start_time_s: The start time of the query (in seconds).
            end_time_s: The end time of the query (in seconds).
            query_id: The ID of the query.
            tpcds_temp_and_q_idx: The TPC-DS template and query index tuple.
            stage_model_prediction: The stage model latency prediction for
                this query (in seconds).
        """

        interval = Interval(
            begin=start_time_s,
            end=end_time_s,
            data={
                "query_id": query_id,
                "tpcds_temp_and_q_idx": tpcds_temp_and_q_idx,
                "featurization": self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                    tpcds_temp_and_q_idx
                ),
                "stage_model_prediction": stage_model_prediction,
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
    ) -> dict[Interval, list[Interval]]:
        """
        For each query that overlaps with the given time window on the specified
        cluster, yield the query and all other queries that overlap with it.

        Parameters:
            cluster_name: The name of the cluster to consider.
            start_time_s: The start time of the window (in seconds).
            end_time_s: The end time of the window (in seconds).

        Returns:
            A dictionary mapping each query Interval to its list of overlapping
                query Intervals.
        """
        tree = self._interval_trees[cluster_name]
        seeds = list(tree.overlap(start_time_s, end_time_s))

        result: dict[Interval, list[Interval]] = {}

        for a in seeds:
            result[a] = [b for b in tree.overlap(a.begin, a.end) if b != a]

        return result

    def get_dataset(
        self,
        start_time_s: float = -float("inf"),
        end_time_s: float = float("inf"),
        use_log_runtime: bool = True,
    ) -> ConcurrentQueryDataset:
        """
        Get a ConcurrentQueryDataset representing the timeline.

        Parameters:
            start_time_s: The start time of the window (in seconds).
            end_time_s: The end time of the window (in seconds).
            use_log_runtime: Whether to use the log of the runtime as the
                target variable (log1p), or the runtime itself.

        Returns:
            A ConcurrentQueryDataset representing the timeline.
        """
        x = []
        y = []
        pinch_points = []
        query_ids = []

        for cluster_name in self.active_clusters:

            query_overlap_mapping = self.queries_in_window(
                cluster_name=cluster_name,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )

            for base_iv, overlapping_ivs in query_overlap_mapping.items():

                interaction_featurizations: dict[
                    float,
                    IconqInteractionFeaturizer.IconqInteractionFeaturization,
                ] = {}

                # Add oneself to the interaction featurizations. This helps with
                # queries that do not have any overlapping neighbors.
                interaction_featurizations[base_iv.begin] = (
                    self._iconq_interaction_featurizer.featurize_from_vectors(
                        cluster_name=cluster_name,
                        qa_features=base_iv.data["featurization"],
                        qa_start_time_s=base_iv.begin,
                        qa_latency_prediction=base_iv.data[
                            "stage_model_prediction"
                        ].overall_mean_s(),
                        qb_features=base_iv.data["featurization"],
                        qb_start_time_s=base_iv.begin,
                        qb_latency_prediction=base_iv.data[
                            "stage_model_prediction"
                        ].overall_mean_s(),
                    )
                )

                # Collect the interaction featurizations with overlapping nodes.
                for neighbor_iv in overlapping_ivs:

                    interaction_featurizations[neighbor_iv.begin] = (
                        self._iconq_interaction_featurizer.featurize_from_vectors(
                            cluster_name=cluster_name,
                            qa_features=base_iv.data["featurization"],
                            qa_start_time_s=base_iv.begin,
                            qa_latency_prediction=base_iv.data[
                                "stage_model_prediction"
                            ].overall_mean_s(),
                            qb_features=neighbor_iv.data["featurization"],
                            qb_start_time_s=neighbor_iv.begin,
                            qb_latency_prediction=neighbor_iv.data[
                                "stage_model_prediction"
                            ].overall_mean_s(),
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

        # Transform lists into tensors.
        x_tensorized = x
        pinch_points_tensorized = torch.tensor(pinch_points, dtype=torch.int8)
        y_tensorized = torch.tensor(y, dtype=torch.float32)

        dataset = ConcurrentQueryDataset(
            x=x_tensorized,
            pinch_points=pinch_points_tensorized,
            y=y_tensorized,
            query_ids=query_ids,
        )

        return dataset

    def update_latency(
        self,
        query_id: str,
        latency_s: float,
    ) -> bool:
        """
        Update the latency of a query in the timeline.

        Parameters:
            query_id: The ID of the query to update.
            latency_s: The new latency of the query (in seconds).

        Returns:
            Whether the end time of the query changed.

        Raises:
            ValueError: If the query ID does not exist in the timeline.
        """
        if query_id not in self.query_ids:
            raise ValueError(f"Query ID {query_id} does not exist in timeline.")

        cluster_name, interval = self._query_id_to_cluster_interval[query_id]
        if isclose(
            interval.end - interval.begin,
            latency_s,
            abs_tol=QueryTimeline.ONE_MICROSECOND,
        ):
            return False

        self._interval_trees[cluster_name].remove(interval)
        new_interval = Interval(
            begin=interval.begin,
            end=interval.begin + latency_s,
            data=interval.data,
        )
        self._interval_trees[cluster_name].add(new_interval)
        self._query_id_to_cluster_interval[query_id] = (
            cluster_name,
            new_interval,
        )

        return True

    def move_to_cluster(self, new_cluster_name: str, query_id: str) -> None:
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
        self._interval_trees[new_cluster_name].add(interval)
        self._query_id_to_cluster_interval[query_id] = (
            new_cluster_name,
            interval,
        )
