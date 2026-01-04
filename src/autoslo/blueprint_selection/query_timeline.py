import bisect
import heapq
from typing import Optional, Union

import torch
import numpy as np

import networkx as nx

from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.stage_model import StageModel
from autoslo.workload_execution.trace import Trace

from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset


class QueryTimeline:
    """Represents a timestamped schedule of query submissions."""

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
        self._overlap_graph: nx.Graph = nx.Graph()
        self._ordered_start_times_s: list[tuple[float, str]] = []

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

        for query_id in query_ids:
            self._overlap_graph.add_node(
                query_id,
                query_id=query_id,
                cluster_name=trace.cluster_name_from_query_id(query_id),
                start_time_s=start_times[query_id].timestamp(),
                end_time_s=end_times[query_id].timestamp(),
                tpcds_temp_and_q_idx=tpcds_temp_and_q_idxs[query_id],
                featurization=(
                    self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                        tpcds_temp_and_q_idxs[query_id]
                    )
                ),
                stage_model_prediction=(
                    stage_model.predict_from_tpcds_temp_and_q_idx(
                        {query_id: tpcds_temp_and_q_idxs[query_id]}
                    )[query_id]
                    if stage_model is not None
                    else 0.0
                ),
                ignored=False,
            )

        self._ordered_start_times_s.extend(
            sorted(
                [
                    (data["start_time_s"], node)
                    for node, data in self._overlap_graph.nodes(data=True)
                ]
            )
        )  # FIXME: In theory there is an edge case where two queries have
        #  same start time and the node name sort order gets messed up,
        # but this is unlikely in practice.

        self._compute_graph_edges() 

    @property
    def query_ids(self) -> list[str]:
        """
        Get the list of query IDs in the timeline.

        Returns:
            A list of query IDs.
        """
        return list(self._overlap_graph.nodes())

    def _compute_graph_edges(self) -> None:
        """
        Compute the edges of the overlap graph upon intialization.

        Raises:
            ValueError: If the overlap graph already has edges.
        """
        # Make sure there are no edges.
        if self._overlap_graph.number_of_edges() > 0:
            raise ValueError(
                "Overlap graph already has edges; cannot recompute edges."
            )

        # Re-add edges based on current start and end times
        active_query_ids: list[tuple[float, str, str]] = []
        sorted_queries = sorted(
            self._overlap_graph.nodes(data=True),
            key=lambda x: (x[1]["start_time_s"], x[1]["end_time_s"]),
        )

        for current_query_id, data in sorted_queries:

            current_start_time_s = data["start_time_s"]
            current_end_time_s = data["end_time_s"]
            current_cluster_name = data["cluster_name"]

            # Remove queries that have ended before the current query starts
            while (
                len(active_query_ids) > 0
                and active_query_ids[0][0] <= current_start_time_s
            ):
                heapq.heappop(active_query_ids)

            # Add edges to all currently active queries on the same cluster.
            for (
                _,
                other_query_id,
                other_cluster_name,
            ) in active_query_ids:
                if other_cluster_name == current_cluster_name:
                    self._overlap_graph.add_edge(
                        current_query_id, other_query_id
                    )

            # Add the current query to the list of active queries
            heapq.heappush(
                active_query_ids,
                (current_end_time_s, current_query_id, current_cluster_name),
            )

    def overlap(self, query_id_a: str, query_id_b: str) -> bool:
        """
        Check if the execution of two queries overlaps on the same cluster.

        Parameters:
            query_id_a: The ID of the first query.
            query_id_b: The ID of the second query.

        Returns:
            True if the queries overlap on the same cluster, False otherwise.
        """
        return self._overlap_graph.has_edge(query_id_a, query_id_b)


    def get_dataset(self, use_log_runtime: bool) -> ConcurrentQueryDataset:
        """
        Get a ConcurrentQueryDataset representing the timeline.

        Parameters:
            use_log_runtime: Whether to use the log of the runtime as the
                target variable (log1p), or the runtime itself.

        Returns:
            A ConcurrentQueryDataset representing the timeline.
        """
        x = []
        y = []
        pinch_points = []
        query_id_hashes = []

        for node, node_data in self._overlap_graph.nodes(data=True):
            if node_data['ignored']:
                continue

            interaction_featurizations: dict[
                float, IconqInteractionFeaturizer.IconqInteractionFeaturization
            ] = {}

            # Add oneself to the interaction featurizations. This helps with
            # queries that do not have any overlapping neighbors.
            interaction_featurizations[node_data["start_time_s"]] = (
                self._iconq_interaction_featurizer.featurize_from_vectors(
                    qa_features=node_data["featurization"],
                    qa_start_time_s=node_data["start_time_s"],
                    qa_latency_prediction=node_data[
                        "stage_model_prediction"
                    ].overall_mean_s(),
                    qb_features=node_data["featurization"],
                    qb_start_time_s=node_data["start_time_s"],
                    qb_latency_prediction=node_data[
                        "stage_model_prediction"
                    ].overall_mean_s(),
                )
            )

            # Collect the interaction featurizations with neighboring nodes, as
            # long as they execute on the same cluster.
            for neighbor in self._overlap_graph.neighbors(node):
                neighbor_data = self._overlap_graph.nodes[neighbor]
                if neighbor_data["cluster_name"] != node_data["cluster_name"]:
                    continue
                interaction_featurizations[neighbor_data["start_time_s"]] = (
                    self._iconq_interaction_featurizer.featurize_from_vectors(
                        qa_features=node_data["featurization"],
                        qa_start_time_s=node_data["start_time_s"],
                        qa_latency_prediction=node_data[
                            "stage_model_prediction"
                        ].overall_mean_s(),
                        qb_features=neighbor_data["featurization"],
                        qb_start_time_s=neighbor_data["start_time_s"],
                        qb_latency_prediction=neighbor_data[
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
                            interaction_featurizations[neighbor_start_time_s],
                            dtype=torch.float32,
                        )
                        for neighbor_start_time_s in neighbor_sort_order
                    ]
                )
            )
            latency = node_data["end_time_s"] - node_data["start_time_s"]
            y.append(latency if not use_log_runtime else np.log1p(latency))
            pinch_points.append(
                neighbor_sort_order.index(node_data["start_time_s"])
            )
            query_id_hashes.append(Trace.hash_query_id(node_data["query_id"]))

        # Transform lists into tensors.
        x_tensorized = x
        pinch_points_tensorized = torch.tensor(pinch_points, dtype=torch.int8)
        y_tensorized = torch.tensor(y, dtype=torch.float32)
        query_id_hashes_tensorized = torch.tensor(
            query_id_hashes, dtype=torch.int64
        )

        dataset = ConcurrentQueryDataset(
            x=x_tensorized,
            pinch_points=pinch_points_tensorized,
            y=y_tensorized,
            query_id_hashes=query_id_hashes_tensorized,
        )

        return dataset


    def update_latency(
        self,
        query_id: str,
        latency_s: float,
    ) -> None:
        """
        Update the latency of a query in the timeline.

        Parameters:
            query_id: The ID of the query to update.
            latency_s: The new latency of the query (in seconds).
        Raises:
            ValueError: If the query ID does not exist in the timeline.
        """
        if query_id not in self._overlap_graph:
            raise ValueError(f"Query ID {query_id} does not exist in timeline.")

        # Determine new end time
        start_time_s = self._overlap_graph.nodes[query_id]["start_time_s"]
        old_end_time_s = self._overlap_graph.nodes[query_id]["end_time_s"]
        new_end_time_s = start_time_s + latency_s
        if new_end_time_s == old_end_time_s:
            return
        self._overlap_graph.nodes[query_id]["end_time_s"] = new_end_time_s

        # If it got shorter, we only need to remove edges
        if new_end_time_s < old_end_time_s:
            original_neighbors = list(self._overlap_graph.neighbors(query_id))
            for neighbor in original_neighbors:
                neighbor_start_time_s = self._overlap_graph.nodes[neighbor][
                    "start_time_s"
                ]
                if neighbor_start_time_s >= new_end_time_s:
                    self._overlap_graph.remove_edge(query_id, neighbor)

        else:
            # If it got longer, we need to add edges for anyone starting
            # during the extended period.
            consideration_period_start_idx = bisect.bisect_left(
                self._ordered_start_times_s, (old_end_time_s, "")
            )
            for i in range(
                consideration_period_start_idx, len(self._ordered_start_times_s)
            ):
                other_start_time_s, other_query_id = (
                    self._ordered_start_times_s[i]
                )
                if other_start_time_s >= new_end_time_s:
                    break
                if other_query_id == query_id:
                    continue
                self._overlap_graph.add_edge(query_id, other_query_id)

    def toggle_ignored_to(
        self, ignored: bool, query_ids: Union[str, list[str]]
    ) -> None:
        """
        Toggle the ignored status of one or more queries.

        Parameters:
            ignored: The new ignored status.
            query_ids: The ID or list of IDs of the queries to update.
        """
        if isinstance(query_ids, str):
            query_ids = [query_ids]

        for query_id in query_ids:
            if query_id not in self._overlap_graph:
                raise ValueError(
                    f"Query ID {query_id} does not exist in timeline."
                )
            self._overlap_graph.nodes[query_id]["ignored"] = ignored
