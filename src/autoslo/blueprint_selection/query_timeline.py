from collections import defaultdict
from math import isclose
from typing import Optional

import numpy as np
import torch
from intervaltree import Interval, IntervalTree  # type: ignore[import]

from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.stage_model import StageModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.utils.billing import Billing
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
        skip_neighbors: bool = False,
    ) -> dict[Interval, list[Interval]]:
        """
        For each query that overlaps with the given time window on the specified
        cluster, yield the query and all other queries that overlap with it.

        Parameters:
            cluster_name: The name of the cluster to consider.
            start_time_s: The start time of the window (in seconds).
            end_time_s: The end time of the window (in seconds).
            skip_neighbors: Whether to skip finding overlapping neighbors, i.e.
                just return an empty list for each query.

        Returns:
            A dictionary mapping each query Interval to its list of overlapping
                query Intervals.
        """
        tree = self._interval_trees[cluster_name]
        seeds = list(tree.overlap(start_time_s, end_time_s))

        return {
            a: (
                []
                if skip_neighbors
                else [b for b in tree.overlap(a.begin, a.end) if b != a]
            )
            for a in seeds
        }

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
                skip_neighbors=False,
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
                        ],
                        qb_features=base_iv.data["featurization"],
                        qb_start_time_s=base_iv.begin,
                        qb_latency_prediction=base_iv.data[
                            "stage_model_prediction"
                        ],
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
                            ],
                            qb_features=neighbor_iv.data["featurization"],
                            qb_start_time_s=neighbor_iv.begin,
                            qb_latency_prediction=neighbor_iv.data[
                                "stage_model_prediction"
                            ],
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

    def move_to_cluster(
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

    def find_intervals_by_slo_adherence(
        self,
        slo_s: float | dict[str, float],
        look_for_slo_violations: bool = True,
        weigh_by_distance: bool = True,
    ) -> list[tuple[str, Interval]]:
        """
        Compare each query latency to its SLO. For each cluster, return the
        interval(s) with the maximum total distance from the SLO, either in the
        violation or the slack direction.

        Parameters:
            slo_s: The SLO threshold (in seconds), or a mapping from query ids
                to SLO thresholds.
            look_for_slo_violations: Whether to look for intervals with maximum
                SLO violations (latency > SLO). If False, will look for
                intervals with maximum SLO slack (latency < SLO).
            weigh_by_distance: Whether to weigh queries by their distance from
                the SLO (violation or slack), or just count them equally.

        Returns:
            A list of pairs of cluster name and Interval.
        """

        result: list[tuple[str, Interval]] = []

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

            current_score = 0
            max_score = 0
            max_intervals: list[Interval] = []
            interval_start: Optional[float] = None

            for time, change in events:
                current_score += change

                if current_score > max_score:
                    max_score = current_score
                    max_intervals = []
                    interval_start = time
                elif (current_score == max_score) and (max_score > 0):
                    interval_start = time
                elif interval_start is not None:
                    max_intervals.append(
                        Interval(begin=interval_start, end=time)
                    )
                    interval_start = None

            assert max_score > 0 or len(max_intervals) == 0
            for iv in max_intervals:
                result.append((cluster_name, iv))

        return result

    def total_cost(
        self,
    ) -> float:
        """
        Calculate the total dollar cost of the timeline, considering billing
        thresholds and granularities.

        Returns:
            The total cost of the timeline in dollars.
        """

        total_cost = 0.0
        for cluster_name in self.active_clusters:
            query_intervals = list(self._interval_trees[cluster_name])
            cluster_billed_s = Billing.billed_s(query_intervals=query_intervals)
            cost_per_second = Cluster.from_config(cluster_name).cost_per_second
            cluster_cost = cluster_billed_s * cost_per_second
            total_cost += cluster_cost
        return total_cost
