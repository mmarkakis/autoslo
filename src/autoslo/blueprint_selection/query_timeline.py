import bisect
import heapq
from typing import Optional
from dataclasses import dataclass

import networkx as nx

from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_execution.trace import Trace


@dataclass
class IngestedQuery:
    """
    Represents a query in the timeline, for ingestion into the QueryTimeline.
    """

    query_id: str
    start_time_s: float
    end_time_s: float
    tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx


class QueryTimeline:
    """Represents a timestamped schedule of query submissions."""

    def __init__(
        self,
        iconq_query_featurizer: IconqQueryFeaturizer,
    ) -> None:
        """
        Initializes the QueryTimeline.

        Parameters:
            iconq_query_featurizer: The IconqQueryFeaturizer to use for
                featurizing the queries.

        """
        self._iconq_query_featurizer = iconq_query_featurizer
        self._overlap_graph: nx.Graph
        self._ordered_start_times_s: list[tuple[float, str]]

    def initialize_from_trace(self, trace: Trace) -> None:
        """
        Initializes the QueryTimeline from a Trace.

        Parameters:
            trace: The Trace containing the query submission events.
        """

        tpcds_temp_and_q_idxs = trace.tpcds_temp_and_q_idxs
        start_times = trace.arrival_times()
        end_times = trace.completion_times()
        query_ids = trace.query_ids

        ingested_queries: list[IngestedQuery] = []
        for query_id in query_ids:
            ingested_queries.append(
                IngestedQuery(
                    query_id=query_id,
                    start_time_s=start_times[query_id].timestamp(),
                    end_time_s=end_times[query_id].timestamp(),
                    tpcds_temp_and_q_idx=tpcds_temp_and_q_idxs[query_id],
                )
            )

        self._overlap_graph = self._build_overlap_graph(ingested_queries)
        self._ordered_start_times_s = sorted(
            [
                (data["start_time_s"], node)
                for node, data in self._overlap_graph.nodes(data=True)
            ]
        )  # FIXME: In theory there is an edge case where two queries have
        #  same start time and the node name sort order gets messed up,
        # but this is unlikely in practice.

    def overlap_graph(self) -> nx.Graph:
        """
        Returns the overlap graph representing query overlaps in time.

        Returns:
            The overlap graph as a NetworkX Graph.
        """
        return self._overlap_graph

    def _build_overlap_graph(
        self, ingested_queries: list[IngestedQuery]
    ) -> nx.Graph:
        """
        Get a representation of the overlaps between the given queries. Each
        node is a query, and there is an edge beetween query A and query B if
        they overlap in time.

        Parameters:
            ingested_queries: The input queries.

        Returns:
            A NetworkX Graph representing the overlaps between queries.
        """

        G: nx.Graph = nx.Graph()

        for ingested_query in ingested_queries:
            G.add_node(
                ingested_query.query_id,
                query_id=ingested_query.query_id,
                start_time_s=ingested_query.start_time_s,
                end_time_s=ingested_query.end_time_s,
                tpcds_temp_and_q_idx=ingested_query.tpcds_temp_and_q_idx,
                featurization=(
                    self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                        ingested_query.tpcds_temp_and_q_idx
                    )
                ),
            )

        self._compute_graph_edges(G)

        return G

    def _compute_graph_edges(self, G: Optional[nx.Graph] = None) -> None:
        """
        Recompute the edges of the overlap graph based on the current start
        and end times of the queries. Only considers queries within the given
        time window.

        Parameters:
            earliest_time: The earliest time (Unix timestamp) to consider.
            latest_time: The latest time (Unix timestamp) to consider.
        """
        if G is None:
            G = self._overlap_graph

        G.remove_edges_from(G.edges())

        # Re-add edges based on current start and end times
        active_query_ids: list[tuple[float, str]] = []
        sorted_queries = sorted(
            G.nodes(data=True),
            key=lambda x: (x[1]["start_time_s"], x[1]["end_time_s"]),
        )

        for current_query_id, data in sorted_queries:

            current_start_time_s = data["start_time_s"]
            current_end_time_s = data["end_time_s"]

            # Remove queries that have ended before the current query starts
            while (
                len(active_query_ids) > 0
                and active_query_ids[0][0] <= current_start_time_s
            ):
                heapq.heappop(active_query_ids)

            # Add edges to all currently active queries
            for active_query_id in active_query_ids:
                G.add_edge(current_query_id, active_query_id[1])

            # Add the current query to the list of active queries
            heapq.heappush(
                active_query_ids, (current_end_time_s, current_query_id)
            )

    def add(
        self,
        query_id: str,
        start_time_s: float,
        end_time_s: float,
        tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx,
    ) -> None:
        """
        Add a query to the timeline.

        Parameters:
            query_id: The ID of the query.
            start_time_s: The start time of the query (Unix timestamp).
            end_time_s: The end time of the query (Unix timestamp).
            tpcds_temp_and_q_idx: The TPC-DS template and query index of the
                query.

        Raises:
            ValueError: If the query ID already exists in the timeline.
        """
        if query_id in self._overlap_graph:
            raise ValueError(
                f"Query ID {query_id} already exists in the timeline."
            )
        self._overlap_graph.add_node(
            query_id,
            query_id=query_id,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            tpcds_temp_and_q_idx=tpcds_temp_and_q_idx,
            featurization=(
                self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                    tpcds_temp_and_q_idx
                )
            ),
        )
        insert_idx = bisect.bisect_left(
            self._ordered_start_times_s, (start_time_s, query_id)
        )
        self._ordered_start_times_s.insert(insert_idx, (start_time_s, query_id))

        # Add edges to all overlapping queries
        for i in range(insert_idx + 1, len(self._ordered_start_times_s)):
            other_start_time_s, other_query_id = self._ordered_start_times_s[i]

            if other_start_time_s >= end_time_s:
                break

            self._overlap_graph.add_edge(query_id, other_query_id)
        for i in range(insert_idx - 1, -1, -1):
            other_start_time_s, other_query_id = self._ordered_start_times_s[i]

            other_end_time_s = self._overlap_graph.nodes[other_query_id][
                "end_time_s"
            ]
            if other_end_time_s <= start_time_s:
                break

            self._overlap_graph.add_edge(query_id, other_query_id)

    def remove(self, query_id: str) -> None:
        """
        Remove a query from the timeline.

        Parameters:
            query_id: The ID of the query to remove.

        Raises:
            ValueError: If the query ID does not exist in the timeline.
        """
        if query_id not in self._overlap_graph:
            raise ValueError(f"Query ID {query_id} does not exist in timeline.")
        self._overlap_graph.remove_node(query_id)
        # Remove from ordered start times
        remove_idx = bisect.bisect_left(
            self._ordered_start_times_s, (0.0, query_id)
        )
        while remove_idx < len(self._ordered_start_times_s):
            if self._ordered_start_times_s[remove_idx][1] == query_id:
                break
            remove_idx += 1
        if remove_idx < len(self._ordered_start_times_s):
            self._ordered_start_times_s.pop(remove_idx)

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
