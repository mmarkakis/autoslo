from datetime import datetime, timedelta
from typing import Any, Optional

from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_execution.trace import Trace


class Day:
    """
    A day consisting of multiple chunk specifications.
    """

    def __init__(self, chunks: list[Chunk]):
        """
        Initialize a new day with a list of chunks.

        Parameters:
            chunks: A list of Chunks.
        """
        self._chunks = chunks
        self._day_id = "_".join([f"H{chunk.H}T{chunk.T}" for chunk in chunks])

    @property
    def day_id(self) -> str:
        """Get the unique identifier for the day."""
        return self._day_id

    @property
    def chunks(self) -> list[Chunk]:
        """Get the list of chunks in the day."""
        return self._chunks

    def colors(self) -> list[str]:
        """Get a list of colors for all chunks in the day."""
        return [chunk.color() for chunk in self.chunks]

    def shapes(self) -> list[str]:
        """Get a list of shapes for all chunks in the day."""
        return [chunk.shape() for chunk in self.chunks]

    def to_dict(self) -> dict[str, Any]:
        """Get the day representation as a dictionary."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Day":
        """Create a Day instance from a dictionary representation."""
        chunks = [Chunk.from_dict(chunk_data) for chunk_data in data["chunks"]]
        return Day(chunks=chunks)

    def get_most_recent_trace_on(
        self,
        blueprint_name: str,
        query_router_name: str,
        normalize_start_to: Optional[datetime] = None,
        inter_chunk_gap: timedelta = timedelta(0),
    ) -> Trace:
        """
        Get the synthesized trace for the entire day on the specified blueprint
        and query router. The synthesized trace is formed by concatenating the
        most recent traces of all chunks in the day on the specified blueprint
        and query router.
        All the timestamps of the non-first chunks are shifted, so that their
        earliest timestamp is equal to the latest timestamp of the previous
        chunk plus the specified gap. Optionally, the entire trace can be
        shifted so that the earliest timestamp (i.e., the earliest timestamp of
        the first chunk) is equal to `normalize_start_to`.

        Parameters:
            blueprint_name: The name of the blueprint.
            query_router_name: The name of the query router.
            normalize_start_to: A datetime object to which the earliest
                timestamp will be normalized.
            inter_chunk_gap: A timedelta object representing the gap to insert
                between consecutive chunks.

        Returns:
            A pandas DataFrame representing the synthesized trace for the day.

        Raises:
            ValueError: If the synthesized trace exceeds 24 hours.
        """

        trace = self.chunks[0].get_most_recent_trace_on(
            blueprint_name=blueprint_name,
            query_router_name=query_router_name,
            normalize_start_to=normalize_start_to,
        )

        for chunk in self.chunks[1:]:
            new_trace_start_time = trace.latest_query_end_time + inter_chunk_gap
            new_trace = chunk.get_most_recent_trace_on(
                blueprint_name=blueprint_name,
                query_router_name=query_router_name,
                normalize_start_to=new_trace_start_time,
            )
            trace.append(new_trace)

        # Check that the trace doesn't actually last more than 24 hours.
        overall_earliest = trace.earliest_query_start_time
        overall_latest = trace.latest_query_end_time
        overall_duration = overall_latest - overall_earliest
        if overall_duration > timedelta(hours=24):
            raise ValueError(
                f"Synthesized trace exceeds 24 hours: {overall_duration}."
            )

        return trace

    def get_available_blueprints_and_query_routers(
        self,
    ) -> dict[str, set[str]]:
        """
        Determine the blueprints and query routers available for this day. These
        are determined by the intersection of the available blueprints and query
        routers for each chunk in the day.

        Returns:
            A dictionary with blueprint names as keys and sets of query router
            names as values.
        """

        result = self.chunks[0].get_available_blueprints_and_query_routers()

        for chunk in self.chunks[1:]:
            chunk_dict = chunk.get_available_blueprints_and_query_routers()
            for blueprint, query_routers in chunk_dict.items():
                if blueprint in result:
                    result[blueprint] = result[blueprint].intersection(
                        query_routers
                    )

                else:
                    del result[blueprint]

        return result
