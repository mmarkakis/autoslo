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
            endpoint_name: The name of the endpoint.
            normalize_start_to: A datetime object to which the earliest
                timestamp will be normalized.
            inter_chunk_gap: A timedelta object representing the gap to insert
                between consecutive chunks.

        Returns:
            A pandas DataFrame representing the synthesized trace for the day.

        Raises:
            ValueError: If the synthesized trace exceeds 24 hours.
        """

        l = [
            self.chunks[0].get_most_recent_trace_on(
                blueprint_name=blueprint_name,
                query_router_name=query_router_name,
                normalize_start_to=normalize_start_to,
            )
        ]
        for chunk in self.chunks[1:]:
            prev_latest_timestamp = l[-1].trace_df["end_time"].max()
            chunk_trace = chunk.get_most_recent_trace_on(
                blueprint_name=blueprint_name,
                query_router_name=query_router_name,
                normalize_start_to=prev_latest_timestamp + inter_chunk_gap,
            )
            l.append(chunk_trace)

        # Check that the trace doesn't actually last more than 24 hours.
        overall_earliest = l[0].trace_df["start_time"].min()
        overall_latest = l[-1].trace_df["end_time"].max()
        overall_duration = overall_latest - overall_earliest
        if overall_duration > timedelta(hours=24):
            raise ValueError(
                f"Synthesized trace exceeds 24 hours: {overall_duration}."
            )

        # Concatenate the synthesized trace.
        synthesized_trace = Trace.concat(l)

        return synthesized_trace
