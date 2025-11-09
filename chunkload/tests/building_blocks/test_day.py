from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytest

from chunkload.building_blocks.chunk import Chunk
from chunkload.building_blocks.day import Day


class FakeChunk:
    """
    Minimal fake chunk used for Day tests. Implements the methods and
    attributes Day relies on so tests stay self-contained.
    """

    def __init__(self, H: int, T: int, base_start: datetime, minutes_offsets):
        self.H = H
        self.T = T
        # Build a tiny base DataFrame with start_time and end_time.
        starts = [
            base_start + timedelta(minutes=off) for off in minutes_offsets
        ]
        ends = [s + timedelta(minutes=1) for s in starts]
        self._base_df = pd.DataFrame({"start_time": starts, "end_time": ends})

    def color(self) -> str:
        return f"color_{self.T}"

    def shape(self) -> str:
        return f"shape_{self.H}"

    def to_dict(self) -> dict:
        # Enough fields for Day.from_dict via Chunk.from_dict compatibility.
        return {
            "H": self.H,
            "T": self.T,
            "schema": "tpcds",
            "chunk_duration_s": 3600,
            "random_seed": 42,
            "num_templates": 1,
            "num_queries_per_template": 1,
            "stddev_interarrival_s": None,
        }

    def get_trace_on(
        self,
        endpoint_name: str,
        normalize_start_to: Optional[datetime] = None,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self._base_df.copy()
        if normalize_start_to is not None:
            earliest = df["start_time"].min()
            shift = normalize_start_to - earliest
            df["start_time"] = df["start_time"] + shift
            df["end_time"] = df["end_time"] + shift
        return df


def test_to_from_dict_roundtrip():
    """
    Verify that Day.to_dict and Day.from_dict round-trip correctly and that
    the contained chunks preserve H and T parameters.
    """
    c1 = Chunk(H=10, T=30)
    c2 = Chunk(H=25, T=60)
    day = Day(chunks=[c1, c2])
    d = day.to_dict()
    day2 = Day.from_dict(d)
    assert len(day2.chunks) == 2
    assert day2.chunks[0].H == c1.H
    assert day2.chunks[0].T == c1.T
    assert day2.chunks[1].H == c2.H
    assert day2.chunks[1].T == c2.T

def test_colors_and_shapes_delegate_to_chunks():
    """
    Ensure Day.colors() and Day.shapes() return values delegated from each
    chunk's color() and shape() methods.
    """
    f1 = FakeChunk(
        H=1, T=10, base_start=datetime(2020, 1, 1), minutes_offsets=[0]
    )
    f2 = FakeChunk(
        H=2, T=20, base_start=datetime(2020, 1, 1), minutes_offsets=[0]
    )
    day = Day(chunks=[f1, f2])
    assert day.colors() == [f1.color(), f2.color()]
    assert day.shapes() == [f1.shape(), f2.shape()]


def test_day_id_formatting():
    """
    Confirm Day.day_id uses the expected "H{H}T{T}" components joined by
    underscores for multiple chunks.
    """
    c1 = FakeChunk(
        H=5, T=7, base_start=datetime(2020, 1, 1), minutes_offsets=[0]
    )
    c2 = FakeChunk(
        H=50, T=120, base_start=datetime(2020, 1, 1), minutes_offsets=[0]
    )
    day = Day(chunks=[c1, c2])
    did = day.day_id
    assert "H5T7" in did
    assert "H50T120" in did
    assert "_" in did


def test_get_trace_on_concatenates_and_respects_gap():
    """
    Test that Day.get_trace_on concatenates chunk traces in order and
    that the gap between chunks equals the requested inter_chunk_gap.
    """
    # First chunk has times anchored at base; will be shifted to normalize_start.
    base0 = datetime(2021, 1, 1, 8, 0, 0)
    f1 = FakeChunk(H=1, T=10, base_start=base0, minutes_offsets=[0, 10])
    num_f1_queries = len(f1.get_trace_on(endpoint_name="ep"))
    # Second chunk base starts at the same anchor but will be shifted by Day.
    f2 = FakeChunk(H=2, T=20, base_start=base0, minutes_offsets=[0, 5])
    day = Day(chunks=[f1, f2])

    normalize_start = datetime(2021, 1, 2, 9, 0, 0)
    gap = timedelta(minutes=5)
    synthesized = day.get_trace_on(
        endpoint_name="ep",
        normalize_start_to=normalize_start,
        inter_chunk_gap=gap,
        save_path=None,
    )
    # Earliest start equals the requested normalization.
    assert synthesized["start_time"].min() == normalize_start
    # Latest end of first chunk:
    first_chunk_end = synthesized.loc[num_f1_queries - 1, "end_time"]
    # Earliest start of second chunk:
    second_start = synthesized.loc[num_f1_queries, "start_time"]
    # The gap between first's latest end and second's earliest start equals gap.
    assert second_start - first_chunk_end == gap


def test_get_trace_on_raises_if_exceeds_24_hours():
    """
    Ensure Day.get_trace_on raises ValueError when the concatenated traces
    would span more than 24 hours.
    """
    # Create two chunks each lasting 13 hours so combined > 24h.
    base = datetime(2020, 1, 1, 0, 0, 0)
    # Build fake chunks with long durations by offsetting start times widely.
    long1 = FakeChunk(H=1, T=10, base_start=base, minutes_offsets=[0, 60 * 13])
    long2 = FakeChunk(H=2, T=20, base_start=base, minutes_offsets=[0, 60 * 13])
    day = Day(chunks=[long1, long2])
    with pytest.raises(ValueError):
        day.get_trace_on(endpoint_name="ep", normalize_start_to=None)
