import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

import chunkload.utils.paths as pu
from chunkload.building_blocks.chunk import Chunk
from chunkload.building_blocks.composite import Composite
from chunkload.building_blocks.day import Day


def _make_trace_df(
    intervals_seconds: List[Tuple[int, int]],
    base: Optional[datetime] = None,
    elapsed_multiplier: float = 1.0,
    extra_columns: Optional[Dict[str, list]] = None,
) -> pd.DataFrame:
    """
    Build a minimal trace DataFrame for testing.

    intervals_seconds: list of (start_offset_s, duration_s) tuples.
    base: optional base datetime (defaults to 2025-09-01).
    elapsed_multiplier: multiplier applied to the measured elapsed time.
    extra_columns: optional dict of additional columns (each list must match
    the number of intervals).
    """
    if base is None:
        base = datetime(2025, 9, 1, 0, 0, 0)
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    for off_s, dur_s in intervals_seconds:
        s = pd.Timestamp(base + timedelta(seconds=off_s))
        e = s + timedelta(seconds=dur_s)
        starts.append(s)
        ends.append(e)
    elapsed_us = [
        int((e - s).total_seconds() * 1_000_000 * elapsed_multiplier)
        for s, e in zip(starts, ends)
    ]
    data = {
        "start_time": starts,
        "end_time": ends,
        "elapsed_time": elapsed_us,
    }
    if extra_columns:
        for k, v in extra_columns.items():
            data[k] = v
    return pd.DataFrame(data)


class FakeDay:
    """
    Minimal fake Day used for Composite tests. Implements only the pieces of
    the Day API that Composite exercises in the tests.
    """

    def __init__(
        self, name: str, base_start: datetime, intervals: List[Tuple[int, int]]
    ):
        self.name = name
        self.base_start = base_start
        self.intervals = intervals
        # For plot_definition compatibility: keep an empty chunks attribute.
        self.chunks: List = []

    def to_dict(self) -> dict:
        # Minimal serializable representation used by Composite.to_dict()
        return {"chunks": []}

    @staticmethod
    def from_dict(data: dict) -> "FakeDay":
        # Not used by Composite.from_dict in tests; present for completeness.
        return FakeDay("from_dict", datetime(2025, 9, 1), [(0, 1)])

    def get_trace_on(
        self,
        endpoint_name: str,
        normalize_start_to: Optional[datetime] = None,
        inter_chunk_gap: timedelta = timedelta(0),
        save_path: Optional[str] = None,
        force_recompose: bool = False,
    ) -> pd.DataFrame:
        """
        Return a trace DataFrame whose earliest start equals normalize_start_to
        if provided; otherwise anchored at self.base_start.
        """
        df = _make_trace_df(
            self.intervals,
            base=self.base_start,
            extra_columns={
                "response_time_ms": [
                    float((end - start).total_seconds() * 1000)
                    for start, end in zip(
                        [
                            self.base_start + timedelta(seconds=off)
                            for off, _ in self.intervals
                        ],
                        [
                            self.base_start + timedelta(seconds=off + dur)
                            for off, dur in self.intervals
                        ],
                    )
                ]
            },
        )
        if normalize_start_to is not None:
            earliest = df["start_time"].min()
            shift = pd.Timestamp(normalize_start_to) - earliest
            df["start_time"] = df["start_time"] + shift
            df["end_time"] = df["end_time"] + shift
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # write a lightweight placeholder to avoid parquet engine issues
            with open(save_path, "wb") as fh:
                fh.write(b"PARQUET_PLACEHOLDER")
        return df


def test_to_from_dict_roundtrip():
    """
    Check Composite.to_dict and Composite.from_dict round-trip when using
    real Day and Chunk objects.
    """
    c1 = Chunk(H=10, T=30)
    c2 = Chunk(H=25, T=60)
    day1 = Day(chunks=[c1])
    day2 = Day(chunks=[c2])
    comp = Composite(name="cmp", days=[day1, day2], monday_index=1)
    d = comp.to_dict()
    comp2 = Composite.from_dict(d)
    assert comp2.name == comp.name
    assert comp2.monday_index == comp.monday_index
    assert len(comp2.days) == len(comp.days)
    for original_day, roundtrip_day in zip(comp.days, comp2.days):
        assert len(original_day.chunks) == len(roundtrip_day.chunks)
        for original_chunk, roundtrip_chunk in zip(
            original_day.chunks, roundtrip_day.chunks
        ):
            assert original_chunk.H == roundtrip_chunk.H
            assert original_chunk.T == roundtrip_chunk.T
            assert original_chunk.schema == roundtrip_chunk.schema
            assert (
                original_chunk.chunk_duration_s
                == roundtrip_chunk.chunk_duration_s
            )


def test_day_initials_various_monday_index():
    """
    Ensure day_initials returns correct day initials sequence starting from
    the configured monday_index.
    """
    days = [Day(chunks=[Chunk(H=0, T=10)]) for _ in range(5)]
    comp = Composite(name="x", days=days, monday_index=2)
    initials = comp.day_initials()
    # Starting at index 2 (W), for 5 days -> W, T, F, S, S
    assert initials == ["W", "T", "F", "S", "S"]


def test_get_trace_on_concatenates_days_with_24h_spacing():
    """
    Verify that Composite.get_trace_on concatenates day traces such that the
    earliest timestamp of each subsequent day is exactly 24 hours after the
    previous day's earliest timestamp.
    """
    base_a = datetime(2025, 9, 1, 8, 0, 0)
    base_b = datetime(2025, 9, 1, 9, 0, 0)
    day_a = FakeDay("A", base_start=base_a, intervals=[(0, 60)])
    day_b = FakeDay("B", base_start=base_b, intervals=[(0, 30)])
    comp = Composite(name="cmp", days=[day_a, day_b], monday_index=0)

    normalize_start = datetime(2025, 9, 10, 6, 0, 0)
    synthesized = comp.get_trace_on(
        endpoint_name="ep", normalize_start_to=normalize_start
    )
    # earliest start equals requested normalize_start_to
    assert synthesized["start_time"].min() == pd.Timestamp(normalize_start)
    # find earliest of first day and earliest of second day in the concatenated DF
    first_day_earliest = synthesized.iloc[0]["start_time"]
    # locate the earliest start that is strictly greater than first_day_earliest
    later_starts = synthesized[synthesized["start_time"] > first_day_earliest][
        "start_time"
    ]
    assert not later_starts.empty
    second_day_earliest = later_starts.min()
    assert (second_day_earliest - first_day_earliest) == timedelta(days=1)


def test_save_writes_definition_and_stats(tmp_path, monkeypatch):
    """
    Test Composite.save writes definition.yml and a day_tail_stats parquet.
    Patch DataFrame.to_parquet to avoid external parquet engine requirements.
    """
    # patch DATA_PATH
    monkeypatch.setattr(pu, "DATA_PATH", str(tmp_path))

    # Create two FakeDays that return traces with elapsed_time column
    d1 = FakeDay(
        "d1", base_start=datetime(2025, 9, 1, 0, 0, 0), intervals=[(0, 10)]
    )
    d2 = FakeDay(
        "d2", base_start=datetime(2025, 9, 1, 0, 0, 0), intervals=[(0, 20)]
    )
    comp = Composite(name="comp_save_test", days=[d1, d2], monday_index=0)

    # Patch pandas.DataFrame.to_parquet to create a placeholder file
    def fake_to_parquet(self, path, index=False):
        with open(path, "wb") as fh:
            fh.write(b"PARQUET_PLACEHOLDER")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    # Run save() which should create definition.yml and day_tail_stats.parquet
    comp.save()

    out_dir = os.path.join(str(tmp_path), "composite_workloads", comp.name)
    assert os.path.exists(os.path.join(out_dir, "definition.yml"))
    assert os.path.exists(os.path.join(out_dir, "day_tail_stats.parquet"))


def test_ground_truth_smallest_adherent_endpoint_errors_and_success(
    tmp_path, monkeypatch
):
    """
    Test error cases for ground_truth_smallest_adherent_endpoint when stats
    file is missing or when percentile is not present, and test success case
    by patching pd.read_parquet to return prepared stats.
    """
    # Ensure missing file raises
    monkeypatch.setattr(pu, "get_data_path", lambda: str(tmp_path))
    with pytest.raises(ValueError):
        Composite.ground_truth_smallest_adherent_endpoint(
            "nonexistent", tail_slo_s=1.0
        )

    # Prepare a stats DataFrame missing the requested percentile
    stats_df90 = pd.DataFrame(
        [
            {
                "composite_name": "c",
                "day_idx": 0,
                "endpoint_name": "4",
                "endpoint_rpu": 4,
                "percentile": 90,
                "tail_s": 0.5,
            }
        ]
    )

    # Patch pd.read_parquet to return our dataframe when the file exists
    def fake_read_parquet(path, *args, **kwargs):
        return stats_df90

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    # Create directory so the initial file-existence check passes
    workload_dir = os.path.join(str(tmp_path), "composite_workloads", "c")
    os.makedirs(workload_dir, exist_ok=True)
    # Write a dummy file so os.path.exists(stats_file) is True
    open(os.path.join(workload_dir, "day_tail_stats.parquet"), "wb").close()

    # Asking for 95 percentile should raise
    with pytest.raises(ValueError):
        Composite.ground_truth_smallest_adherent_endpoint(
            "c", tail_slo_s=1.0, tail_percentile=95.0
        )

    # Now provide stats containing the requested percentile and test return
    # value
    stats_df95 = pd.DataFrame(
        [
            {
                "composite_name": "c",
                "day_idx": 0,
                "endpoint_name": "4",
                "endpoint_rpu": 4,
                "percentile": 95.0,
                "tail_s": 1.5,
            },
            {
                "composite_name": "c",
                "day_idx": 0,
                "endpoint_name": "8",
                "endpoint_rpu": 8,
                "percentile": 95.0,
                "tail_s": 0.99,
            },
            {
                "composite_name": "c",
                "day_idx": 0,
                "endpoint_name": "16",
                "endpoint_rpu": 16,
                "percentile": 95.0,
                "tail_s": 0.5,
            },
            {
                "composite_name": "c",
                "day_idx": 1,
                "endpoint_name": "4",
                "endpoint_rpu": 4,
                "percentile": 95.0,
                "tail_s": 2.0,
            },
        ]
    )

    def fake_read_parquet2(path, *args, **kwargs):
        return stats_df95

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet2)

    # The smallest RPU meeting tail_slo_s=1.0 for day 0 is 8; for day 1 none
    result = Composite.ground_truth_smallest_adherent_endpoint(
        "c", tail_slo_s=1.0, tail_percentile=95.0
    )
    assert isinstance(result, list)
    assert result[0] == 8
    assert result[1] is None
