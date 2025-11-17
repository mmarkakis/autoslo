from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import pytest

from autoslo.workload_execution.trace import Trace


def _make_trace_df(
    intervals_seconds: List[Tuple[int, int]],
    base: Optional[datetime] = None,
    templates: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Helper to build a trace DataFrame.

    Parameters:
        intervals_seconds: list of (start_offset_s, duration_s) tuples.
        base: optional base datetime (defaults to 2021-01-01).
        templates: optional list of template ids per interval.
    """
    if base is None:
        base = datetime(2021, 1, 1, 0, 0, 0)
    starts = []
    ends = []
    for off_s, dur_s in intervals_seconds:
        s = pd.Timestamp(base + timedelta(seconds=off_s))
        e = s + timedelta(seconds=dur_s)
        starts.append(s)
        ends.append(e)
    if templates is None:
        templates = [1] * len(starts)
    elapsed_us = [
        int((e - s).total_seconds() * 1_000_000) for s, e in zip(starts, ends)
    ]
    df = pd.DataFrame(
        {
            "start_time": starts,
            "end_time": ends,
            "elapsed_time": elapsed_us,
            "query_template": templates,
        }
    )
    return df


def test_validate_missing_columns_raises():
    """
    Verify that creating a Trace from a DataFrame missing required columns
    raises a ValueError.
    """
    df = pd.DataFrame({"start_time": [pd.Timestamp(datetime.now())]})
    with pytest.raises(ValueError):
        Trace(df)


def test_normalize_and_reset_start_shifts_and_restores_times():
    """
    Test that normalize_start_to shifts times and reset_start restores the
    original earliest start time.
    """
    base = datetime(2021, 1, 1, 0, 0, 0)
    df = _make_trace_df([(0, 10), (20, 5)], base=base)
    tr = Trace(df)
    orig_earliest = tr._trace_df["start_time"].min()
    new_start = pd.Timestamp(datetime(2021, 1, 2, 0, 0, 0))
    tr.normalize_start_to(new_start)
    assert tr._trace_df["start_time"].min() == new_start
    assert tr._trace_df["end_time"].min() == new_start + timedelta(seconds=10)
    assert tr._trace_df["end_time"].max() == new_start + timedelta(seconds=25)
    tr.reset_start()
    assert tr._trace_df["start_time"].min() == orig_earliest


def test_latency_and_counts_and_quantile_behavior():
    """
    Verify latency_s_at, num_queries and num_queries_with_latency_over
    produce expected results for a simple trace.
    """
    # elapsed times: 1s, 2s, 4s
    base = datetime(2021, 1, 1)
    df = _make_trace_df([(0, 1), (10, 2), (30, 4)], base=base)
    tr = Trace(df)
    assert tr.num_queries() == 3
    # count queries over 1.5s -> should be the two with 2s and 4s
    assert tr.num_queries_with_latency_over(1.5) == 2
    # 50th percentile should be 2000000 us -> 2.0 seconds
    assert pytest.approx(tr.latency_s_at(0.5), rel=1e-6) == 2.0
    # invalid quantile should raise
    with pytest.raises(ValueError):
        tr.latency_s_at(-0.1)
    with pytest.raises(ValueError):
        tr.latency_s_at(1.1)


def test_billed_s_simple_cases():
    """
    Test billed_s for simple scenarios:
    - empty trace => 0
    - single short query (< threshold) => billed == threshold
    - two distant short queries => billed == 2 * threshold
    """
    # empty
    empty = pd.DataFrame(columns=["start_time", "end_time", "elapsed_time"])
    tr_empty = Trace(empty)
    assert tr_empty.billed_s() == 0.0

    # single short query (10s) -> billed at the given threshold 60s
    threshold_s = 60
    base = datetime(2021, 1, 1)
    df_single = _make_trace_df([(0, 10)], base=base)
    tr_single = Trace(df_single)
    assert tr_single.billed_s(threshold_s=threshold_s) == pytest.approx(
        threshold_s
    )

    # two short queries far apart -> each billed separately
    df_two = _make_trace_df([(0, 10), (3600, 5)], base=base)
    tr_two = Trace(df_two)
    assert tr_two.billed_s(threshold_s=threshold_s) == pytest.approx(
        2 * threshold_s
    )


def test_billed_s_merged_intervals():
    """
    Test billed_s for scenarios where short queries are close enough to be
    merged into a single billed interval.
    """
    base = datetime(2021, 1, 1)
    threshold_s = 60
    # two short queries 30s apart -> should be merged into a single billed interval
    df_merged = _make_trace_df([(0, 10), (30, 5)], base=base)
    tr_merged = Trace(df_merged)
    assert tr_merged.billed_s(threshold_s=threshold_s) == pytest.approx(
        threshold_s
    )

    # three short queries, first two close, third far -> first two merged,
    # third separate
    df_mixed = _make_trace_df([(0, 10), (50, 5), (400, 8)], base=base)
    tr_mixed = Trace(df_mixed)
    assert tr_mixed.billed_s(threshold_s=threshold_s) == pytest.approx(
        2 * threshold_s
    )


def test_invalid_constructor_arguments_raise_value_error():
    """
    Ensure that invalid constructor arguments raise ValueError.
    """
    # Invalid DataFrame (missing columns)
    required_columns = Trace.REQUIRED_COLUMNS
    df_correct = pd.DataFrame({col: [] for col in required_columns})
    df_missing = df_correct.drop(columns=Trace.REQUIRED_COLUMNS[0])
    with pytest.raises(ValueError):
        Trace(df_missing)

    # Invalid elapsed_time mismatch
    base = datetime(2021, 1, 1)
    df_mismatch = _make_trace_df([(0, 10)], base=base)
    df_mismatch.loc[0, "elapsed_time"] = int(5 * 1_000_000)
    with pytest.raises(ValueError):
        Trace(df_mismatch)
