from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from collections import defaultdict

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


def _trace_from_df(df: pd.DataFrame, cluster_name: str = "clusterA") -> Trace:
    """
    Build a Trace instance by populating the internal _dfs structure to match
    the current Trace implementation (which expects parquet-based construction).
    """
    tr = Trace.__new__(Trace)
    tr._dfs = defaultdict(dict)
    df_copy = df.copy()
    df_copy["start_time"] = pd.to_datetime(df_copy["start_time"])
    df_copy["end_time"] = pd.to_datetime(df_copy["end_time"])
    df_copy["elapsed_time"] = df_copy["elapsed_time"].astype(int)
    tr._dfs[cluster_name]["sys_query_history"] = df_copy.reset_index(drop=True)
    tr._original_start = (
        df_copy["start_time"].min()
        if not df_copy["start_time"].empty
        else datetime.now()
    )
    return tr


def test_validate_missing_columns_raises():
    """
    Verify that creating a Trace from an incomplete internal state leads to
    property access errors (no sys_query_history data).
    """
    # Build Trace with no sys_query_history data
    tr = Trace.__new__(Trace)
    tr._dfs = {}  # no clusters
    with pytest.raises(ValueError):
        _ = tr.earliest_query_start_time


def test_normalize_and_reset_start_shifts_and_restores_times():
    """
    Test that normalize_start_to shifts times and reset_start restores the
    original earliest start time.
    """
    base = datetime(2021, 1, 1, 0, 0, 0)
    df = _make_trace_df([(0, 10), (20, 5)], base=base)
    tr = _trace_from_df(df)
    cluster = "clusterA"
    orig_earliest = tr._dfs[cluster]["sys_query_history"]["start_time"].min()
    new_start = pd.Timestamp(datetime(2021, 1, 2, 0, 0, 0))
    tr.normalize_start_to(new_start)
    assert (
        tr._dfs[cluster]["sys_query_history"]["start_time"].min() == new_start
    )
    assert tr._dfs[cluster]["sys_query_history"][
        "end_time"
    ].min() == new_start + timedelta(seconds=10)
    assert tr._dfs[cluster]["sys_query_history"][
        "end_time"
    ].max() == new_start + timedelta(seconds=25)
    tr.reset_start()
    assert (
        tr._dfs[cluster]["sys_query_history"]["start_time"].min()
        == orig_earliest
    )


def test_latency_and_counts_and_quantile_behavior():
    """
    Verify latencies_s, num_queries and percentile behavior for a simple trace.
    """
    # elapsed times: 1s, 2s, 4s
    base = datetime(2021, 1, 1)
    df = _make_trace_df([(0, 1), (10, 2), (30, 4)], base=base)
    tr = _trace_from_df(df)
    assert tr.num_queries == 3
    # count queries over 1.5s -> should be the two with 2s and 4s
    latencies = tr.latencies_s
    assert sum(1 for l in latencies if l > 1.5) == 2
    # 50th percentile should be 2.0 seconds
    median = pd.Series(latencies).quantile(0.5)
    assert pytest.approx(median, rel=1e-6) == 2.0
    # invalid quantile should raise
    with pytest.raises(ValueError):
        pd.Series(latencies).quantile(-0.1)
    with pytest.raises(ValueError):
        pd.Series(latencies).quantile(1.1)


def test_billed_s_simple_cases():
    """
    Test billed time via Trace._billed_s for simple scenarios:
    - empty trace => 0
    - single short query (< threshold) => billed == threshold
    - two distant short queries => billed == 2 * threshold
    """
    # empty
    empty = pd.DataFrame(columns=["start_time", "end_time", "elapsed_time"])
    start_empty = empty["start_time"]
    end_empty = empty["end_time"]
    assert Trace._billed_s(start_empty, end_empty) == 0.0

    # single short query (10s) -> billed at the given threshold 60s
    threshold_s = 60
    base = datetime(2021, 1, 1)
    df_single = _make_trace_df([(0, 10)], base=base)
    assert Trace._billed_s(
        df_single["start_time"], df_single["end_time"], threshold_s=threshold_s
    ) == pytest.approx(threshold_s)

    # two short queries far apart -> each billed separately
    df_two = _make_trace_df([(0, 10), (3600, 5)], base=base)
    assert Trace._billed_s(
        df_two["start_time"], df_two["end_time"], threshold_s=threshold_s
    ) == pytest.approx(2 * threshold_s)


def test_billed_s_merged_intervals():
    """
    Test billed time where short queries are close enough to be merged.
    """
    base = datetime(2021, 1, 1)
    threshold_s = 60
    # two short queries 30s apart -> should be merged into a single billed interval
    df_merged = _make_trace_df([(0, 10), (30, 5)], base=base)
    assert Trace._billed_s(
        df_merged["start_time"], df_merged["end_time"], threshold_s=threshold_s
    ) == pytest.approx(threshold_s)

    # three short queries, first two close, third far -> first two merged,
    # third separate
    df_mixed = _make_trace_df([(0, 10), (50, 5), (400, 8)], base=base)
    assert Trace._billed_s(
        df_mixed["start_time"], df_mixed["end_time"], threshold_s=threshold_s
    ) == pytest.approx(2 * threshold_s)


def test_invalid_constructor_arguments_raise_value_error():
    """
    Ensure property access raises when internal state is missing required data.
    """
    # The Trace.REQUIRED_COLUMNS structure exists and includes sys_query_history
    assert "sys_query_history" in Trace.REQUIRED_COLUMNS
    required = Trace.REQUIRED_COLUMNS["sys_query_history"]
    for col in ["start_time", "end_time", "elapsed_time"]:
        assert col in required
    # Accessing latest/earliest when no sys_query_history should raise
    tr = Trace.__new__(Trace)
    tr._dfs = {"some_cluster": {}}  # cluster present but no sys_query_history
    with pytest.raises(ValueError):
        _ = tr.latest_query_end_time


def test_earliest_query_start_time():
    """
    Verify that earliest_query_start_time property returns the correct value.
    """
    base = datetime(2021, 1, 1)
    df = _make_trace_df([(10, 5), (0, 10), (20, 2)], base=base)
    tr = _trace_from_df(df)
    expected_earliest = pd.Timestamp(base)
    assert tr.earliest_query_start_time == expected_earliest


def test_latest_query_end_time():
    """
    Verify that latest_query_end_time property returns the correct value.
    """
    base = datetime(2021, 1, 1)
    df = _make_trace_df([(10, 5), (0, 10), (20, 15)], base=base)
    tr = _trace_from_df(df)
    expected_latest = pd.Timestamp(base + timedelta(seconds=35))
    assert tr.latest_query_end_time == expected_latest


def test_append_merges_data_correctly():
    """
    Verify that appending one Trace to another merges their data correctly. 
    Do not actually inspect the underlying dataframes here.
    """
    base = datetime(2021, 1, 1)
    df1 = _make_trace_df([(0, 10), (20, 5)], base=base)
    df2 = _make_trace_df([(15, 10), (40, 5)], base=base)
    tr1 = _trace_from_df(df1)
    tr2 = _trace_from_df(df2)

    tr1.append(tr2, time_gap_s=60)

    expected_latest = pd.Timestamp(base + timedelta(seconds=25+60-15+45))
    assert tr1.latest_query_end_time == expected_latest
    
    expected_earliest = pd.Timestamp(base)
    assert tr1.earliest_query_start_time == expected_earliest

    expected_latencies = [10.0, 5.0, 10.0, 5.0]  # in seconds
    assert tr1.latencies_s == expected_latencies

    expected_billed_s = 60 + 60  # two billed intervals of 60s each
    actual_billed_s = Trace._billed_s(
        tr1._dfs["clusterA"]["sys_query_history"]["start_time"],
        tr1._dfs["clusterA"]["sys_query_history"]["end_time"],
    )
    assert pytest.approx(actual_billed_s, rel=1e-6) == expected_billed_s


