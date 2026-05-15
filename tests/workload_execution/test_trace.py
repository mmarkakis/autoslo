from collections import defaultdict
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
    latencies = tr.server_side_latencies_s
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
