"""Tests for QueryReservoir."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoslo.tuner.reservoir import QueryReservoir


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_workload_df(
    n: int = 50,
    base_time: datetime | None = None,
    schema: str = "ext_tpcds1000",
    n_templates: int = 5,
) -> pd.DataFrame:
    """Create a minimal workload DataFrame spanning a few hours."""
    if base_time is None:
        # Monday 2024-06-03 09:00 UTC
        base_time = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)

    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        offset_s = rng.uniform(0, 4 * 3600)  # spread over 4 hours
        t = base_time + timedelta(seconds=offset_s)
        template_id = rng.integers(1, n_templates + 1)
        qtid = f"{schema}#{template_id}#001"
        rows.append(
            {
                "query_id": f"q_{i:04d}",
                "abs_start_time": pd.Timestamp(t),
                "query_text_id": qtid,
                "repetition_id": f"rep_{template_id}",
            }
        )
    return pd.DataFrame(rows)


class _FakeWorkload:
    """Minimal workload-like object for reservoir tests."""

    def __init__(self, df: pd.DataFrame, name: str = "test_wl"):
        self._df = df
        self._workload_name = name

    @property
    def df(self) -> pd.DataFrame:
        return self._df


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestQueryReservoirBuild:
    def test_basic_build(self):
        df = _make_workload_df(n=30)
        wl = _FakeWorkload(df)
        reservoir = QueryReservoir.build([wl], schema_name="ext_tpcds1000")

        assert len(reservoir.df) == 30
        assert set(reservoir.df.columns) >= set(QueryReservoir.COLUMNS)
        assert reservoir.meta["schema_name"] == "ext_tpcds1000"
        assert reservoir.meta["num_workloads"] == 1
        assert reservoir.meta["num_arrivals"] == 30

    def test_multiple_workloads(self):
        df1 = _make_workload_df(n=20, base_time=datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc))
        df2 = _make_workload_df(n=15, base_time=datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc))
        wl1 = _FakeWorkload(df1, "wl1")
        wl2 = _FakeWorkload(df2, "wl2")

        reservoir = QueryReservoir.build([wl1, wl2], schema_name="ext_tpcds1000")

        assert len(reservoir.df) == 35
        assert reservoir.meta["num_workloads"] == 2

    def test_day_of_week_and_hour_extracted(self):
        # Monday 09:00 UTC
        base = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)
        df = pd.DataFrame(
            [
                {
                    "query_id": "q0",
                    "abs_start_time": pd.Timestamp(base),
                    "query_text_id": "s#1#001",
                    "repetition_id": "r1",
                }
            ]
        )
        wl = _FakeWorkload(df)
        r = QueryReservoir.build([wl], schema_name="s")

        assert r.df.iloc[0]["day_of_week"] == 0  # Monday
        assert r.df.iloc[0]["hour"] == 9
        assert 0.0 <= r.df.iloc[0]["timestamp_within_hour"] < 1.0

    def test_use_repetition_id_false(self):
        df = _make_workload_df(n=5)
        wl = _FakeWorkload(df)
        r = QueryReservoir.build([wl], schema_name="s", use_repetition_id=False)

        # When use_repetition_id=False, repetition_id should equal query_text_id.
        for _, row in r.df.iterrows():
            assert row["repetition_id"] == row["query_text_id"]

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"day_of_week": [0], "hour": [9]})
        with pytest.raises(ValueError, match="missing columns"):
            QueryReservoir(bad_df, {})


class TestQueryReservoirIO:
    def test_save_load_round_trip(self, tmp_path: Path):
        df = _make_workload_df(n=20)
        wl = _FakeWorkload(df)
        original = QueryReservoir.build([wl], schema_name="ext_tpcds1000")

        original.save(tmp_path / "reservoir")
        loaded = QueryReservoir.load(tmp_path / "reservoir")

        assert len(loaded.df) == len(original.df)
        assert loaded.meta["schema_name"] == "ext_tpcds1000"
        pd.testing.assert_frame_equal(
            loaded.df.reset_index(drop=True),
            original.df.reset_index(drop=True),
        )

    def test_save_creates_files(self, tmp_path: Path):
        df = _make_workload_df(n=5)
        wl = _FakeWorkload(df)
        r = QueryReservoir.build([wl], schema_name="s")

        pq_path, meta_path = r.save(tmp_path / "out")
        assert pq_path.exists()
        assert meta_path.exists()
        assert pq_path.name == "reservoir.parquet"
        assert meta_path.name == "reservoir_meta.yml"


class TestQueryReservoirQueries:
    def test_query_rate_per_hour(self):
        # Create a reservoir with exactly 10 rows in (Monday, 9).
        base = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(10):
            t = base + timedelta(minutes=i)
            rows.append(
                {
                    "query_id": f"q{i}",
                    "abs_start_time": pd.Timestamp(t),
                    "query_text_id": "s#1#001",
                    "repetition_id": "r1",
                }
            )
        df = pd.DataFrame(rows)
        wl = _FakeWorkload(df)
        r = QueryReservoir.build([wl], schema_name="s")

        assert r.query_rate_per_hour(0, 9) == 10.0  # 10 rows / 1 workload
        assert r.query_rate_per_hour(0, 10) == 0.0  # no rows for hour 10

    def test_unique_query_text_ids(self):
        base = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)
        rows = [
            {"query_id": "q0", "abs_start_time": pd.Timestamp(base),
             "query_text_id": "s#1#001", "repetition_id": "r1"},
            {"query_id": "q1", "abs_start_time": pd.Timestamp(base + timedelta(minutes=1)),
             "query_text_id": "s#2#001", "repetition_id": "r2"},
            {"query_id": "q2", "abs_start_time": pd.Timestamp(base + timedelta(minutes=2)),
             "query_text_id": "s#1#001", "repetition_id": "r1"},
        ]
        df = pd.DataFrame(rows)
        wl = _FakeWorkload(df)
        r = QueryReservoir.build([wl], schema_name="s")

        ids = r.unique_query_text_ids(0, 9)
        assert ids == ["s#1#001", "s#2#001"]

    def test_bin_df(self):
        df = _make_workload_df(n=50)
        wl = _FakeWorkload(df)
        r = QueryReservoir.build([wl], schema_name="s")

        bin_9 = r.bin_df(0, 9)
        assert all(bin_9["day_of_week"] == 0)
        assert all(bin_9["hour"] == 9)

    def test_summary(self):
        df = _make_workload_df(n=50)
        wl = _FakeWorkload(df)
        r = QueryReservoir.build([wl], schema_name="s")

        summary = r.summary()
        assert "day_of_week" in summary.columns
        assert "hour" in summary.columns
        assert "count" in summary.columns
        assert summary["count"].sum() == 50
