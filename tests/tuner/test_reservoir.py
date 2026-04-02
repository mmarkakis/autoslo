"""Tests for QueryReservoir."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoslo.tuner.reservoir import QueryReservoir
from autoslo.workload_definition.workload import Workload


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_workload_df(
    n: int = 50,
    base_time: pd.Timestamp | None = None,
    schema: str = "ext_tpcds1000",
    n_templates: int = 5,
) -> pd.DataFrame:
    """Create a minimal workload DataFrame spanning a few hours."""
    if base_time is None:
        # Monday 2024-06-03 09:00 UTC
        base_time = pd.Timestamp("2024-06-03 09:00:00", tz="UTC")

    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        offset_s = rng.uniform(0, 4 * 3600)  # spread over 4 hours
        t = base_time + pd.to_timedelta(offset_s, unit="s")
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
    df = pd.DataFrame(rows)
    df = df.sort_values("abs_start_time").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestQueryReservoirBuild:

    def test_build_based_on_df(self):
        df = _make_workload_df(n=30)
        reservoir = QueryReservoir(df=df)
        assert reservoir.count_df["count"].sum() == 30

    def test_build_based_on_workload(self):
        df = _make_workload_df(n=30)
        wl = Workload("test_wl", "ext_tpcds1000", df)
        reservoir = QueryReservoir(workload=wl)
        assert reservoir.count_df["count"].sum() == 30

    def test_build_with_count_df(self):
        df = _make_workload_df(n=30)
        df["date"] = df["abs_start_time"].dt.date
        df["hour"] = df["abs_start_time"].dt.hour
        count_df = (
            df.groupby(["date", "hour", "query_text_id"])
            .size()
            .reset_index(name="count")
        )
        reservoir = QueryReservoir(count_df=count_df)
        assert reservoir.count_df["count"].sum() == 30

    def test_none_given_raises(self):
        with pytest.raises(ValueError):
            QueryReservoir()

    def test_empty_df_raises(self):
        empty_df = pd.DataFrame(
            columns=[
                "query_id",
                "abs_start_time",
                "query_text_id",
                "repetition_id",
            ]
        )
        with pytest.raises(ValueError):
            QueryReservoir(df=empty_df)

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"abs_start_time": [pd.Timestamp.now()]})
        with pytest.raises(ValueError):
            QueryReservoir(df=bad_df)

    def test_min_date_extracted(self):
        base_date = pd.Timestamp(2024, 6, 3, 9, 0, 0)
        df = pd.DataFrame(
            [
                {
                    "query_id": "q0",
                    "abs_start_time": pd.Timestamp(base_date),
                    "query_text_id": "s#1#001",
                    "repetition_id": "r1",
                },
                {
                    "query_id": "q1",
                    "abs_start_time": pd.Timestamp(
                        base_date + pd.Timedelta(days=1)
                    ),
                    "query_text_id": "s#1#001",
                    "repetition_id": "r1",
                },
            ]
        )
        r = QueryReservoir(df=df)

        assert r.min_date == base_date.date()


class TestQueryReservoirIO:
    def test_save_load_round_trip(self, tmp_path: Path):
        df = _make_workload_df(n=20)
        original = QueryReservoir(df=df)

        original.save(tmp_path / "reservoir")
        loaded = QueryReservoir.load(tmp_path / "reservoir")

        assert loaded.count_df.equals(original.count_df)
        assert loaded.min_date == original.min_date

    def test_save_creates_files(self, tmp_path: Path):
        df = _make_workload_df(n=20)
        reservoir = QueryReservoir(df=df)

        reservoir.save(tmp_path / "reservoir")
        count_df_path = tmp_path / "reservoir" / "reservoir.parquet"
        assert count_df_path.exists()

        read_df = pd.read_parquet(count_df_path)
        assert read_df.equals(reservoir.count_df)


class TestQueryReservoirCounts:

    def test_bin_df_multiple_bins(self):
        df = _make_workload_df(n=50)
        base_date = df["abs_start_time"].dt.date.min()
        r = QueryReservoir(df=df)

        bin_9 = r.bin_df(base_date, 9)
        correct_count_9 = df[df["abs_start_time"].dt.hour == 9].shape[0]
        print(bin_9)
        assert all(bin_9["date"] == r.min_date)
        assert all(bin_9["hour"] == 9)
        assert bin_9["count"].sum() == correct_count_9

        bin_10 = r.bin_df(base_date, 10)
        correct_count_10 = df[df["abs_start_time"].dt.hour == 10].shape[0]
        assert all(bin_10["date"] == r.min_date)
        assert all(bin_10["hour"] == 10)
        assert bin_10["count"].sum() == correct_count_10

    def test_bin_df_no_data(self):
        df = _make_workload_df(n=50)
        r = QueryReservoir(df=df)
        base_date = df["abs_start_time"].dt.date.min()

        bin_3 = r.bin_df(base_date, 3)
        assert bin_3.empty

    def test_bin_df_invalid_hour(self):
        df = _make_workload_df(n=50)
        base_date = df["abs_start_time"].dt.date.min()
        r = QueryReservoir(df=df)

        with pytest.raises(ValueError):
            r.bin_df(base_date, 24)

    def test_per_query_counts(self):
        base_date = pd.Timestamp(2024, 6, 3, 9, 0, 0, tz="UTC")
        rows = [
            {
                "query_id": "q0",
                "abs_start_time": pd.Timestamp(base_date),
                "query_text_id": "s#1#001",
                "repetition_id": "r1",
            },
            {
                "query_id": "q1",
                "abs_start_time": pd.Timestamp(
                    base_date + pd.to_timedelta(1, unit="m")
                ),
                "query_text_id": "s#2#001",
                "repetition_id": "r2",
            },
            {
                "query_id": "q2",
                "abs_start_time": pd.Timestamp(
                    base_date + pd.to_timedelta(2, unit="m")
                ),
                "query_text_id": "s#1#001",
                "repetition_id": "r1",
            },
        ]
        df = pd.DataFrame(rows)
        r = QueryReservoir(df=df)

        bin_9 = r.bin_df(base_date.date(), 9)
        print(bin_9)
        print(r.count_df)
        assert bin_9.shape[0] == 2  # two unique query_text_ids
        assert set(bin_9["query_text_id"]) == {"s#1#001", "s#2#001"}
        assert bin_9[bin_9["query_text_id"] == "s#1#001"]["count"].iloc[0] == 2
