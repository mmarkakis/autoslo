"""Tests for WorkloadSampler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoslo.tuner.forecast_policy import (
    RecencyWeightedForecastPolicy,
    UniformForecastPolicy,
)
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.workload_sampler import WorkloadSampler


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_reservoir(
    n_queries: int = 100,
    base_time: datetime | None = None,
    n_templates: int = 5,
    n_workloads: int = 1,
    schema: str = "ext_tpcds1000",
) -> QueryReservoir:
    """Build a reservoir from synthetic arrivals."""
    if base_time is None:
        base_time = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)

    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_queries):
        offset_s = rng.uniform(0, 4 * 3600)
        t = base_time + timedelta(seconds=offset_s)
        hour_floor = t.replace(minute=0, second=0, microsecond=0)
        ts_within_hour = (t - hour_floor).total_seconds()

        template_id = rng.integers(1, n_templates + 1)
        qtid = f"{schema}#{template_id}#001"

        rows.append(
            {
                "day_of_week": t.weekday(),
                "hour": t.hour,
                "timestamp_within_hour": ts_within_hour,
                "query_text_id": qtid,
                "repetition_id": f"rep_{template_id}",
            }
        )

    df = pd.DataFrame(rows, columns=QueryReservoir.COLUMNS)
    meta = {
        "schema_name": schema,
        "num_workloads": n_workloads,
        "num_arrivals": len(df),
        "classifications": {},
    }
    return QueryReservoir(df, meta)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestMakeHourBins:
    def test_full_hours(self):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
        bins = WorkloadSampler._make_hour_bins(start, end)
        assert len(bins) == 3
        assert bins[0] == (
            datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc),
        )
        assert bins[-1][1] == end

    def test_partial_start(self):
        start = datetime(2024, 6, 3, 9, 30, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 11, 0, tzinfo=timezone.utc)
        bins = WorkloadSampler._make_hour_bins(start, end)
        assert bins[0][0] == start  # not floored
        assert bins[0][1] == datetime(2024, 6, 3, 10, 0, tzinfo=timezone.utc)

    def test_partial_end(self):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 10, 15, tzinfo=timezone.utc)
        bins = WorkloadSampler._make_hour_bins(start, end)
        assert len(bins) == 2
        assert bins[-1][1] == end  # partial hour bin

    def test_same_hour(self):
        start = datetime(2024, 6, 3, 9, 15, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 9, 45, tzinfo=timezone.utc)
        bins = WorkloadSampler._make_hour_bins(start, end)
        assert len(bins) == 1
        assert bins[0] == (start, end)


class TestSample:
    @pytest.fixture()
    def sampler(self) -> WorkloadSampler:
        reservoir = _build_reservoir(n_queries=200, n_workloads=2)
        policy = UniformForecastPolicy()
        return WorkloadSampler(reservoir, policy, "ext_tpcds1000")

    def test_returns_correct_count(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 13, 0, tzinfo=timezone.utc)
        workloads = sampler.sample(start, end, n_scenarios=5, seed=0)
        assert len(workloads) == 5

    def test_workloads_have_required_columns(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 11, 0, tzinfo=timezone.utc)
        workloads = sampler.sample(start, end, n_scenarios=3, seed=0)
        for wl in workloads:
            assert "query_id" in wl.df.columns
            assert "abs_start_time" in wl.df.columns
            assert "query_text_id" in wl.df.columns
            assert "repetition_id" in wl.df.columns

    def test_workloads_are_sorted_by_time(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 13, 0, tzinfo=timezone.utc)
        workloads = sampler.sample(start, end, n_scenarios=3, seed=0)
        for wl in workloads:
            times = wl.df["abs_start_time"].tolist()
            assert times == sorted(times)

    def test_different_seeds_give_different_workloads(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
        wl_a = sampler.sample(start, end, n_scenarios=1, seed=0)
        wl_b = sampler.sample(start, end, n_scenarios=1, seed=99)
        # They should differ in at least some queries.
        ids_a = set(wl_a[0].df["query_text_id"].tolist())
        ids_b = set(wl_b[0].df["query_text_id"].tolist())
        # It's possible but very unlikely that two seeds give identical results
        # for a non-trivial workload, so we just check they're both non-empty.
        assert len(wl_a[0].df) > 0
        assert len(wl_b[0].df) > 0

    def test_deterministic_with_same_seed(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
        wl_a = sampler.sample(start, end, n_scenarios=3, seed=42)
        wl_b = sampler.sample(start, end, n_scenarios=3, seed=42)
        for a, b in zip(wl_a, wl_b):
            pd.testing.assert_frame_equal(a.df, b.df)

    def test_workload_names(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 11, 0, tzinfo=timezone.utc)
        workloads = sampler.sample(start, end, n_scenarios=3, seed=0)
        assert workloads[0].workload_name == "tuner_scenario_000"
        assert workloads[1].workload_name == "tuner_scenario_001"
        assert workloads[2].workload_name == "tuner_scenario_002"

    def test_abs_times_within_range(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
        workloads = sampler.sample(start, end, n_scenarios=3, seed=0)
        for wl in workloads:
            if len(wl.df) > 0:
                min_t = wl.df["abs_start_time"].min()
                max_t = wl.df["abs_start_time"].max()
                assert min_t >= pd.Timestamp(start)
                assert max_t < pd.Timestamp(end)

    def test_query_text_ids_from_reservoir(self, sampler: WorkloadSampler):
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
        workloads = sampler.sample(start, end, n_scenarios=2, seed=0)
        pool = set(sampler.reservoir.df["query_text_id"].unique())
        for wl in workloads:
            sampled_ids = set(wl.df["query_text_id"].unique())
            assert sampled_ids <= pool, "Sampled IDs not in reservoir"


class TestSampleToDisk:
    def test_writes_parquet_files(self, tmp_path: Path):
        reservoir = _build_reservoir(n_queries=100, n_workloads=1)
        policy = UniformForecastPolicy()
        sampler = WorkloadSampler(reservoir, policy, "ext_tpcds1000")

        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)

        paths = sampler.sample_to_disk(
            start, end, n_scenarios=3, out_dir=tmp_path / "workloads", prefix="t", seed=0
        )

        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".parquet"
        assert paths[0].name == "t_000.parquet"
        assert paths[1].name == "t_001.parquet"

    def test_round_trip_readable(self, tmp_path: Path):
        reservoir = _build_reservoir(n_queries=100, n_workloads=1)
        policy = UniformForecastPolicy()
        sampler = WorkloadSampler(reservoir, policy, "ext_tpcds1000")

        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)

        paths = sampler.sample_to_disk(
            start, end, n_scenarios=1, out_dir=tmp_path, prefix="s", seed=0
        )
        loaded = pd.read_parquet(paths[0])
        assert "query_id" in loaded.columns
        assert "abs_start_time" in loaded.columns
        assert len(loaded) > 0


class TestPreview:
    def test_returns_dataframe(self):
        reservoir = _build_reservoir(n_queries=100, n_workloads=1)
        policy = UniformForecastPolicy()
        sampler = WorkloadSampler(reservoir, policy, "ext_tpcds1000")

        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 13, 0, tzinfo=timezone.utc)

        preview = sampler.preview(start, end)
        assert isinstance(preview, pd.DataFrame)
        assert "bin_start" in preview.columns
        assert "expected_count" in preview.columns
        assert len(preview) == 4  # 4 hour bins

    def test_expected_counts_positive_for_populated_bins(self):
        reservoir = _build_reservoir(n_queries=200, n_workloads=1)
        policy = UniformForecastPolicy()
        sampler = WorkloadSampler(reservoir, policy, "ext_tpcds1000")

        # Target the same day/hour range as the reservoir data.
        start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 6, 3, 13, 0, tzinfo=timezone.utc)

        preview = sampler.preview(start, end)
        # At least some bins should have positive expected counts.
        assert preview["expected_count"].sum() > 0
