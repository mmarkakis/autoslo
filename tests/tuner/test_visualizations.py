"""Tests for tuner visualization helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from autoslo.tuner.forecast_policy import UniformForecastPolicy
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.tuner.workload_sampler import WorkloadSampler

# Try importing matplotlib — skip all tests if not installed.
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")


# ------------------------------------------------------------------
# Fixture: reservoir + sampled workloads
# ------------------------------------------------------------------

def _build_test_data():
    """Build a small reservoir and sampled workloads for visualization tests."""
    base = datetime(2024, 6, 3, 9, 0, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    rows = []
    for i in range(100):
        offset_s = rng.uniform(0, 4 * 3600)
        t = base + timedelta(seconds=offset_s)
        hour_floor = t.replace(minute=0, second=0, microsecond=0)
        ts_within_hour = (t - hour_floor).total_seconds()
        template_id = rng.integers(1, 6)
        qtid = f"ext_tpcds1000#{template_id}#001"
        rows.append({
            "day_of_week": t.weekday(),
            "hour": t.hour,
            "timestamp_within_hour": ts_within_hour,
            "query_text_id": qtid,
            "repetition_id": f"rep_{template_id}",
            "obs_date": t.strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(rows, columns=QueryReservoir.COLUMNS)
    meta = {"schema_name": "ext_tpcds1000", "num_workloads": 1,
            "num_arrivals": len(df), "classifications": {}}
    reservoir = QueryReservoir(df, meta)

    policy = UniformForecastPolicy()
    sampler = WorkloadSampler(reservoir, policy, "ext_tpcds1000")

    start = datetime(2024, 6, 3, 9, 0, tzinfo=timezone.utc)
    end = datetime(2024, 6, 3, 13, 0, tzinfo=timezone.utc)
    workloads = sampler.sample(start, end, n_scenarios=5, seed=0)
    preview = sampler.preview(start, end)

    return reservoir, workloads, preview, start, end


@pytest.fixture(scope="module")
def test_data():
    return _build_test_data()


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestPlotReservoirHeatmap:
    def test_returns_figure(self, test_data):
        from autoslo.tuner.visualizations import plot_reservoir_heatmap
        reservoir, *_ = test_data
        fig = plot_reservoir_heatmap(reservoir)
        assert isinstance(fig, Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)


class TestPlotForecastPreview:
    def test_returns_figure(self, test_data):
        from autoslo.tuner.visualizations import plot_forecast_preview
        _, _, preview, *_ = test_data
        fig = plot_forecast_preview(preview)
        assert isinstance(fig, Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)


class TestPlotWorkloadArrivals:
    def test_returns_figure(self, test_data):
        from autoslo.tuner.visualizations import plot_workload_arrivals
        _, workloads, *_ = test_data
        fig = plot_workload_arrivals(workloads)
        assert isinstance(fig, Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)


class TestPlotQueryCountDistribution:
    def test_returns_figure(self, test_data):
        from autoslo.tuner.visualizations import plot_query_count_distribution
        _, workloads, *_ = test_data
        fig = plot_query_count_distribution(workloads)
        assert isinstance(fig, Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)


class TestPlotTemplateFrequency:
    def test_returns_figure(self, test_data):
        from autoslo.tuner.visualizations import plot_template_frequency
        reservoir, workloads, *_ = test_data
        fig = plot_template_frequency(reservoir, workloads, top_k=5)
        assert isinstance(fig, Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)


class TestPlotHourlyRates:
    def test_returns_figure(self, test_data):
        from autoslo.tuner.visualizations import plot_hourly_rates
        reservoir, workloads, _, start, end = test_data
        fig = plot_hourly_rates(reservoir, workloads, start, end)
        assert isinstance(fig, Figure)
        plt_mod = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_mod.close(fig)
