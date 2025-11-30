import os
import tempfile
from datetime import datetime
from typing import List

import pandas as pd
import pytest
import yaml

import autoslo.utils.paths as pu
from autoslo.featurization.f_minimal import FMinimal
from autoslo.workload_execution.trace import Trace


class StatSeq:
    """A tiny sequence wrapper providing mean() and quantile()."""

    def __init__(self, values: List[float]) -> None:
        self.values = list(values)

    def mean(self) -> float:
        return sum(self.values) / len(self.values)

    def quantile(self, q: float) -> float:
        # simple empirical quantile: nearest-rank method
        if not self.values:
            return 0.0
        vals = sorted(self.values)
        idx = int(round((len(vals) - 1) * q))
        return float(vals[max(0, min(len(vals) - 1, idx))])


class DummyTrace:
    """Minimal Trace-like object implementing the used methods."""

    def __init__(self, num_queries: int, values: List[float], rpu: int) -> None:
        self.num_queries = num_queries
        self._values = values
        self._rpu = rpu

    def mbytes_scanned(self) -> StatSeq:
        return StatSeq(self._values)

    def num_joins(self) -> StatSeq:
        return StatSeq(self._values)

    def num_scans(self) -> StatSeq:
        return StatSeq(self._values)

    def num_aggregates(self) -> StatSeq:
        return StatSeq(self._values)

    def rpu_per_cluster(self) -> dict[str, int]:
        return {"default_cluster": self._rpu}


def test_fminimal_default_summary_metric() -> None:
    """
    FMinimal with the default 'mean' metric should return floats and the
    expected number of features.
    """
    f = FMinimal()
    rpu = 5
    trace = DummyTrace(num_queries=3, values=[1.0, 2.0, 3.0], rpu=rpu)
    vec = f._featurize_trace_impl(trace)  # type: ignore
    assert isinstance(vec, list)
    assert len(vec) == len(f.feature_names)
    assert all(isinstance(x, float) for x in vec)
    assert vec[0] == float(trace.num_queries)
    assert vec[-1] == float(rpu)


def test_feature_names_reflect_metric() -> None:
    """
    The feature_names should include the configured summary metric suffix
    (for example 'p95' when summary_metric='p95').
    """
    f = FMinimal(summary_metric="p95")
    names = f.feature_names
    assert any("mbytes_scanned_p95" == n for n in names)


def test_invalid_summary_metric_raises() -> None:
    """
    Constructing FMinimal with an unsupported summary_metric should raise
    a ValueError.
    """
    with pytest.raises(ValueError):
        FMinimal(summary_metric="unsupported_metric")


@pytest.mark.integration
def test_fminimal_integration_with_real_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    End-to-end integration: write minimal parquet files and ensure
    FMinimal featurizes the Trace as expected.
    """
    with tempfile.TemporaryDirectory() as td:
        run_id = "run1"
        run_dir = os.path.join(td, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # sys_query_history: three queries with start/end and elapsed_time
        history = pd.DataFrame(
            {
                "query_id": ["q1", "q2", "q3"],
                "start_time": [
                    pd.Timestamp(datetime(2020, 1, 1, 0, 0, 0)),
                    pd.Timestamp(datetime(2020, 1, 1, 0, 0, 10)),
                    pd.Timestamp(datetime(2020, 1, 1, 0, 0, 20)),
                ],
                "end_time": [
                    pd.Timestamp(datetime(2020, 1, 1, 0, 0, 1)),
                    pd.Timestamp(datetime(2020, 1, 1, 0, 0, 11)),
                    pd.Timestamp(datetime(2020, 1, 1, 0, 0, 21)),
                ],
                "elapsed_time": [1000000, 1000000, 1000000],  # us
            }
        )
        history.to_parquet(
            os.path.join(run_dir, "sys_query_history+cluster.parquet"),
            engine="pyarrow",
        )

        # sys_query_detail: produce output_bytes per query (bytes)
        detail = pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q2", "q3"],
                "step_name": ["scan", "scan", "scan", "scan"],
                "output_bytes": [1_000_000, 2_000_000, 0, 2_000_000],
            }
        )
        detail.to_parquet(
            os.path.join(run_dir, "sys_query_detail+cluster.parquet"),
            engine="pyarrow",
        )

        # sys_query_explain: plan nodes to count joins/scans per query
        explain = pd.DataFrame(
            {
                "query_id": ["q1", "q1", "q2", "q3"],
                "plan_node": ["Join", "Scan", "Scan", "Join"],
            }
        )
        explain.to_parquet(
            os.path.join(run_dir, "sys_query_explain+cluster.parquet"),
            engine="pyarrow",
        )

        # write run_params.yml so Trace can discover the blueprint name
        run_params = {"blueprint_name": "bp1"}
        with open(os.path.join(run_dir, "run_params.yml"), "w") as f:
            yaml.safe_dump(run_params, f)

        # Point get_runs_path at our temp dir
        monkeypatch.setattr(pu, "get_runs_path", lambda: td)

        # Monkeypatch config helpers to return dicts referencing our
        # blueprint and cluster with the desired RPU.
        rpu = 5
        monkeypatch.setattr(
            pu,
            "get_blueprint_dicts_from_config",
            lambda: {"bp1": {"cluster_names": ["default_cluster"]}},
        )
        monkeypatch.setattr(
            pu,
            "get_cluster_dicts_from_config",
            lambda: {"default_cluster": {"rpu": rpu}},
        )

        trace = Trace(run_id)
        featurizer = FMinimal()
        vec = featurizer._featurize_trace_impl(trace)  # type: ignore

        # Expected values:
        # total queries = 3
        # mbytes per query = [3,0,2] -> mean = 5/3
        # num_joins per query = [1,0,1] -> mean = 2/3
        # num_scans per query = [1,1,0] -> mean = 2/3
        expected = [
            float(3),
            pytest.approx(5.0 / 3.0),
            pytest.approx(2.0 / 3.0),
            pytest.approx(2.0 / 3.0),
            pytest.approx(0.0),
            float(rpu),
        ]
        assert len(vec) == len(expected)
        for got, exp in zip(vec, expected):
            if isinstance(exp, pytest.approx.__class__):
                assert got == exp
            else:
                assert got == exp


@pytest.mark.integration
def test_fminimal_integration_p95_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that configuring summary_metric='p95' works with a real Trace and
    that feature_names reflect the suffix.
    """
    with tempfile.TemporaryDirectory() as td:
        run_id = "run2"
        run_dir = os.path.join(td, run_id)
        os.makedirs(run_dir, exist_ok=True)

        history = pd.DataFrame(
            {
                "query_id": ["a", "b", "c"],
                "start_time": [
                    pd.Timestamp(datetime(2020, 1, 2)),
                    pd.Timestamp(datetime(2020, 1, 2, 0, 0, 10)),
                    pd.Timestamp(datetime(2020, 1, 2, 0, 0, 20)),
                ],
                "end_time": [
                    pd.Timestamp(datetime(2020, 1, 2, 0, 0, 1)),
                    pd.Timestamp(datetime(2020, 1, 2, 0, 0, 11)),
                    pd.Timestamp(datetime(2020, 1, 2, 0, 0, 21)),
                ],
                "elapsed_time": [1000000, 1000000, 1000000],
            }
        )
        history.to_parquet(
            os.path.join(run_dir, "sys_query_history+cluster.parquet"),
            engine="pyarrow",
        )

        detail = pd.DataFrame(
            {
                "query_id": ["a", "b", "c"],
                "step_name": ["scan", "scan", "scan"],
                "output_bytes": [1_000_000, 4_000_000, 9_000_000],
            }
        )
        detail.to_parquet(
            os.path.join(run_dir, "sys_query_detail+cluster.parquet"),
            engine="pyarrow",
        )

        explain = pd.DataFrame(
            {
                "query_id": ["a", "b", "c"],
                "plan_node": ["Scan", "Scan", "Scan"],
            }
        )
        explain.to_parquet(
            os.path.join(run_dir, "sys_query_explain+cluster.parquet"),
            engine="pyarrow",
        )

        # write run_params.yml so Trace can discover the blueprint name
        run_params = {"blueprint_name": "bp2"}
        with open(os.path.join(run_dir, "run_params.yml"), "w") as f:

            yaml.safe_dump(run_params, f)

        monkeypatch.setattr(pu, "get_runs_path", lambda: td)

        # Monkeypatch config helpers to return dicts referencing our
        # blueprint and cluster with the desired RPU.
        rpu = 1
        monkeypatch.setattr(
            pu,
            "get_blueprint_dicts_from_config",
            lambda: {"bp2": {"cluster_names": ["default_cluster"]}},
        )
        monkeypatch.setattr(
            pu,
            "get_cluster_dicts_from_config",
            lambda: {"default_cluster": {"rpu": rpu}},
        )

        trace = Trace(run_id)
        featurizer = FMinimal(summary_metric="p95")
        vec = featurizer._featurize_trace_impl(trace)  # type: ignore

        # p95 of mbytes [1,4,9]
        p95_correct = pd.Series([1.0, 4.0, 9.0]).quantile(0.95)
        assert any("mbytes_scanned_p95" == n for n in featurizer.feature_names)
        assert pytest.approx(p95_correct) == vec[1]
