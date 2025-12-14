import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import autoslo.models.xgboost_model as xgb_module
from autoslo.models.xgboost_model import XGBoostModel


def _make_query_text(template: str) -> str:
    """
    Return a query text string that Trace can parse for the template string.
    """
    return f"SELECT 1\\nabc{template}tail"


def _create_trace_run(
    root: Path,
    run_id: str,
    latencies: list[float],
    templates: list[str],
    cluster_name: str = "cluster",
) -> Path:
    """
    Create a minimal run consumable by Trace using synthetic parquet files.
    """
    run_dir = root / run_id
    run_dir.mkdir()
    base = pd.Timestamp("2024-01-01 00:00:00")
    records = []
    for idx, (latency, template) in enumerate(
        zip(latencies, templates), start=1
    ):
        query_id = f"q{idx}"
        start = base + pd.Timedelta(seconds=idx)
        end = start + pd.Timedelta(seconds=latency)
        records.append(
            {
                "query_id": query_id,
                "start_time": start,
                "end_time": end,
                "elapsed_time": latency * 1_000_000,
                "status": "success",
                "result_cache_hit": False,
                "query_type": "SELECT",
                "query_text": _make_query_text(template),
            }
        )
    history = pd.DataFrame(records)
    history_path = run_dir / f"sys_query_history+{cluster_name}.parquet"
    history.to_parquet(history_path)
    return run_dir


@pytest.mark.unit
def test_xgboost_model_predict_uses_featurizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure prediction uses featurizer output and applies inverse log transform.
    """

    class PredictStub:
        def __init__(self) -> None:
            self.received: list[list[float]] | None = None

        def predict(self, array: np.ndarray) -> np.ndarray:
            self.received = array.tolist()
            return np.array([np.log1p(4.0)])

        def evals_result(self) -> dict[str, Any]:
            return {}

    predict_stub = PredictStub()
    monkeypatch.setattr(
        xgb_module, "XGBRegressor", lambda *args, **kwargs: predict_stub
    )

    class FeaturizerStub:
        def featurize(self, query_text: str) -> list[float]:
            return [1.0, 2.0]

        def featurize_trace(self, trace: Any) -> dict[str, list[float]]:
            return {}

        def featurize_from_tpcds_temp_and_q_idx(
            self, template: str
        ) -> list[float]:
            return [1.0, 2.0]

    featurizer_stub = FeaturizerStub()
    monkeypatch.setattr(
        xgb_module.IconqQueryFeaturizer,
        "load",
        staticmethod(lambda _: featurizer_stub),
    )

    model = XGBoostModel(
        iconq_query_featurizer_id="stub", train_on_log_runtime=True
    )
    predictions = model.predict({"q1": _make_query_text("001_001")})

    assert predict_stub.received == [[1.0, 2.0]]
    assert predictions["q1"].overall_mean_s() == pytest.approx(4.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_xgboost_model_train_prepares_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure training prepares arrays and returns the last recorded losses.
    """

    class DummyTrace:
        def __init__(self, run_id: str) -> None:
            self.latencies_s = pd.Series(
                [1.0, 2.0, 3.0], index=["q1", "q2", "q3"]
            )
            self.tpcds_temp_and_q_idxs = pd.Series(
                ["001_001", "001_002", "001_003"],
                index=["q1", "q2", "q3"],
            )

    monkeypatch.setattr(xgb_module, "Trace", DummyTrace)

    class TraceFeaturizerStub:
        def featurize(self, query_text: str) -> list[float]:
            return [0.0]

        def featurize_trace(self, trace: DummyTrace) -> dict[str, list[float]]:
            return {
                "q1": [0.1],
                "q2": [0.2],
                "q3": [0.3],
            }

    featurizer_stub = TraceFeaturizerStub()
    monkeypatch.setattr(
        xgb_module.IconqQueryFeaturizer,
        "load",
        staticmethod(lambda _: featurizer_stub),
    )

    class FitStub:
        def __init__(self) -> None:
            self.fit_called = False
            self.recorded_eval_set: list[tuple[np.ndarray, np.ndarray]] = []

        def fit(
            self,
            train_X: np.ndarray,
            train_y: np.ndarray,
            eval_set: list[tuple[np.ndarray, np.ndarray]],
            verbose: bool,
        ) -> None:
            self.fit_called = True
            self.train_X = train_X
            self.train_y = train_y
            self.recorded_eval_set = eval_set

        def predict(self, array: np.ndarray) -> np.ndarray:
            return np.array([1.0])

        def evals_result(self) -> dict[str, dict[str, list[float]]]:
            return {
                "validation_0": {"mae": [0.5]},
                "validation_1": {"mae": [0.7]},
            }

    fit_stubs: list[FitStub] = []

    def stub_factory(*args: Any, **kwargs: Any) -> FitStub:
        stub = FitStub()
        fit_stubs.append(stub)
        return stub

    monkeypatch.setattr(xgb_module, "XGBRegressor", stub_factory)

    model = XGBoostModel(
        iconq_query_featurizer_id="stub",
        n_estimators=5,
        early_stopping_rounds=2,
    )
    train_loss, val_loss = model.train(["run-1"], from_scratch=True)

    final_stub = fit_stubs[-1]
    assert final_stub.fit_called
    assert final_stub.train_X.shape == (2, 1)
    assert final_stub.recorded_eval_set[-1][0].shape == (1, 1)
    assert train_loss == pytest.approx(0.5)
    assert val_loss == pytest.approx(0.7)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_xgboost_model_with_real_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Exercise XGBoostModel with real Trace data and persistence.
    """
    runs_root = tmp_path / "runs"
    data_root = tmp_path / "data"
    runs_root.mkdir()
    data_root.mkdir()
    monkeypatch.setattr(
        xgb_module.pu, "get_runs_path", lambda: os.fspath(runs_root)
    )
    monkeypatch.setattr(
        xgb_module.pu, "get_data_path", lambda: os.fspath(data_root)
    )
    run_id = "xgb-trace"
    _create_trace_run(
        runs_root,
        run_id,
        [2.0, 4.0, 6.0, 8.0, 10.0],
        ["001_001", "001_002", "002_001", "002_002", "003_003"],
    )

    def to_features(template: str) -> list[float]:
        template_id, query_idx = template.split("_")
        return [float(template_id), float(query_idx)]

    class SimpleFeaturizer:
        def featurize(self, query_text: str) -> list[float]:
            template = xgb_module.Trace.extract_temp_and_q_idxs(query_text)
            return to_features(template)

        def featurize_trace(self, trace: Any) -> dict[str, list[float]]:
            return {
                query_id: to_features(template)
                for query_id, template in trace.tpcds_temp_and_q_idxs.items()
            }

        def featurize_from_tpcds_temp_and_q_idx(
            self, template: str
        ) -> list[float]:
            return to_features(template)

    simple_featurizer = SimpleFeaturizer()
    monkeypatch.setattr(
        xgb_module.IconqQueryFeaturizer,
        "load",
        staticmethod(lambda _: simple_featurizer),
    )

    model = XGBoostModel(
        iconq_query_featurizer_id="stub",
        n_estimators=25,
        max_depth=3,
        eta=0.3,
        early_stopping_rounds=5,
        random_seed=0,
    )
    train_loss, val_loss = model.train([run_id], from_scratch=True)

    assert train_loss >= 0.0
    assert val_loss >= 0.0

    timestamp = model.save()
    save_dir = data_root / "xgboost_models" / timestamp
    assert save_dir.is_dir()
    assert (save_dir / "params.yml").is_file()
    assert (save_dir / "model.json").is_file()

    loaded = XGBoostModel.load(timestamp)
    predictions = loaded.predict(
        {
            "seen": _make_query_text("001_001"),
            "unseen": _make_query_text("004_001"),
        },
    )

    assert np.isfinite(predictions["seen"].overall_mean_s())
    assert predictions["seen"].overall_mean_s() > 0.0
    assert predictions["unseen"].overall_mean_s() > 0.0
