import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import autoslo.models.stage_model as stage_module
import autoslo.models.xgboost_model as xgb_module
import autoslo.utils.paths as pu
from autoslo.models.model_prediction import ModelPrediction
from autoslo.models.stage_model import StageModel


def _make_query_text(template: str) -> str:
    """
    Return a query text string containing the template marker Trace expects.
    """
    return f"SELECT 1\\n-- Filename: query{template}.sql\\n"


def _create_trace_run(root: Path,
                      run_id: str,
                      latencies: list[float],
                      templates: list[str],
                      cluster_name: str = "cluster") -> Path:
    """
    Emit a minimal parquet-backed run consumable by Trace.
    """
    run_dir = root / run_id
    run_dir.mkdir()
    base = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    for offset, (latency, template) in enumerate(
        zip(latencies, templates), start=1
    ):
        start_time = base + pd.Timedelta(seconds=offset)
        end_time = start_time + pd.Timedelta(seconds=latency)
        rows.append(
            {
                "query_id": f"q{offset}",
                "start_time": start_time,
                "end_time": end_time,
                "elapsed_time": latency * 1_000_000,
                "status": "success",
                "result_cache_hit": False,
                "query_type": "SELECT",
                "query_text": _make_query_text(template),
            }
        )
    history = pd.DataFrame(rows)
    history.to_parquet(
        run_dir / f"sys_query_history+{cluster_name}.parquet",
        index=False,
    )
    return run_dir


@pytest.mark.unit
def test_stage_model_requires_cache_configuration(monkeypatch: pytest.MonkeyPatch
                                                 ) -> None:
    """
    Ensure StageModel enforces cache model configuration requirements.
    """

    class DummyXGB:
        def predict(self, _: dict[str, str]) -> dict[str, ModelPrediction]:
            return {}

    monkeypatch.setattr(
        stage_module.XGBoostModel,
        "load",
        staticmethod(lambda _: DummyXGB()),
    )

    with pytest.raises(ValueError):
        StageModel(
            cache_model_id=None,
            cache_model_init_params=None,
            cache_model_train_params=None,
            xgboost_model_id="xgb-id",
        )


@pytest.mark.unit
def test_stage_model_predicts_with_loaded_models(monkeypatch: pytest.MonkeyPatch
                                                ) -> None:
    """
    Confirm StageModel combines cache hits with XGBoost fallbacks.
    """

    class CacheStub:
        def predict(self, _: dict[str, str]) -> dict[str, ModelPrediction | None]:
            return {
                "cached": ModelPrediction(mean_s=1.0, std_s=0.1),
                "ml": None,
            }

    class XGBStub:
        def predict(self, _: dict[str, str]) -> dict[str, ModelPrediction]:
            return {"ml": ModelPrediction(mean_s=5.0, std_s=0.5)}

    monkeypatch.setattr(
        stage_module.CacheModel,
        "load",
        staticmethod(lambda _: CacheStub()),
    )
    monkeypatch.setattr(
        stage_module.XGBoostModel,
        "load",
        staticmethod(lambda _: XGBStub()),
    )

    model = StageModel(
        cache_model_id="cache-id",
        xgboost_model_id="xgb-id",
    )
    results = model.predict(
        {
            "cached": _make_query_text("001_001"),
            "ml": _make_query_text("002_001"),
        }
    )

    assert results["cached"].mean_s == pytest.approx(1.0)
    assert results["ml"].mean_s == pytest.approx(5.0)


@pytest.mark.unit
def test_stage_model_trains_and_saves_new_models(monkeypatch: pytest.MonkeyPatch
                                                ) -> None:
    """
    Ensure StageModel trains, saves, and reuses newly built sub-models.
    """

    class CacheStub:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.trained: list[dict[str, Any]] = []
            self.saved = False

        def train(self, **kwargs: Any) -> None:
            self.trained.append(kwargs)

        def save(self) -> str:
            self.saved = True
            return "cache-trained"

        def predict(self, _: dict[str, str]) -> dict[str, ModelPrediction | None]:
            return {
                "cached": ModelPrediction(mean_s=2.0, std_s=0.2),
                "ml": None,
            }

    class XGBStub:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.trained: list[dict[str, Any]] = []
            self.saved = False

        def train(self, **kwargs: Any) -> tuple[float, float]:
            self.trained.append(kwargs)
            return (0.1, 0.2)

        def save(self) -> str:
            self.saved = True
            return "xgb-trained"

        def predict(self, _: dict[str, str]) -> dict[str, ModelPrediction]:
            return {"ml": ModelPrediction(mean_s=4.0, std_s=0.4)}

    cache_instance = CacheStub()
    xgb_instance = XGBStub()
    monkeypatch.setattr(
        stage_module,
        "CacheModel",
        lambda **kwargs: cache_instance,
    )
    monkeypatch.setattr(
        stage_module,
        "XGBoostModel",
        lambda **kwargs: xgb_instance,
    )

    model = StageModel(
        cache_model_init_params={"best_effort": False},
        cache_model_train_params={"run_ids": ["run-1"], "from_scratch": True},
        xgboost_model_init_params={"n_estimators": 10},
        xgboost_model_train_params={"run_ids": ["run-2"], "from_scratch": True},
    )
    results = model.predict(
        {
            "cached": _make_query_text("001_001"),
            "ml": _make_query_text("002_001"),
        }
    )

    assert cache_instance.trained[0]["run_ids"] == ["run-1"]
    assert cache_instance.saved
    assert xgb_instance.trained[0]["run_ids"] == ["run-2"]
    assert xgb_instance.saved
    assert results["cached"].mean_s == pytest.approx(2.0)
    assert results["ml"].mean_s == pytest.approx(4.0)


@pytest.mark.integration
def test_stage_model_end_to_end_with_real_models(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch
                                                 ) -> None:
    """
    Exercise StageModel with real sub-models over synthetic traces.
    """

    class SimpleFeaturizer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._id = "simple-featurizer"

        def save(self) -> str:
            return self._id

        @staticmethod
        def load(_: str) -> "SimpleFeaturizer":
            return SimpleFeaturizer()

        def featurize(self, query_text: str) -> list[float]:
            template = xgb_module.Trace.extract_temp_and_q_idxs(query_text)
            return self._featurize_template(template)

        def featurize_trace(self, trace: Any) -> dict[str, list[float]]:
            return {
                query_id: self._featurize_template(template)
                for query_id, template in trace.tpcds_temp_and_q_idxs.items()
            }

        @staticmethod
        def _featurize_template(template: str) -> list[float]:
            temp = xgb_module.Trace.extract_temp(template)
            idx = xgb_module.Trace.extract_q_idx(template)
            return [float(temp), float(idx)]

    runs_root = tmp_path / "runs"
    data_root = tmp_path / "data"
    runs_root.mkdir()
    data_root.mkdir()
    monkeypatch.setattr(
        pu, "get_runs_path", lambda: os.fspath(runs_root)
    )
    monkeypatch.setattr(
        pu, "get_data_path", lambda: os.fspath(data_root)
    )
   
    monkeypatch.setattr(
        xgb_module,
        "IconqQueryFeaturizer",
        SimpleFeaturizer,
    )

    run_id = "stage-run"
    _create_trace_run(
        runs_root,
        run_id,
        [2.0, 4.0, 6.0],
        ["001_001", "001_001", "002_005"],
    )

    model = StageModel(
        cache_model_init_params={
            "enable_template_cache": True,
            "best_effort": False,
        },
        cache_model_train_params={"run_ids": [run_id], "from_scratch": True},
        xgboost_model_init_params={
            "iconq_query_featurizer_init_params": {},
            "n_estimators": 25,
            "max_depth": 3,
            "eta": 0.3,
            "early_stopping_rounds": 5,
            "random_seed": 0,
        },
        xgboost_model_train_params={"run_ids": [run_id], "from_scratch": True},
    )

    predictions = model.predict(
        {
            "cached": _make_query_text("001_001"),
            "ml": _make_query_text("003_007"),
        }
    )

    assert model._cache_model_id is not None
    assert model._xgboost_model_id is not None
    assert predictions["cached"].mean_s == pytest.approx(3.0)
    assert predictions["cached"].std_s == pytest.approx(1.0)
    assert predictions["ml"].mean_s > 0.0
    assert np.isfinite(predictions["ml"].mean_s)
