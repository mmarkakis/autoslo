from pathlib import Path
from typing import Any

import pytest

import autoslo.models.cache_model as cache_model
from autoslo.models.cache_model import CacheModel


def _install_dummy_trace(
    monkeypatch: pytest.MonkeyPatch, runs: dict[str, dict[str, Any]]
) -> None:
    """
    Install dummy Trace class configured with canned training data.
    """

    class DummyTrace:
        def __init__(self, run_id: str) -> None:
            payload = runs[run_id]
            self.latencies_s = payload["latencies"]
            self.tpcds_temp_and_q_idxs = payload["pairs"]

        @staticmethod
        def extract_temp_and_q_idxs(query_text: Any) -> Any:
            return query_text

        @staticmethod
        def extract_temp(temp_and_q_idx: str) -> int:
            return int(temp_and_q_idx.split("_")[0])

        @staticmethod
        def extract_q_idx(temp_and_q_idx: str) -> int:
            return int(temp_and_q_idx.split("_")[1])

    monkeypatch.setattr(cache_model, "Trace", DummyTrace)


@pytest.mark.unit
def test_cache_model_cache_hits_and_misses_no_template_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure cache hits and misses return correct runtimes when template caching
    is disabled.
    """
    runs = {
        "run-a": {
            "latencies": [1.0, 2.0, 3.0],
            "pairs": ["1_1", "1_1", "2_5"],
        },
    }
    _install_dummy_trace(monkeypatch, runs)
    model = CacheModel(enable_template_cache=False)
    model.train(["run-a"], from_scratch=True)

    predictions = model.predict(
        {
            "hit": "1_1",
            "miss": "2_99",
        },
    )

    assert predictions["hit"][0] == pytest.approx(1.5)
    assert predictions["hit"][1] == pytest.approx(0.5)
    assert predictions["miss"][0] == pytest.approx(2.0)
    assert predictions["miss"][1] == pytest.approx(0.81649658)
    assert model._overall_mean_runtime_s == pytest.approx(2.0)
    assert model._overall_std_runtime_s == pytest.approx(0.81649658)


@pytest.mark.unit
def test_cache_model_cache_hits_and_misses_with_template_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure cache hits and misses return correct runtimes when template caching
    is enabled.
    """
    runs = {
        "run-a": {
            "latencies": [1.0, 2.0, 3.0],
            "pairs": ["1_1", "1_1", "2_5"],
        },
    }
    _install_dummy_trace(monkeypatch, runs)
    model = CacheModel(enable_template_cache=True)
    model.train(["run-a"], from_scratch=True)

    predictions = model.predict(
        {
            "hit": "1_1",
            "template_miss": "2_99",
            "miss": "3_42",
        },
    )

    assert predictions["hit"][0] == pytest.approx(1.5)
    assert predictions["hit"][1] == pytest.approx(0.5)
    assert predictions["template_miss"][0] == pytest.approx(3.0)
    assert predictions["template_miss"][1] == pytest.approx(0.0)
    assert predictions["miss"][0] == pytest.approx(2.0)
    assert predictions["miss"][1] == pytest.approx(0.81649658)
    assert model._overall_mean_runtime_s == pytest.approx(2.0)
    assert model._overall_std_runtime_s == pytest.approx(0.81649658)


@pytest.mark.unit
def test_cache_model_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Ensure that saving and loading the model preserves its state.
    """
    runs = {
        "run-a": {
            "latencies": [1.0, 2.0, 3.0],
            "pairs": ["1_1", "1_1", "2_5"],
        },
    }
    _install_dummy_trace(monkeypatch, runs)
    model = CacheModel(enable_template_cache=True)
    model.train(["run-a"], from_scratch=True)

    ts = model.save(parent_save_dir=str(tmp_path))
    loaded_model = CacheModel.load(ts, parent_load_dir=str(tmp_path))

    assert loaded_model._cache == model._cache
    assert loaded_model._overall_mean_runtime_s == model._overall_mean_runtime_s
    assert loaded_model._overall_std_runtime_s == model._overall_std_runtime_s
    assert loaded_model._enable_template_cache == model._enable_template_cache
