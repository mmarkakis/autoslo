"""Tests for IconqModel."""

from __future__ import annotations

import pytest

from autoslo.models.iconq_dataset_builder import build_dataset_from_trace
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_execution.trace import Trace

ICONQ_MODEL_ID = "1767526817"
TRACE_RUN_ID = "1763941019"


@pytest.mark.integration
def test_predict_from_query_timeline_with_trained_model() -> None:
    """Ensure predict_from_dataset returns predictions for real traces."""

    if "FILL_ME_IN" in {
        ICONQ_MODEL_ID,
        TRACE_RUN_ID,
    }:
        pytest.skip("Set ICONQ_MODEL_ID and TRACE_RUN_ID to run this test.")

    model = IconqModel.load(
        model_id=ICONQ_MODEL_ID,
    )

    trace = Trace(TRACE_RUN_ID)
    dataset = build_dataset_from_trace(
        trace=trace, iconq_model=model, run_id=TRACE_RUN_ID
    )
    predictions = model.predict_from_dataset(dataset)

    all_predictions = {
        qid: pred
        for run_preds in predictions.values()
        for qid, pred in run_preds.items()
    }
    assert all_predictions, "Expected predictions for at least one query"

    sample_prediction = next(iter(all_predictions.values()))
    assert isinstance(sample_prediction, ModelPrediction)
    assert sample_prediction.overall_mean_s() > 0
