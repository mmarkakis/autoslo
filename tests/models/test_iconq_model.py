"""Tests for IconqModel."""

from __future__ import annotations

import pytest

from autoslo.filesystem.structured_events import EventType
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.models.iconq_dataset_builder import build_dataset_from_trace
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_execution.trace import Trace

ICONQ_MODEL_ID = "1771539369"
TRACE_RUN_ID = "1778596415878"


@pytest.mark.integration
def test_predict_from_query_timeline_with_trained_model() -> None:
    """Ensure predict_from_dataset returns one prediction per routed query."""

    if "FILL_ME_IN" in {
        ICONQ_MODEL_ID,
        TRACE_RUN_ID,
    }:
        pytest.skip("Set ICONQ_MODEL_ID and TRACE_RUN_ID to run this test.")

    model = IconqModel.load(model_id=ICONQ_MODEL_ID)

    trace = Trace(TRACE_RUN_ID)
    dataset = build_dataset_from_trace(trace=trace, iconq_model=model)
    predictions = model.predict_from_dataset(dataset)

    assert predictions, "Expected predictions for at least one query"

    sample_prediction = next(iter(predictions.values()))
    assert isinstance(sample_prediction, ModelPrediction)
    assert sample_prediction.overall_mean_s() > 0

    log = StructuredLog.load(TRACE_RUN_ID)
    routed_query_ids = set(
        log.df.loc[
            log.df["event_type"] == EventType.QUERY_ROUTED.value, "query_id"
        ]
    )
    assert len(predictions) == len(routed_query_ids), (
        f"Expected {len(routed_query_ids)} predictions (one per QUERY_ROUTED "
        f"event) but got {len(predictions)}"
    )
