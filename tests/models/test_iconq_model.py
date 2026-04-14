"""Tests for IconqModel."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoslo.models.query_timeline import QueryTimeline
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_execution.trace import Trace

ICONQ_MODEL_ID = "1767526817"
TRACE_RUN_ID = "1763941019"


@pytest.mark.integration
def test_predict_from_query_timeline_with_trained_model() -> None:
    """Ensure predict_from_query_timeline returns predictions for real traces."""

    if "FILL_ME_IN" in {
        ICONQ_MODEL_ID,
        TRACE_RUN_ID,
    }:
        pytest.skip("Set ICONQ_MODEL_ID and TRACE_RUN_ID to run this test.")

    model = IconqModel.load(
        model_id=ICONQ_MODEL_ID,
    )

    trace = Trace(TRACE_RUN_ID)
    query_timeline = QueryTimeline(model._iconq_query_featurizer, model._iconq_interaction_featurizer)  # type: ignore[attr-defined]
    query_timeline.initialize_from_trace(trace, stage_model=model._stage_model)  # type: ignore[attr-defined]

    predictions = model.predict_from_query_timeline(query_timeline)

    assert predictions, "Expected predictions for at least one query"
    timeline_query_ids = set(query_timeline.query_ids)
    assert set(predictions).issubset(timeline_query_ids)

    sample_prediction = next(iter(predictions.values()))
    assert isinstance(sample_prediction, ModelPrediction)
    assert sample_prediction.overall_mean_s() > 0
