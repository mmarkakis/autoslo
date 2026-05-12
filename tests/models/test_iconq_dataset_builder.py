from datetime import datetime, timedelta, timezone
from typing import cast

import numpy as np
import pytest

from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.iconq_dataset_builder import build_dataset_from_trace
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.models.stage_model import StageModel
from autoslo.workload_definition.query import QueryTextId
from autoslo.workload_execution.trace import Trace

BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
_DUMMY_QTID = QueryTextId("ext_tpcds1000#1#001")
_DUMMY_CLUSTER = "autoslo-4-default-0"
_DUMMY_RUN_ID = "test-run-0"


def _dt(seconds: float) -> datetime:
    return BASE_TIME + timedelta(seconds=seconds)


class DummyTrace:
    def __init__(self, specs: list[tuple[str, float, float]]) -> None:
        self.run_id = _DUMMY_RUN_ID
        self.query_ids = [qid for qid, _, _ in specs]
        self._query_text_ids = {qid: _DUMMY_QTID for qid, _, _ in specs}
        self._arrival = {qid: _dt(start) for qid, start, _ in specs}
        self._completion = {qid: _dt(end) for qid, _, end in specs}
        self._was_aborted = {qid: False for qid, _, _ in specs}

    @property
    def query_text_ids(self):
        import pandas as pd

        return pd.Series(self._query_text_ids)

    def arrival_times(self) -> dict[str, datetime]:
        return self._arrival

    def completion_times(self) -> dict[str, datetime]:
        return self._completion

    def was_aborted(self) -> dict[str, bool]:
        return self._was_aborted

    @staticmethod
    def cluster_name_from_query_id(query_id: str) -> str:
        return _DUMMY_CLUSTER


class DummyIconqQueryFeaturizer:
    num_dims = 3

    def featurize_from_query_text_id(
        self, query_text_id: QueryTextId
    ) -> list[float]:
        return [0.0, 1.5, 2.5]

    def featurize_from_query_text_id_as_numpy(
        self, query_text_id: QueryTextId
    ) -> np.ndarray:
        return np.array([0.0, 1.5, 2.5], dtype=np.float32)


class DummyStageModel:
    def predict_from_query_text_id(
        self, query_text_ids: dict[str, QueryTextId], cluster_rpu: int
    ) -> dict[str, ModelPrediction]:
        return {qid: ModelPrediction(mean_s=[5.0]) for qid in query_text_ids}


class DummyIconqInteractionFeaturizer:
    num_dims = 4

    def _get_rpu(self, cluster_name: str) -> int:
        return 8

    def featurize_one_vs_many_to_numpy(
        self,
        cluster_name: str,
        qa_query_text_id: QueryTextId,
        qa_start_time_s: float,
        qa_latency_prediction: float,
        qb_entries: list[tuple[float, QueryTextId, float, bool]],
    ) -> tuple[np.ndarray, int]:
        sorted_entries = sorted(qb_entries, key=lambda e: e[0])
        pinch_idx = next(
            j for j, (_, _, _, is_self) in enumerate(sorted_entries) if is_self
        )
        return (
            np.zeros((len(sorted_entries), self.num_dims), dtype=np.float32),
            pinch_idx,
        )


class DummyIconqModel:
    trained_on_log_runtime = False

    def __init__(self) -> None:
        self.iconq_query_featurizer = cast(
            IconqQueryFeaturizer, DummyIconqQueryFeaturizer()
        )
        self.iconq_interaction_featurizer = cast(
            IconqInteractionFeaturizer, DummyIconqInteractionFeaturizer()
        )
        self.stage_model = cast(StageModel, DummyStageModel())


def _build(specs: list[tuple[str, float, float]], **kwargs):
    trace = cast(Trace, DummyTrace(specs))
    model = cast(IconqModel, DummyIconqModel())
    return build_dataset_from_trace(
        trace=trace, iconq_model=model, run_id=_DUMMY_RUN_ID, **kwargs
    )


def test_overlapping_queries_are_neighbors() -> None:
    dataset = _build([("q1", 0.0, 10.0), ("q2", 5.0, 15.0), ("q3", 20.0, 30.0)])
    # q1 and q2 overlap → each has the other as neighbor (2 rows in qb_entries)
    # q3 is isolated → only self-entry (1 row in qb_entries)
    qid_to_idx = {qid: i for i, qid in enumerate(dataset.query_ids)}
    assert "q1" in qid_to_idx and "q2" in qid_to_idx and "q3" in qid_to_idx

    q1_tensor = dataset[qid_to_idx["q1"]][0]  # x tensor for q1
    q2_tensor = dataset[qid_to_idx["q2"]][0]
    q3_tensor = dataset[qid_to_idx["q3"]][0]

    # q1 and q2 overlap: their interaction tensors have 2 rows (self + one neighbor)
    assert (
        q1_tensor.shape[0] == 2
    ), f"q1 expected 2 rows, got {q1_tensor.shape[0]}"
    assert (
        q2_tensor.shape[0] == 2
    ), f"q2 expected 2 rows, got {q2_tensor.shape[0]}"
    # q3 is isolated: only self-row
    assert (
        q3_tensor.shape[0] == 1
    ), f"q3 expected 1 row, got {q3_tensor.shape[0]}"


def test_non_overlapping_queries_each_have_only_self_row() -> None:
    dataset = _build([("q1", 0.0, 5.0), ("q2", 10.0, 15.0)])
    for i, qid in enumerate(dataset.query_ids):
        tensor = dataset[i][0]
        assert (
            tensor.shape[0] == 1
        ), f"{qid} expected 1 row (no neighbors), got {tensor.shape[0]}"


def test_fixed_window_includes_non_overlapping_neighbors() -> None:
    # q1(0–1) and q2(5–6) do not overlap, but are within a 10s window
    dataset = _build(
        [("q1", 0.0, 1.0), ("q2", 5.0, 6.0)],
        use_fixed_window_radius_s=10.0,
    )
    qid_to_idx = {qid: i for i, qid in enumerate(dataset.query_ids)}
    # Both should include the other as neighbor
    assert dataset[qid_to_idx["q1"]][0].shape[0] == 2
    assert dataset[qid_to_idx["q2"]][0].shape[0] == 2


def test_fixed_window_excludes_distant_queries() -> None:
    # q1(0–1) and q2(100–101) — outside any reasonable radius
    dataset = _build(
        [("q1", 0.0, 1.0), ("q2", 100.0, 101.0)],
        use_fixed_window_radius_s=10.0,
    )
    for i in range(len(dataset.query_ids)):
        assert dataset[i][0].shape[0] == 1


def test_fixed_window_max_neighbors_per_side() -> None:
    # q3 is the base; q1 and q2 come before it within the window; only 1 allowed per side
    dataset = _build(
        [("q1", 0.0, 1.0), ("q2", 2.0, 3.0), ("q3", 4.0, 5.0)],
        use_fixed_window_radius_s=10.0,
        use_fixed_window_max_neighbors_per_side=1,
    )
    qid_to_idx = {qid: i for i, qid in enumerate(dataset.query_ids)}
    # q3 has q1 and q2 before it, but capped at 1 → only q2 (closest before)
    assert dataset[qid_to_idx["q3"]][0].shape[0] == 2  # self + 1 neighbor


def test_run_id_stored_in_dataset() -> None:
    dataset = _build([("q1", 0.0, 5.0)])
    assert dataset.run_ids == [_DUMMY_RUN_ID]


def test_aborted_query_is_lower_bound() -> None:
    trace = cast(Trace, DummyTrace([("q1", 0.0, 5.0)]))
    trace._was_aborted["q1"] = True  # type: ignore[attr-defined]
    model = cast(IconqModel, DummyIconqModel())
    dataset = build_dataset_from_trace(
        trace=trace, iconq_model=model, run_id=_DUMMY_RUN_ID
    )
    assert bool(dataset.y_is_lower_bound[0].item()) is True


def test_empty_trace_returns_empty_dataset() -> None:
    dataset = _build([])
    assert len(dataset) == 0
