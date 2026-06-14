from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch

from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.models.iconq_model import _validate_runtime_net_input_size
from autoslo.workload_definition.query import QueryTextId


class _DummyQueryFeaturizer:
    def __init__(self, num_dims: int):
        self.num_dims = num_dims

    def featurize_from_query_text_id_as_numpy(self, query_text_id: object) -> np.ndarray:
        if str(query_text_id) == "qa":
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return np.array([4.0, 5.0, 6.0], dtype=np.float32)


def _make_featurizer(
    version: str,
    ignore_cluster_size: bool = False,
    q_dim: int = 3,
) -> IconqInteractionFeaturizer:
    f = IconqInteractionFeaturizer.__new__(IconqInteractionFeaturizer)
    f._iconq_query_featurizer = cast(Any, _DummyQueryFeaturizer(q_dim))
    f._ignore_cluster_size = ignore_cluster_size
    f._interaction_feature_version = version
    return f


def test_v1_dims_and_indices_match_legacy_layout() -> None:
    f = _make_featurizer("v1")

    assert f.num_dims == 11
    assert f.arrival_time_diff_dim_idx == 8
    assert f.arrival_time_sign_dim_idx == 9
    assert f.rpu_dim_idx == 10


def test_v2_dims_and_indices_keep_legacy_core_positions() -> None:
    f = _make_featurizer("v2")

    assert f.num_dims == 16
    assert f.arrival_time_diff_dim_idx == 8
    assert f.arrival_time_sign_dim_idx == 9
    assert f.rpu_dim_idx == 10


def test_v2_rpu_derived_features_are_computed_as_expected() -> None:
    f = _make_featurizer("v2")

    arr, pinch_idx = f.featurize_one_vs_many_to_numpy(
        rpu=8,
        qa_query_text_id=QueryTextId("qa"),
        qa_start_time_s=10.0,
        qa_latency_prediction=2.0,
        qb_entries=[
            (11.0, QueryTextId("qb_b"), 3.0, False),
            (9.0, QueryTextId("qa"), 2.0, True),
        ],
    )

    assert arr.shape == (2, 16)
    assert pinch_idx == 0

    rpu_dim = f.rpu_dim_idx

    np.testing.assert_allclose(arr[:, rpu_dim], np.array([8.0, 8.0], dtype=np.float32))
    np.testing.assert_allclose(arr[:, rpu_dim + 1], np.array([3.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(arr[:, rpu_dim + 2], np.array([0.125, 0.125], dtype=np.float32))
    np.testing.assert_allclose(arr[:, rpu_dim + 3], np.array([16.0, 16.0], dtype=np.float32))
    np.testing.assert_allclose(arr[:, rpu_dim + 4], np.array([16.0, 24.0], dtype=np.float32))
    np.testing.assert_allclose(arr[:, rpu_dim + 5], np.array([0.5, 0.625], dtype=np.float32))


def test_v2_ignore_cluster_size_zeros_entire_rpu_block() -> None:
    f = _make_featurizer("v2", ignore_cluster_size=True)

    arr, _ = f.featurize_one_vs_many_to_numpy(
        rpu=16,
        qa_query_text_id=QueryTextId("qa"),
        qa_start_time_s=10.0,
        qa_latency_prediction=2.0,
        qb_entries=[
            (11.0, QueryTextId("qb_b"), 3.0, False),
            (9.0, QueryTextId("qa"), 2.0, True),
        ],
    )

    rpu_dim = f.rpu_dim_idx
    np.testing.assert_allclose(arr[:, rpu_dim : rpu_dim + 6], 0.0)


def test_validate_runtime_net_input_size_raises_for_mismatch() -> None:
    state_dict = {
        "_bn.weight": torch.zeros(11),
        "_in_model.0.weight": torch.zeros((64, 11)),
    }

    with pytest.raises(ValueError, match="Checkpoint/input feature mismatch"):
        _validate_runtime_net_input_size(
            state_dict=state_dict,
            expected_input_size=16,
            interaction_feature_version="v2",
        )


def test_validate_runtime_net_input_size_accepts_matching_shapes() -> None:
    state_dict = {
        "_bn.weight": torch.zeros(16),
        "_in_model.0.weight": torch.zeros((64, 16)),
    }

    _validate_runtime_net_input_size(
        state_dict=state_dict,
        expected_input_size=16,
        interaction_feature_version="v2",
    )
