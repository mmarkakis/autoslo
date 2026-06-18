import torch

from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset

INPUT_SIZE = 2


def _build_dataset() -> ConcurrentQueryDataset:
    x = [
        torch.ones((1, INPUT_SIZE)),
        torch.ones((2, INPUT_SIZE)),
        torch.ones((3, INPUT_SIZE)),
    ]
    pinch_points = torch.tensor([1, 2, 0], dtype=torch.long)
    y = torch.tensor([0.5, 1.0, -1.5], dtype=torch.float32)
    query_ids = ["q10", "q20", "q30"]
    query_text_id = ["qt10", "qt20", "qt30"]
    run_ids = ["run1", "run1", "run1"]
    y_is_lower_bound = torch.tensor([False, False, True], dtype=torch.bool)
    return ConcurrentQueryDataset(
        x, pinch_points, y, query_ids, query_text_id, run_ids, y_is_lower_bound
    )


def test_dataset_len_returns_target_count():
    """Check that __len__ matches the size of y."""
    dataset = _build_dataset()
    assert len(dataset) == 3


def test_dataset_getitem_returns_expected_tensors():
    """Ensure __getitem__ returns the indexed sample."""
    dataset = _build_dataset()
    sample = dataset[1]
    assert torch.allclose(sample[0], torch.ones((2, INPUT_SIZE)))
    assert sample[1].item() == 2
    assert torch.isclose(sample[2], torch.tensor(1.0, dtype=torch.float32))
    assert sample[3] == "q20"
    assert sample[4] == "qt20"
    assert sample[5] == "run1"
    assert sample[6].item() is False


def test_collate_and_pad_orders_and_pads_batch():
    """Validate padding, sorting, and metadata alignment."""
    dataset = _build_dataset()
    batch = [dataset[i] for i in range(len(dataset))]
    (
        padded_x,
        x_len,
        pinch_points,
        targets,
        query_ids,
        query_text_ids,
        run_ids,
        y_is_lower_bound,
    ) = ConcurrentQueryDataset.collate_and_pad(batch)
    assert padded_x.shape == (3, 3, INPUT_SIZE)
    assert torch.equal(
        x_len,
        torch.tensor([3, 2, 1], dtype=torch.long),
    )
    assert torch.equal(
        pinch_points,
        torch.tensor([0, 2, 1], dtype=torch.long),
    )
    expected_targets = torch.tensor([-1.5, 1.0, 0.5], dtype=torch.float32)
    assert torch.allclose(targets, expected_targets)
    assert query_ids == ["q30", "q20", "q10"]
    assert query_text_ids == ["qt30", "qt20", "qt10"]
    assert run_ids == ["run1", "run1", "run1"]
    assert torch.equal(
        y_is_lower_bound,
        torch.tensor([True, False, False], dtype=torch.bool),
    )
    assert torch.allclose(
        padded_x[0, :3],
        torch.ones((3, INPUT_SIZE)),
    )
    assert torch.allclose(
        padded_x[1, :2],
        torch.ones((2, INPUT_SIZE)),
    )
    assert torch.allclose(
        padded_x[2, :1],
        torch.ones((1, INPUT_SIZE)),
    )
    assert torch.count_nonzero(padded_x[0, 3:]) == 0
    assert torch.count_nonzero(padded_x[1, 2:]) == 0
    assert torch.count_nonzero(padded_x[2, 1:]) == 0


# ---------------------------------------------------------------------------
# build_from_query_groups — neighbor-derived censored augmentation
# ---------------------------------------------------------------------------
import math
from unittest.mock import MagicMock

import numpy as np

from autoslo.workload_definition.query import ClusterAwareQueryId, Query, QueryTextId

_CLUSTER = "autoslo-8-testrun-0"
_RPU = 8
_FEAT_DIM = 5  # arbitrary; only the row count matters in assertions


def _q(qid: str, start_s: float) -> Query:
    """Minimal Query fixture with a stage prediction for _RPU."""
    return Query(
        query_id=qid,
        query_text_id=QueryTextId(f"qt_{qid}"),
        rel_start_time_s=start_s,
        stage_predictions_per_rpu={_RPU: 1.0},
    )


def _caqi(qid: str) -> ClusterAwareQueryId:
    return ClusterAwareQueryId.make(_CLUSTER, qid)


def _mock_featurizer() -> MagicMock:
    """
    Featurizer mock whose return array encodes context size in shape[0],
    making it easy to assert how many entries were passed per call.
    """
    m = MagicMock()
    m.featurize_one_vs_many_to_numpy.side_effect = lambda **kw: (
        np.ones((len(kw["qb_entries"]), _FEAT_DIM), dtype=np.float32),
        0,
    )
    return m


def _mock_featurizer_with_snapshots() -> tuple[MagicMock, list[list]]:
    """
    Like _mock_featurizer but also captures a deep copy of qb_entries at each
    call, so tests can inspect context contents even though qb_entries is a
    mutable list that grows across calls.
    """
    snapshots: list[list] = []

    def _side_effect(**kw):
        snapshots.append(list(kw["qb_entries"]))  # copy at call time
        return np.ones((len(kw["qb_entries"]), _FEAT_DIM), dtype=np.float32), 0

    m = MagicMock()
    m.featurize_one_vs_many_to_numpy.side_effect = _side_effect
    return m, snapshots


def _build_ds(
    base: Query,
    neighbors: list[Query],
    *,
    targets: dict | None = None,
    is_lower_bound: dict | None = None,
    augment_prob: float = 0.0,
    featurizer: MagicMock | None = None,
) -> tuple["ConcurrentQueryDataset", MagicMock]:
    if featurizer is None:
        featurizer = _mock_featurizer()
    ds = ConcurrentQueryDataset.build_from_query_groups(
        iconq_interaction_featurizer=featurizer,
        cluster_to_base_to_neighbors={_CLUSTER: {base: neighbors}},
        targets=targets,
        is_lower_bound=is_lower_bound,
        use_log_runtime=False,
        censored_observation_sample_prob=augment_prob,
    )
    return ds, featurizer


def test_build_augment_off_one_row_per_base_query():
    """With augmentation disabled, exactly one row is produced per base query."""
    base = _q("q1", 0.0)
    ds, feat = _build_ds(
        base,
        [_q("f0", 3.0), _q("f1", 7.0)],
        targets={_caqi("q1"): 10.0},
        
    )
    assert len(ds) == 1
    assert feat.featurize_one_vs_many_to_numpy.call_count == 1


def test_build_augment_no_future_neighbors_one_row():
    """Augmentation with only a past neighbor still yields one row."""
    base = _q("q1", 5.0)
    ds, _ = _build_ds(
        base,
        [_q("q0", 2.0)],  # past only
        targets={_caqi("q1"): 10.0},
        augment_prob=1.0,
    )
    assert len(ds) == 1


def test_build_augment_one_future_neighbor_no_censored_obs():
    """
    With exactly one future neighbor the 'skip last' rule suppresses the
    censored emission, leaving only the main observation.
    """
    base = _q("q1", 0.0)
    ds, _ = _build_ds(
        base,
        [_q("f0", 3.0)],
        targets={_caqi("q1"): 10.0},
        augment_prob=1.0,
    )
    assert len(ds) == 1
    assert not ds.y_is_lower_bound[0].item()


def test_build_augment_two_future_neighbors_one_censored_obs():
    """Two future neighbors → one censored obs (for f0) + one main obs."""
    base = _q("q1", 0.0)
    ds, _ = _build_ds(
        base,
        [_q("f0", 3.0), _q("f1", 7.0)],
        targets={_caqi("q1"): 10.0},
        augment_prob=1.0,
    )
    assert len(ds) == 2
    assert ds.y_is_lower_bound[0].item()   # censored
    assert not ds.y_is_lower_bound[1].item()  # main


def test_build_augment_three_future_neighbors_two_censored_obs():
    """Three future neighbors → two censored obs + one main obs."""
    base = _q("q1", 0.0)
    ds, _ = _build_ds(
        base,
        [_q("f0", 3.0), _q("f1", 7.0), _q("f2", 9.0)],
        targets={_caqi("q1"): 12.0},
        augment_prob=1.0,
    )
    assert len(ds) == 3
    assert ds.y_is_lower_bound[0].item()
    assert ds.y_is_lower_bound[1].item()
    assert not ds.y_is_lower_bound[2].item()


def test_build_augment_censored_targets_are_arrival_deltas():
    """Each censored target equals neighbor arrival time minus base start time."""
    base = _q("q1", 1.0)
    ds, _ = _build_ds(
        base,
        [_q("f0", 4.0), _q("f1", 6.0)],  # deltas: 3.0, 5.0
        targets={_caqi("q1"): 10.0},
        augment_prob=1.0,
    )
    assert len(ds) == 2
    assert math.isclose(ds.y[0].item(), 3.0)   # 4.0 - 1.0
    assert math.isclose(ds.y[1].item(), 10.0)  # main target


def test_build_augment_context_grows_incrementally():
    """
    Verify that the featurizer sees a growing context per call.

    Setup: base at t=0, no past neighbors, f0 at t=3, f1 at t=7.
      call 0 (censored): [self, f0]      → 2 entries
      call 1 (main):     [self, f0, f1]  → 3 entries
    """
    base = _q("q1", 0.0)
    feat, snapshots = _mock_featurizer_with_snapshots()
    _build_ds(
        base,
        [_q("f0", 3.0), _q("f1", 7.0)],
        targets={_caqi("q1"): 10.0},
        augment_prob=1.0,
        featurizer=feat,
    )
    assert len(snapshots) == 2
    assert len(snapshots[0]) == 2  # censored: self + f0
    assert len(snapshots[1]) == 3  # main: self + f0 + f1


def test_build_augment_past_neighbors_always_in_censored_context():
    """
    A past neighbor must appear in every censored observation's context.

    Setup: base at t=5, past at t=2, f0 at t=8, f1 at t=11.
      censored call: [self, past, f0] → 3 entries
      main call:     [self, past, f0, f1] → 4 entries
    """
    base = _q("q1", 5.0)
    past = _q("q0", 2.0)
    f0 = _q("f0", 8.0)
    f1 = _q("f1", 11.0)
    feat, snapshots = _mock_featurizer_with_snapshots()
    _build_ds(
        base,
        [past, f0, f1],
        targets={_caqi("q1"): 15.0},
        augment_prob=1.0,
        featurizer=feat,
    )
    assert len(snapshots) == 2
    assert len(snapshots[0]) == 3  # self + past + f0
    assert len(snapshots[1]) == 4  # self + past + f0 + f1
    entry_times = {e[0] for e in snapshots[0]}
    assert past.rel_start_time_s in entry_times


def test_build_augment_future_neighbors_sorted_regardless_of_input_order():
    """
    Neighbors passed in reverse arrival order must still be processed
    in ascending arrival-time order; the censored target must be for
    the earlier neighbor.
    """
    base = _q("q1", 0.0)
    f1 = _q("f1", 7.0)
    f0 = _q("f0", 3.0)
    # Pass in reverse order: f1 first, then f0
    ds, _ = _build_ds(
        base,
        [f1, f0],
        targets={_caqi("q1"): 10.0},
        augment_prob=1.0,
    )
    assert len(ds) == 2
    assert math.isclose(ds.y[0].item(), 3.0)  # f0 arrives first → delta=3.0


def test_build_augment_base_not_in_targets_no_censored_obs():
    """If the base query has no target entry, no censored obs are generated."""
    base = _q("q1", 0.0)
    ds, _ = _build_ds(
        base,
        [_q("f0", 3.0), _q("f1", 7.0)],
        targets={},  # q1 absent
        augment_prob=1.0,
    )
    assert len(ds) == 1


def test_build_augment_targets_none_no_censored_obs():
    """targets=None (inference mode) suppresses all censored obs."""
    base = _q("q1", 0.0)
    ds, _ = _build_ds(
        base,
        [_q("f0", 3.0), _q("f1", 7.0)],
        targets=None,
        augment_prob=1.0,
    )
    assert len(ds) == 1


def test_build_augment_main_obs_inherits_is_lower_bound():
    """The main observation's is_lower_bound comes from the input dict."""
    base = _q("q1", 0.0)
    caqi = _caqi("q1")
    ds, _ = _build_ds(
        base,
        [],
        targets={caqi: 5.0},
        is_lower_bound={caqi: True},
        augment_prob=1.0,
    )
    assert len(ds) == 1
    assert ds.y_is_lower_bound[0].item()


def test_build_augment_censored_ids_follow_naming_convention():
    """Synthetic ClusterAwareQueryIds embed '__censor_<j>' in the query part."""
    base = _q("q1", 0.0)
    ds, _ = _build_ds(
        base,
        [_q("f0", 3.0), _q("f1", 7.0)],
        targets={_caqi("q1"): 10.0},
        augment_prob=1.0,
    )
    assert len(ds) == 2
    censored_caqi = ds.cluster_aware_query_ids[0]
    assert "__censor_0" in censored_caqi.query_id
    assert censored_caqi.cluster_name == _CLUSTER
