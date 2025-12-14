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
    query_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    return ConcurrentQueryDataset(x, pinch_points, y, query_ids)


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
    assert sample[3].item() == 20


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
    assert torch.equal(
        query_ids,
        torch.tensor(
            [30, 20, 10],
            dtype=torch.long,
        ),
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
