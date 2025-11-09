from datetime import datetime
from typing import Any, Dict, List

import pytest

from slostrats.strategies_total.ts_replay_past import (
    TSReplayPast,
    TSReplayPast1Cost,
)
from slostrats.strategies_selection.ss_min_cost_once_acceptable import (
    SSMinCostOnceAcceptable,
)


class DummyBlueprint:
    """
    Very small blueprint-like object used as a key in mappings passed to
    selection strategies. It is intentionally hashable and carries a name.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # helpful debugging if a test fails
        return f"<DummyBlueprint {self.name}>"


def test_suggest_blueprint_passes_full_mapping_to_selection_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure TSReplayPast.suggest_blueprint enumerates candidate blueprints,
    calls the predictor for each, and passes the full blueprint->prediction
    mapping into the provided selection strategy.
    """
    # Prepare TSReplayPast instance with a recorder selection strategy.
    recorded: Dict[str, Any] = {}

    class RecorderSelection:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # accept any constructor signature
            pass

        def select(self, bp_to_pred: Dict[Any, Any], *args: Any, **kwargs: Any) -> Any:
            recorded["mapping"] = bp_to_pred
            # return the first blueprint as the chosen one
            return next(iter(bp_to_pred.keys()))

    ts = TSReplayPast(
        slo_violation_rate_threshold=0.01,
        window_size=1,
        selection_strategy=RecorderSelection,
    )

    # Replace enumerator results and predictor behaviour
    bps = [DummyBlueprint("a"), DummyBlueprint("b"), DummyBlueprint("c")]
    monkeypatch.setattr(ts.es, "enumerate", lambda: bps)

    predictions = {bp.name: f"pred-{bp.name}" for bp in bps}

    def fake_predict(workload_name: str, day_idx: int, blueprint: Any, latency_slo_s: float):
        return predictions[blueprint.name]

    monkeypatch.setattr(ts.ps, "predict", fake_predict)

    chosen = ts.suggest_blueprint("wl", 0, latency_slo_s=1.0)

    # Ensure selection strategy was called and received the full mapping
    assert "mapping" in recorded
    mapping = recorded["mapping"]
    assert set(mapping.keys()) == set(bps)
    # Values are exactly what our fake_predict returned
    assert all(mapping[bp] == predictions[bp.name] for bp in bps)
    # The returned blueprint is one of the enumerated blueprints
    assert chosen in bps


def test_suggest_blueprint_propagates_predict_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If the predictor raises an error for a candidate blueprint, ensure the
    error propagates through suggest_blueprint so callers can observe it.
    """
    ts = TSReplayPast(
        slo_violation_rate_threshold=0.1,
        window_size=1,
        selection_strategy=lambda *a, **k: None,
    )

    # Single blueprint that will cause predict to raise.
    bp = DummyBlueprint("bad")
    monkeypatch.setattr(ts.es, "enumerate", lambda: [bp])

    def raising_predict(*args: Any, **kwargs: Any):
        raise RuntimeError("predict failure")

    monkeypatch.setattr(ts.ps, "predict", raising_predict)

    with pytest.raises(RuntimeError):
        ts.suggest_blueprint("wl", 0, latency_slo_s=1.0)


def test_ts_replay_past1cost_constructs_with_min_cost_selection() -> None:
    """
    TSReplayPast1Cost is a convenience subclass that should use the
    SSMinCostOnceAcceptable selection strategy class for selection.
    """
    ts1 = TSReplayPast1Cost(slo_violation_rate_threshold=0.05)
    # The selection strategy instance attached should be an instance of the
    # SSMinCostOnceAcceptable class (or a compatible subclass).
    assert isinstance(ts1.ss, SSMinCostOnceAcceptable)

    