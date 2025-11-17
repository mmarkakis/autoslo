import re
from typing import Dict

import pytest

from autoslo.prediction.p_exact import PExact
from autoslo.strategies_prediction import ps_replay_past as module

# ...existing imports...
from autoslo.strategies_prediction.ps_replay_past import PSReplayPast


class DummyTrace:
    """Lightweight fake Trace with the minimal API used by PSReplayPast."""

    def __init__(self, total: int, violating: int, billed_s: float) -> None:
        self._total = total
        self._violating = violating
        self._billed_s = billed_s

    def num_queries_with_latency_over(self, latency_slo_s: float) -> int:
        # latency_slo_s is ignored; the test controls violating directly
        return self._violating

    def num_queries(self) -> int:
        return self._total

    def billed_s(self) -> float:
        return self._billed_s


def make_blueprint(cluster_name: str, rpu: int, cost_factor: float = 1.0):
    """
    Create a minimal Blueprint-like object used by PSReplayPast tests.

    Arguments:
    - cluster_name: name of the single cluster.
    - rpu: requests-per-second value for the cluster.
    - cost_factor: multiplier applied to billed seconds to compute cost.
    """

    class Cluster:
        def __init__(self, rpu: int) -> None:
            self.rpu = rpu

    class BlueprintLike:
        def __init__(
            self, cluster_name: str, rpu: int, cost_factor: float
        ) -> None:
            self.clusters = [Cluster(rpu)]
            self.cluster_names = [cluster_name]
            self._cost_factor = cost_factor

        def total_cost(self, billed_map: Dict[str, float]) -> float:
            # Return sum of billed_s values times factor.
            return sum(billed_map.values()) * self._cost_factor

    return BlueprintLike(cluster_name, rpu, cost_factor)


def _install_monkeypatched_traces(
    monkeypatch: pytest.MonkeyPatch, traces_by_idx: Dict[int, DummyTrace]
) -> None:
    """
    Monkeypatch the Composite and Trace usage in the module under test
    so PSReplayPast uses our DummyTrace instances based on the day index.
    """
    # Provide any directory (unused by our fake Trace.from_path).
    monkeypatch.setattr(
        module.Composite,
        "dir_for_composite_workload",
        lambda workload_name: "/fake/dir",
    )

    def fake_from_path(path: str) -> DummyTrace:
        # extract day index from path like .../day_traces/day_{idx}/...
        m = re.search(r"day_(\d+)", path)
        if not m:
            raise ValueError("Could not parse day index from path")
        idx = int(m.group(1))
        return traces_by_idx[idx]

    monkeypatch.setattr(module.Trace, "from_path", staticmethod(fake_from_path))


def test_predict_pooled_average(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that PSReplayPast predicts pooled violation rate and total cost
    when per_period_average is False.
    """
    # Arrange: three past days with known totals/violations/billed_s
    traces = {
        0: DummyTrace(total=100, violating=10, billed_s=5.0),
        1: DummyTrace(total=200, violating=30, billed_s=6.0),
        2: DummyTrace(total=50, violating=5, billed_s=1.0),
    }
    _install_monkeypatched_traces(monkeypatch, traces)

    blueprint = make_blueprint("clusterA", rpu=100, cost_factor=1.0)
    strategy = PSReplayPast(window_size=3, per_period_average=False)

    # Act
    pred: PExact = strategy.predict(
        workload_name="wl",
        day_idx=3,
        blueprint=blueprint,
        latency_slo_s=0.1,
    )

    # Assert pooled violation rate = sum(violating)/sum(total)
    expected_viols = 10 + 30 + 5
    expected_total = 100 + 200 + 50
    assert (
        pytest.approx(pred.slo_violation_rate, rel=1e-9)
        == expected_viols / expected_total
    )
    # Assert total cost is sum of billed_s (cost_factor=1.0)
    assert pred.cost == pytest.approx(5.0 + 6.0 + 1.0)


def test_predict_per_period_average(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that PSReplayPast computes the average of per-period violation
    rates when per_period_average is True.
    """
    # Arrange: three days with different per-day rates
    traces = {
        0: DummyTrace(total=100, violating=10, billed_s=2.0),  # 0.10
        1: DummyTrace(total=100, violating=20, billed_s=3.0),  # 0.20
        2: DummyTrace(total=100, violating=30, billed_s=4.0),  # 0.30
    }
    _install_monkeypatched_traces(monkeypatch, traces)

    blueprint = make_blueprint("clusterB", rpu=50, cost_factor=1.0)
    strategy = PSReplayPast(window_size=5, per_period_average=True)

    # Act
    pred: PExact = strategy.predict(
        workload_name="wl",
        day_idx=3,
        blueprint=blueprint,
        latency_slo_s=0.2,
    )

    # Assert average rate is (0.10 + 0.20 + 0.30) / 3 = 0.20
    assert pytest.approx(pred.slo_violation_rate, rel=1e-9) == 0.2
    # Cost should be sum of billed_s
    assert pred.cost == pytest.approx(2.0 + 3.0 + 4.0)


def test_window_clips_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that the strategy clips the earliest day index at zero and still
    uses the available past traces when day_idx < window_size.
    """
    # day_idx=1 and window_size=5 -> only index 0 is considered
    traces = {0: DummyTrace(total=10, violating=1, billed_s=1.5)}
    _install_monkeypatched_traces(monkeypatch, traces)

    blueprint = make_blueprint("clusterC", rpu=10, cost_factor=2.0)
    strategy = PSReplayPast(window_size=5, per_period_average=False)

    pred: PExact = strategy.predict(
        workload_name="wl",
        day_idx=1,
        blueprint=blueprint,
        latency_slo_s=0.5,
    )

    assert pytest.approx(pred.slo_violation_rate, rel=1e-9) == 1 / 10
    # cost_factor=2.0 -> billed_s 1.5 * 2.0
    assert pred.cost == pytest.approx(1.5 * 2.0)


def test_no_past_traces_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test the current implementation behavior when there are no past
    traces (e.g., day_idx == 0). The implementation currently raises
    a ValueError in this case.
    """
    # No traces for indices < 0 when day_idx == 0 -> empty past_traces
    _install_monkeypatched_traces(monkeypatch, traces_by_idx={})

    blueprint = make_blueprint("clusterD", rpu=1, cost_factor=1.0)
    strategy = PSReplayPast(window_size=3, per_period_average=False)

    with pytest.raises(ValueError):
        _ = strategy.predict(
            workload_name="wl",
            day_idx=0,
            blueprint=blueprint,
            latency_slo_s=1.0,
        )
