from datetime import datetime, timedelta

import pytest

from slostrats.building_blocks.blueprint import Blueprint
from slostrats.building_blocks.cluster import Cluster
from slostrats.strategies_prediction.ps_past_window import PSPastWindow


class DummyTrace:
    """
    Simple fake Trace implementing the methods PSPastWindow needs.
    """

    def __init__(self, num_queries, violating, billed_s):
        self._num = num_queries
        self._violating = violating
        self._billed = billed_s

    def num_queries_with_latency_over(self, latency_slo_s):
        return self._violating

    def num_queries(self):
        return self._num

    def billed_s(self):
        return self._billed


def test_predict_empty_past_traces_raises():
    """
    Empty past_traces should raise a ValueError.
    """
    ps = PSPastWindow(window_size=1, per_period_average=False)
    bp = Blueprint([Cluster(rpu=4)])
    with pytest.raises(ValueError):
        ps.predict(bp, 0.5, {})


def test_predict_pooled_rate_and_cost():
    """
    Pooled violation rate and total cost are computed correctly.
    """
    # two traces: 10/100 and 20/200 -> pooled rate = 30/300 = 0.1
    t1 = DummyTrace(num_queries=100, violating=10, billed_s=3600.0)
    t2 = DummyTrace(num_queries=200, violating=20, billed_s=7200.0)
    now = datetime.now()
    past = {now - timedelta(hours=2): t1, now - timedelta(hours=1): t2}

    ps = PSPastWindow(window_size=2, per_period_average=False)
    bp = Blueprint([Cluster(rpu=4)])
    pred = ps.predict(bp, 1.0, past)

    expected_rate = 30.0 / 300.0
    # cost: cluster.cost(3600) + cluster.cost(7200)
    c = bp.clusters[0]
    expected_cost = c.cost(duration_s=3600.0) + c.cost(duration_s=7200.0)

    assert pred.slo_violation_rate == pytest.approx(expected_rate)
    assert pred.cost == pytest.approx(expected_cost)


def test_predict_per_period_average():
    """
    Per-period average of rates is computed when requested.
    """
    # trace rates: 10/100 = 0.1, 40/200 = 0.2 -> average = 0.15
    t1 = DummyTrace(num_queries=100, violating=10, billed_s=3600.0)
    t2 = DummyTrace(num_queries=200, violating=40, billed_s=3600.0)
    now = datetime.now()
    past = {now - timedelta(minutes=30): t1, now: t2}

    ps = PSPastWindow(window_size=2, per_period_average=True)
    bp = Blueprint([Cluster(rpu=8)])
    pred = ps.predict(bp, 0.5, past)

    expected_rate = (0.1 + 0.2) / 2.0
    expected_cost = bp.clusters[0].cost(duration_s=3600.0) * 2.0

    assert pred.slo_violation_rate == pytest.approx(expected_rate)
    assert pred.cost == pytest.approx(expected_cost)


def test_window_size_limits_to_most_recent_traces():
    """
    Only the most recent `window_size` traces should be considered for
    prediction.
    """
    now = datetime.utcnow()
    # oldest, middle, newest
    t_old = DummyTrace(num_queries=10, violating=1, billed_s=3600.0)
    t_mid = DummyTrace(num_queries=10, violating=2, billed_s=3600.0)
    t_new = DummyTrace(num_queries=10, violating=3, billed_s=3600.0)
    past = {
        now - timedelta(hours=3): t_old,
        now - timedelta(hours=2): t_mid,
        now - timedelta(hours=1): t_new,
    }

    ps = PSPastWindow(window_size=2, per_period_average=False)
    bp = Blueprint([Cluster(rpu=4)])
    pred = ps.predict(bp, 0.5, past)

    # only mid and new considered -> violations = 2+3, queries = 10+10
    expected_rate = 5.0 / 20.0
    expected_cost = bp.clusters[0].cost(duration_s=3600.0) * 2.0

    assert pred.slo_violation_rate == pytest.approx(expected_rate)
    assert pred.cost == pytest.approx(expected_cost)
