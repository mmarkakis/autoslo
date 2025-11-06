import pytest

from slostrats.building_blocks.blueprint import Blueprint
from slostrats.building_blocks.cluster import Cluster
from slostrats.strategies_total.total_strategy import TotalStrategy


def test_cannot_instantiate_abstract_total_strategy():
    """
    Cannot instantiate TotalStrategy abstract base class.
    """
    with pytest.raises(TypeError):
        TotalStrategy()


def test_suggest_blueprint_returns_blueprint_instance():
    """
    Concrete suggest_blueprint should return the suggested Blueprint.
    """

    class DummyTotal(TotalStrategy):
        def suggest_blueprint(self, latency_slo_s, *args, **kwargs):
            return Blueprint([Cluster(rpu=4)])

    d = DummyTotal()
    bp = d.suggest_blueprint(0.5)
    assert isinstance(bp, Blueprint)
    assert bp.clusters[0].rpu == 4


def test_suggest_blueprint_receives_args_and_kwargs():
    """
    suggest_blueprint should receive positional and keyword arguments.
    """

    class RecorderTotal(TotalStrategy):
        def __init__(self):
            self.called = None

        def suggest_blueprint(self, latency_slo_s, *args, **kwargs):
            self.called = (latency_slo_s, args, kwargs)
            return Blueprint([Cluster(rpu=8)])

    r = RecorderTotal()
    out = r.suggest_blueprint(1.0, 42, "x", flag=True)
    assert isinstance(out, Blueprint)
    assert r.called is not None
    assert r.called[0] == 1.0
    assert r.called[1] == (42, "x")
    assert r.called[2] == {"flag": True}
