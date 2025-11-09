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


def test_strategy_name_and_class_methods():
    """
    Test that the strategy name and class methods return expected values.
    """

    class DummyTotal(TotalStrategy):
        def __init__(self):
            self.es = type("DummyES", (), {})()
            self.ps = type("DummyPS", (), {})()
            self.ss = type("DummySS", (), {})()

        def suggest_blueprint(self, latency_slo_s, *args, **kwargs):
            return Blueprint([Cluster(rpu=2)])

    d = DummyTotal()
    es_name = d.es_name()
    es_class = d.es_class()
    ps_name = d.ps_name()
    ps_class = d.ps_class()
    ss_name = d.ss_name()
    ss_class = d.ss_class()

    assert es_name == "DummyES"
    assert es_class.__name__ == "DummyES"
    assert ps_name == "DummyPS"
    assert ps_class.__name__ == "DummyPS"
    assert ss_name == "DummySS"
    assert ss_class.__name__ == "DummySS"
