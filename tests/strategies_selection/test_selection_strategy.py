import pytest

from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.prediction.p_exact import PExact
from autoslo.strategies_selection.selection_strategy import (
    SelectionStrategy,
)


def test_cannot_instantiate_abstract_selection_strategy():
    """
    Cannot instantiate SelectionStrategy abstract base class.
    """
    with pytest.raises(TypeError):
        SelectionStrategy()


def test_select_returns_blueprint_instance():
    """
    A concrete select() implementation should return a Blueprint instance.
    """

    class DummySelection(SelectionStrategy):
        def select(self, bp_to_pred, *args, **kwargs):
            # return the first blueprint key from the mapping
            return next(iter(bp_to_pred.keys()))

    bp = Blueprint([Cluster(rpu=4)])
    pred = PExact(slo_violation_rate=0.0, cost=1.0)
    mapping = {bp: pred}

    s = DummySelection()
    chosen = s.select(mapping)
    assert isinstance(chosen, Blueprint)
    assert chosen is bp


def test_select_receives_args_and_kwargs():
    """
    select() should receive positional and keyword args and be callable.
    """

    class RecorderSelection(SelectionStrategy):
        def __init__(self):
            self.called = None

        def select(self, bp_to_pred, *args, **kwargs):
            self.called = (bp_to_pred, args, kwargs)
            # return any blueprint from the mapping
            return next(iter(bp_to_pred.keys()))

    bp = Blueprint([Cluster(rpu=8)])
    pred = PExact(slo_violation_rate=0.1, cost=2.0)
    mapping = {bp: pred}

    r = RecorderSelection()
    out = r.select(mapping, 1, "x", flag=True)
    assert out is bp
    assert r.called is not None
    assert r.called[0] == mapping
    assert r.called[1] == (1, "x")
    assert r.called[2] == {"flag": True}
