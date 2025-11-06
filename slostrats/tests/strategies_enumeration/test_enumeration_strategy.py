import pytest

from slostrats.building_blocks.blueprint import Blueprint
from slostrats.building_blocks.cluster import Cluster
from slostrats.strategies_enumeration.enumeration_strategy import (
    EnumerationStrategy,
)


def test_cannot_instantiate_abstract_enumeration_strategy():
    """
    Cannot instantiate EnumerationStrategy abstract base class.
    """
    with pytest.raises(TypeError):
        EnumerationStrategy()


def test_enumerate_returns_list_of_blueprints():
    """
    enumerate() should return a list of Blueprint instances.
    """

    class Dummy(EnumerationStrategy):
        def enumerate(self, *args, **kwargs):
            c = Cluster(rpu=4)
            return [Blueprint([c])]

    d = Dummy()
    result = d.enumerate()
    assert isinstance(result, list)
    assert all(isinstance(bp, Blueprint) for bp in result)
    assert result[0].clusters[0].rpu == 4


def test_enumerate_receives_args_and_kwargs():
    """
    enumerate() should accept positional and keyword arguments.
    """

    class Recorder(EnumerationStrategy):
        def __init__(self):
            self.called = None

        def enumerate(self, *args, **kwargs):
            self.called = (args, kwargs)
            return []

    r = Recorder()
    res = r.enumerate(1, "a", flag=True)
    assert res == []
    assert r.called is not None
    assert r.called[0] == (1, "a")
    assert r.called[1] == {"flag": True}
