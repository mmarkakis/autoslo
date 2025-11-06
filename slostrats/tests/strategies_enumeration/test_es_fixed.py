import pytest

from slostrats.strategies_enumeration.es_fixed import ESFixed
from slostrats.building_blocks.blueprint import Blueprint
from slostrats.building_blocks.cluster import Cluster


def test_enumerate_returns_single_blueprint():
    """
    enumerate() should return a list containing a single Blueprint.
    """
    es = ESFixed(rpu=8)
    res = es.enumerate()
    assert isinstance(res, list)
    assert len(res) == 1
    assert isinstance(res[0], Blueprint)


def test_blueprint_contains_cluster_with_given_rpu():
    """
    The produced Blueprint must contain a Cluster with the specified RPU.
    """
    rpu = 16
    es = ESFixed(rpu=rpu)
    bp = es.enumerate()[0]
    assert bp.clusters[0].rpu == rpu


def test_cluster_has_default_name():
    """
    Cluster default name follows the pattern cluster_{rpu}rpu.
    """
    rpu = 4
    es = ESFixed(rpu=rpu)
    bp = es.enumerate()[0]
    expected_name = f"cluster_{rpu}rpu"
    assert bp.clusters[0].name == expected_name


def test_enumerate_accepts_args_and_kwargs():
    """
    enumerate() should accept positional and keyword arguments and still
    produce the expected blueprint.
    """
    es = ESFixed(rpu=32)
    bp = es.enumerate(1, "x", flag=True)[0]
    assert bp.clusters[0].rpu == 32
