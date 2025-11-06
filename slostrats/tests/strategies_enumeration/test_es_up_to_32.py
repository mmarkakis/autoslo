import pytest

from slostrats.building_blocks.blueprint import Blueprint
from slostrats.building_blocks.cluster import Cluster
from slostrats.strategies_enumeration.es_up_to_32 import ESUpTo32


def test_enumerate_returns_list_of_blueprints():
    """
    enumerate() should return a list of Blueprint instances.
    """
    es = ESUpTo32()
    res = es.enumerate()
    assert isinstance(res, list)
    assert res, "Expected at least one blueprint"
    assert all(isinstance(bp, Blueprint) for bp in res)


def test_produced_rpu_sizes_match_expected():
    """
    The RPUs of produced clusters should equal the expected sizes.
    """
    es = ESUpTo32()
    blueprints = es.enumerate()
    actual_sizes = [bp.clusters[0].rpu for bp in blueprints]
    expected = list(Cluster.UP_TO_32_RPU_SIZES)
    assert actual_sizes == expected


def test_uses_simple_blueprints_factory_equivalence():
    """
    enumerate() should produce the same RPU sizes as the simple factory.
    """
    es = ESUpTo32()
    produced = es.enumerate()
    factory = Blueprint.simple_blueprints_up_to_32_rpu()
    prod_sizes = [bp.clusters[0].rpu for bp in produced]
    fact_sizes = [bp.clusters[0].rpu for bp in factory]
    assert prod_sizes == fact_sizes


def test_enumerate_accepts_args_and_kwargs():
    """
    enumerate() should accept positional and keyword args and still return
    the expected blueprints.
    """
    es = ESUpTo32()
    blueprints = es.enumerate(1, "x", flag=True)
    sizes = [bp.clusters[0].rpu for bp in blueprints]
    assert sizes == list(Cluster.UP_TO_32_RPU_SIZES)
