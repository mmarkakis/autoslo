import pytest

from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.prediction.p_exact import PExact
from autoslo.strategies_selection.ss_min_slo_violation_rate import (
    SSMinSLOViolationRate,
)


def test_select_empty_mapping_raises():
    """
    Selecting with an empty mapping raises ValueError.
    """
    selector = SSMinSLOViolationRate()
    with pytest.raises(ValueError):
        selector.select({})


def test_selects_blueprint_with_min_slo_rate():
    """
    Selects the blueprint with the smallest predicted SLO violation rate.
    """
    bp1 = Blueprint([Cluster(rpu=4)])
    bp2 = Blueprint([Cluster(rpu=8)])
    bp3 = Blueprint([Cluster(rpu=16)])

    mapping = {
        bp1: PExact(slo_violation_rate=0.05, cost=10.0),
        bp2: PExact(slo_violation_rate=0.02, cost=20.0),
        bp3: PExact(slo_violation_rate=0.03, cost=5.0),
    }

    selector = SSMinSLOViolationRate()
    chosen = selector.select(mapping)
    assert chosen is bp2


def test_ties_preserve_first_seen_blueprint():
    """
    When multiple blueprints tie on SLO rate, the first inserted blueprint
    is selected (stable behavior).
    """
    bp_first = Blueprint([Cluster(rpu=4)])
    bp_second = Blueprint([Cluster(rpu=8)])
    mapping = {
        bp_first: PExact(slo_violation_rate=0.01, cost=100.0),
        bp_second: PExact(slo_violation_rate=0.01, cost=1.0),
    }

    selector = SSMinSLOViolationRate()
    chosen = selector.select(mapping)
    assert chosen is bp_first


def test_select_accepts_extra_args_and_kwargs():
    """
    select() accepts extra args/kwargs without affecting the result.
    """
    bp_a = Blueprint([Cluster(rpu=4)])
    bp_b = Blueprint([Cluster(rpu=8)])
    mapping = {
        bp_a: PExact(slo_violation_rate=0.04, cost=5.0),
        bp_b: PExact(slo_violation_rate=0.02, cost=7.0),
    }

    selector = SSMinSLOViolationRate()
    chosen = selector.select(mapping, 1, "x", flag=True)
    assert chosen is bp_b
