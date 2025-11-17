import pytest

from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.prediction.p_exact import PExact
from autoslo.strategies_selection.ss_min_cost_once_acceptable import (
    SSMinCostOnceAcceptable,
)


def test_select_min_cost_among_acceptable():
    """
    Selects the blueprint with minimum cost among acceptable ones.
    """
    bp1 = Blueprint([Cluster(rpu=4)])
    bp2 = Blueprint([Cluster(rpu=8)])
    bp3 = Blueprint([Cluster(rpu=16)])

    # bp1 and bp2 are below the threshold; bp2 has lower cost -> choose bp2
    mapping = {
        bp1: PExact(slo_violation_rate=0.01, cost=10.0),
        bp2: PExact(slo_violation_rate=0.02, cost=5.0),
        bp3: PExact(slo_violation_rate=0.10, cost=1.0),
    }

    selector = SSMinCostOnceAcceptable(slo_violation_rate_threshold=0.03)
    chosen = selector.select(mapping)
    assert chosen is bp2


def test_fallback_to_min_slo_when_none_acceptable():
    """
    Falls back to blueprint with minimum SLO rate if none are acceptable.
    """
    bp_a = Blueprint([Cluster(rpu=4)])
    bp_b = Blueprint([Cluster(rpu=8)])

    mapping = {
        bp_a: PExact(slo_violation_rate=0.05, cost=100.0),
        bp_b: PExact(slo_violation_rate=0.02, cost=200.0),
    }

    # threshold is low so no prediction is strictly under it -> fallback
    selector = SSMinCostOnceAcceptable(slo_violation_rate_threshold=0.01)
    chosen = selector.select(mapping)
    # bp_b has the lower SLO violation rate and should be selected
    assert chosen is bp_b


def test_select_empty_mapping_raises():
    """
    Providing an empty bp_to_pred mapping raises ValueError.
    """
    selector = SSMinCostOnceAcceptable(slo_violation_rate_threshold=0.1)
    with pytest.raises(ValueError):
        selector.select({})
