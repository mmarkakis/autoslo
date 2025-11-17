import pytest

from autoslo.prediction.p_exact import PExact


def test_slo_violation_rate_comparison():
    """
    Lower slo_violation_rate should be reported as better.
    """
    p_low = PExact(slo_violation_rate=0.01, cost=10.0)
    p_high = PExact(slo_violation_rate=0.02, cost=5.0)
    assert p_low.has_lower_predicted_slo_violation_rate(p_high)
    assert not p_high.has_lower_predicted_slo_violation_rate(p_low)


def test_cost_comparison():
    """
    Lower predicted cost should be reported as better.
    """
    p_cheap = PExact(slo_violation_rate=0.05, cost=3.0)
    p_exp = PExact(slo_violation_rate=0.05, cost=7.0)
    assert p_cheap.has_lower_predicted_cost(p_exp)
    assert not p_exp.has_lower_predicted_cost(p_cheap)


def test_slo_threshold_check():
    """
    Check that _has_predicted_slo_violation_rate_under works as expected.
    """
    p = PExact(slo_violation_rate=0.05, cost=1.0)
    assert p._has_predicted_slo_violation_rate_under(0.1)
    assert not p._has_predicted_slo_violation_rate_under(0.01)


def test_strict_comparisons_on_equal_values():
    """
    Comparisons are strict (<), so equal values return False.
    """
    p1 = PExact(slo_violation_rate=0.02, cost=5.0)
    p2 = PExact(slo_violation_rate=0.02, cost=5.0)
    assert not p1.has_lower_predicted_slo_violation_rate(p2)
    assert not p1.has_lower_predicted_cost(p2)
