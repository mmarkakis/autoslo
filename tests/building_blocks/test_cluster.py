import pytest

from autoslo.blueprints.cluster import Cluster


def test_default_name_and_attributes():
    """
    Default name and attributes are set from rpu and defaults.
    """
    c = Cluster(rpu=8)
    assert c.rpu == 8
    assert c.name == "cluster_8rpu"
    assert c.cost_per_rpu_hour == Cluster.US_EAST_1_COST_PER_RPU_HOUR


def test_custom_name_and_rate():
    """
    Custom name and cost rate are accepted and stored.
    """
    c = Cluster(rpu=4, name="db", cost_per_rpu_hour=0.5)
    assert c.name == "db"
    assert c.cost_per_rpu_hour == 0.5
    assert c.rpu == 4


def test_cost_default_duration_one_hour():
    """
    cost() with no args equals rpu * rate for one hour.
    """
    c = Cluster(rpu=4)
    expected = 4 * c.cost_per_rpu_hour * 1.0
    assert c.cost() == pytest.approx(expected)


def test_cost_custom_duration_half_hour():
    """
    cost() with 1800s equals half an hour of running cost.
    """
    c = Cluster(rpu=8)
    duration_s = 1800.0  # 0.5 hour
    expected = 8 * c.cost_per_rpu_hour * 0.5
    assert c.cost(duration_s=duration_s) == pytest.approx(expected)


def test_cost_zero_duration_is_zero():
    """
    cost(0) returns 0.0.
    """
    c = Cluster(rpu=16)
    assert c.cost(duration_s=0.0) == pytest.approx(0.0)


def test_up_to_32_rpu_sizes_constant():
    """
    UP_TO_32_RPU_SIZES contains the expected supported sizes.
    """
    assert Cluster.UP_TO_32_RPU_SIZES == [4, 8, 16, 32]
