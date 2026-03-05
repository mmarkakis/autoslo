import pytest

from autoslo.blueprints.cluster import Cluster

from autoslo.blueprints.cluster_conn_info import ClusterConnInfo


def test_default_name_and_attributes():
    """
    Default attributes are set from rpu and defaults (name must now be supplied).
    """
    c = Cluster(rpu=8, name="cluster_8rpu")
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


def test_cost_per_second():
    """
    cost_per_second returns the correct per-second cost.
    """
    c = Cluster(rpu=4, name="cluster_4rpu")
    expected = 4 * Cluster.US_EAST_1_COST_PER_RPU_HOUR / 3600
    assert c.cost_per_second == pytest.approx(expected)


def test_up_to_32_rpu_sizes_constant():
    """
    UP_TO_32_RPU_SIZES contains the expected supported sizes.
    """
    assert Cluster.UP_TO_32_RPU_SIZES == [4, 8, 16, 32]


def test_missing_name_raises_type_error():
    """
    Constructing Cluster without the required name argument raises TypeError.
    """
    with pytest.raises(TypeError):
        Cluster(rpu=8)


def test_rpu_for_cluster_name_dynamic():
    """
    rpu_for_cluster_name parses RPU from dynamic cluster names.
    """
    assert Cluster.rpu_for_cluster_name("cluster_8_1234_0") == 8
    assert Cluster.rpu_for_cluster_name("cluster_32_9999_5") == 32


def test_rpu_for_cluster_name_invalid():
    """
    rpu_for_cluster_name raises ValueError for unparseable names.
    """
    with pytest.raises(ValueError):
        Cluster.rpu_for_cluster_name("badname")


def test_cost_per_second_for_rpu():
    """
    Static utility computes cost_per_second from RPU.
    """
    expected = 8 * Cluster.US_EAST_1_COST_PER_RPU_HOUR / 3600
    assert Cluster.cost_per_second_for_rpu(8) == pytest.approx(expected)


def test_frozen_cluster():
    """
    Cluster is a frozen dataclass — attribute mutation is forbidden.
    """
    c = Cluster(rpu=8, name="cluster_8_0_0")
    with pytest.raises(AttributeError):
        c.rpu = 16  # type: ignore[misc]


def test_cluster_equality():
    """
    Equality compares all fields.
    """
    c1 = Cluster(rpu=8, name="a")
    c2 = Cluster(rpu=8, name="a")
    c3 = Cluster(rpu=16, name="a")
    assert c1 == c2
    assert c1 != c3


def test_new_factory():
    """
    Cluster.new() creates clusters with auto-generated names.
    """
    c = Cluster.new(rpu=8)
    assert c.rpu == 8
    assert c.name.startswith("cluster_8_")
    assert c.conn_info is None

