import pytest

from autoslo.blueprints.cluster import Cluster

from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from psycopg2.pool import ThreadedConnectionPool


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


def test_cost_default_duration_one_hour():
    """
    cost() with no args equals rpu * rate for one hour.
    """
    c = Cluster(rpu=4, name="cluster_4rpu")
    expected = 4 * c.cost_per_rpu_hour * 1.0
    assert c.cost() == pytest.approx(expected)


def test_cost_custom_duration_half_hour():
    """
    cost() with 1800s equals half an hour of running cost.
    """
    c = Cluster(rpu=8, name="cluster_8rpu")
    duration_s = 1800.0  # 0.5 hour
    expected = 8 * c.cost_per_rpu_hour * 0.5
    assert c.cost(duration_s=duration_s) == pytest.approx(expected)


def test_cost_zero_duration_is_zero():
    """
    cost(0) returns 0.0.
    """
    c = Cluster(rpu=16, name="cluster_16rpu")
    assert c.cost(duration_s=0.0) == pytest.approx(0.0)


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


def test_from_config_key_error():
    """
    from_config with unknown cluster name raises KeyError.
    """
    with pytest.raises(KeyError):
        Cluster.from_config("non_existent_cluster")


def test_from_dict_creates_cluster():
    """
    from_dict creates a Cluster instance with correct attributes.
    """
    cluster_dict = {
        "rpu": 16,
        "cluster_name": "test_cluster",
        "cost_per_rpu_hour": 0.4,
        "host": "localhost",
        "port": 5439,
        "user": "admin",
        "password": "password",
        "dbname": "dev",
    }
    cluster = Cluster.from_dict(cluster_dict)
    assert cluster.rpu == 16
    assert cluster.name == "test_cluster"
    assert cluster.cost_per_rpu_hour == 0.4
    assert cluster.conn_info.host == "localhost"
    assert cluster.conn_info.port == 5439
    assert cluster.conn_info.user == "admin"
    assert cluster.conn_info.password == "password"
    assert cluster.conn_info.dbname == "dev"

