import pytest

from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster


def test_init_empty_raises():
    """
    Ensure initializing a Blueprint with an empty cluster list raises
    ValueError.
    """
    with pytest.raises(ValueError):
        Blueprint([])


def test_init_multiple_raises():
    """
    Ensure initializing a Blueprint with more than one cluster raises
    ValueError (temporary constraint).
    """
    c1 = Cluster(rpu=4, name="cluster_4rpu")
    c2 = Cluster(rpu=8, name="cluster_8rpu")
    with pytest.raises(ValueError):
        Blueprint([c1, c2])


def test_clusters_and_names():
    """
    Verify that clusters and cluster_names properties return the correct
    cluster and its name.
    """
    cluster = Cluster(rpu=8, name="cluster_8rpu")
    blueprint = Blueprint([cluster])
    assert blueprint.clusters == [cluster]
    assert blueprint.cluster_names == [cluster.name]


def test_total_cost_computation():
    """
    Check that total_cost computes the expected cost for a single cluster
    usage.
    """
    cluster = Cluster(rpu=4, name="cluster_4rpu", cost_per_rpu_hour=0.5)
    blueprint = Blueprint([cluster])
    usage_s = 3600.0  # one hour
    # compute expected using the cluster's cost method to stay consistent
    expected = cluster.cost(duration_s=usage_s)
    total = blueprint.total_cost({cluster.name: usage_s})
    assert total == pytest.approx(expected)


def test_total_cost_unknown_cluster_raises():
    """
    Ensure total_cost raises KeyError when provided a usage map for an unknown
    cluster name.
    """
    cluster = Cluster(rpu=4, name="cluster_4rpu")
    blueprint = Blueprint([cluster])
    with pytest.raises(KeyError):
        blueprint.total_cost({"nonexistent-cluster": 100.0})


def test_simple_blueprints_up_to_32_rpu():
    """
    Verify simple_blueprints_up_to_32_rpu produces blueprints for each supported
    RPU size.
    """
    blueprints = Blueprint.simple_blueprints_up_to_32_rpu()
    expected_sizes = list(Cluster.UP_TO_32_RPU_SIZES)
    assert len(blueprints) == len(expected_sizes)
    # ensure each produced blueprint contains a single cluster with expected rpu
    actual_sizes = [bp.clusters[0].rpu for bp in blueprints]
    assert actual_sizes == expected_sizes

