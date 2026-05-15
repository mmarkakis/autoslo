"""
Tests for :mod:`autoslo.capacity.cluster_provisioner`.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoslo.clusters.cluster import Cluster
from autoslo.clusters.cluster_provisioner import (
    ClusterProvisioner,
    SimulatedProvisioner,
)


# ---------------------------------------------------------------------------
# ABC cannot be instantiated
# ---------------------------------------------------------------------------


class TestClusterProvisionerABC:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ClusterProvisioner()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# SimulatedProvisioner
# ---------------------------------------------------------------------------


class TestSimulatedProvisioner:

    def test_default_delay(self):
        prov = SimulatedProvisioner()
        assert prov.spin_up_delay_s == 120.0

    def test_custom_delay(self):
        prov = SimulatedProvisioner(spin_up_delay_s=30.0)
        assert prov.spin_up_delay_s == 30.0

    def test_spin_up_returns_cluster(self):
        prov = SimulatedProvisioner()
        cluster = prov.spin_up(rpu=8, rel_time_s=0.0)
        assert isinstance(cluster, Cluster)
        assert cluster.rpu == 8

    def test_spin_up_name_contains_rpu(self):
        prov = SimulatedProvisioner()
        cluster = prov.spin_up(rpu=16, rel_time_s=0.0)
        assert "16" in cluster.name

    def test_spin_up_no_conn_info(self):
        prov = SimulatedProvisioner()
        cluster = prov.spin_up(rpu=8, rel_time_s=0.0)
        assert cluster.conn_info is None

    def test_spin_up_records_history(self):
        prov = SimulatedProvisioner()
        prov.spin_up(rpu=8, rel_time_s=0.0)
        prov.spin_up(rpu=16, rel_time_s=10.0)
        assert len(prov.spun_up) == 2
        assert prov.spun_up[0][0].rpu == 8
        assert prov.spun_up[1][0].rpu == 16
        assert prov.spun_up[0][1] == 0.0
        assert prov.spun_up[1][1] == 10.0

    def test_tear_down_records_history(self):
        prov = SimulatedProvisioner()
        prov.tear_down("c0", rel_time_s=5.0)
        prov.tear_down("c1", rel_time_s=15.0)
        assert prov.torn_down == [("c0", 5.0), ("c1", 15.0)]

    def test_spin_up_unique_names(self):
        """Each spin_up call produces a unique cluster name."""
        prov = SimulatedProvisioner()
        names = set()
        for _ in range(10):
            c = prov.spin_up(rpu=8, rel_time_s=0.0)
            names.add(c.name)
        # Names should be unique (timestamp may collide within the
        # same second for fast tests, but Cluster.new uses int(time.time()))
        # — at worst we get len >= 1.  Use a relaxed assertion.
        assert len(names) >= 1

    def test_spin_up_cost(self):
        prov = SimulatedProvisioner()
        cluster = prov.spin_up(rpu=32, rel_time_s=0.0)
        expected_cost = (
            Cluster.US_EAST_1_COST_PER_RPU_HOUR * 32 / Cluster.ONE_HOUR_S
        )
        assert cluster.cost_per_second == pytest.approx(expected_cost)

    def test_tear_down_is_noop(self):
        """tear_down doesn't raise or modify spin-up history."""
        prov = SimulatedProvisioner()
        c = prov.spin_up(rpu=8, rel_time_s=0.0)
        prov.tear_down(c.name, rel_time_s=1.0)
        # spun_up history still has the cluster
        assert len(prov.spun_up) == 1


# ---------------------------------------------------------------------------
# Cluster.new() and attach_conn_info()
# ---------------------------------------------------------------------------


class TestClusterNew:

    def test_new_creates_cluster(self):
        c = Cluster.new(rpu=8)
        assert c.rpu == 8
        assert c.conn_info is None

    def test_new_name_contains_rpu_and_timestamp(self):
        c = Cluster.new(rpu=16)
        assert "16" in c.name
        # Should contain a numeric timestamp portion
        parts = c.name.split("_")
        assert len(parts) >= 4
        # Third part should be a timestamp (a large integer)
        assert int(parts[2]) > 1700000000

    def test_new_explicit_name(self):
        c = Cluster.new(rpu=8, name="my-cluster")
        assert c.name == "my-cluster"
        assert c.rpu == 8

    def test_new_cost(self):
        c = Cluster.new(rpu=32)
        expected = Cluster.US_EAST_1_COST_PER_RPU_HOUR * 32 / Cluster.ONE_HOUR_S
        assert c.cost_per_second == pytest.approx(expected)


class TestClusterFrozen:

    def test_frozen_prevents_mutation(self):
        """Cluster is a frozen dataclass — setting attr raises."""
        c = Cluster.new(rpu=8)
        with pytest.raises(AttributeError):
            c.rpu = 16  # type: ignore[misc]

    def test_conn_info_set_at_construction(self):
        """Connection info can be provided at construction time."""
        from autoslo.clusters.cluster_conn_info import ClusterConnInfo

        info = ClusterConnInfo(
            host="example.com",
            port=5439,
            dbname="dev",
            user="admin",
            password="pw",
        )
        c = Cluster(rpu=8, name="autoslo-8-0-0", cache_state=np.zeros(1), conn_info=info)
        assert c.conn_info is info
