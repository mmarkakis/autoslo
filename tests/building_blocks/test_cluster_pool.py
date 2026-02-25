"""
Tests for :mod:`autoslo.blueprints.cluster_pool`.
"""

from __future__ import annotations

import threading

import pytest

from autoslo.blueprints.cluster import Cluster
from autoslo.blueprints.cluster_pool import ClusterPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cluster(rpu: int = 8, name: str = "c0") -> Cluster:
    """Create a lightweight spec-only cluster."""
    return Cluster(rpu=rpu, name=name)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestClusterPoolConstruction:

    def test_empty_pool(self):
        pool = ClusterPool()
        assert len(pool) == 0
        assert pool.cluster_names == []

    def test_initial_clusters(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "a"), _cluster(16, "b")])
        assert len(pool) == 2
        assert "a" in pool
        assert "b" in pool
        assert pool.cluster_names == ["a", "b"]

    def test_default_allowed_rpu_sizes(self):
        pool = ClusterPool()
        assert pool.allowed_rpu_sizes == Cluster.ALL_ALLOWED_RPU_SIZES

    def test_custom_allowed_rpu_sizes(self):
        pool = ClusterPool(allowed_rpu_sizes=[32, 8, 16])
        assert pool.allowed_rpu_sizes == [8, 16, 32]  # sorted


# ---------------------------------------------------------------------------
# Cluster lifecycle
# ---------------------------------------------------------------------------


class TestClusterPoolLifecycle:

    def test_add_cluster(self):
        pool = ClusterPool()
        pool.add_cluster(_cluster(8, "c0"))
        assert len(pool) == 1
        assert "c0" in pool
        assert pool.get_cluster("c0").rpu == 8

    def test_add_duplicate_raises(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "c0")])
        with pytest.raises(ValueError, match="already in pool"):
            pool.add_cluster(_cluster(16, "c0"))

    def test_remove_cluster(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "c0"), _cluster(16, "c1")])
        removed = pool.remove_cluster("c0")
        assert removed.name == "c0"
        assert len(pool) == 1
        assert "c0" not in pool

    def test_remove_nonexistent_raises(self):
        pool = ClusterPool()
        with pytest.raises(KeyError, match="not in pool"):
            pool.remove_cluster("nonexistent")

    def test_add_after_remove(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "c0")])
        pool.remove_cluster("c0")
        pool.add_cluster(_cluster(32, "c0"))
        assert pool.get_cluster("c0").rpu == 32


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------


class TestClusterPoolQueryInterface:

    def test_cluster_names_sorted(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "z"), _cluster(16, "a")])
        assert pool.cluster_names == ["a", "z"]

    def test_clusters_sorted_by_name(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "z"), _cluster(16, "a")])
        names = [c.name for c in pool.clusters]
        assert names == ["a", "z"]

    def test_get_cluster(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "c0")])
        c = pool.get_cluster("c0")
        assert c.rpu == 8

    def test_get_cluster_missing_raises(self):
        pool = ClusterPool()
        with pytest.raises(KeyError):
            pool.get_cluster("missing")

    def test_get_cost_per_second(self):
        c = _cluster(8, "c0")
        pool = ClusterPool(initial_clusters=[c])
        assert pool.get_cost_per_second("c0") == c.cost_per_second

    def test_contains(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "c0")])
        assert "c0" in pool
        assert "c1" not in pool


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


class TestClusterPoolFactories:

    def test_from_rpu_list(self):
        pool = ClusterPool.from_rpu_list([8, 16, 32])
        assert len(pool) == 3
        rpus = sorted(c.rpu for c in pool.clusters)
        assert rpus == [8, 16, 32]

    def test_from_rpu_list_names_contain_rpu(self):
        pool = ClusterPool.from_rpu_list([8])
        c = pool.clusters[0]
        assert "8" in c.name

    def test_from_rpu_list_custom_allowed(self):
        pool = ClusterPool.from_rpu_list([8], allowed_rpu_sizes=[4, 8])
        assert pool.allowed_rpu_sizes == [4, 8]


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


class TestClusterPoolCost:

    def test_total_cost(self):
        c0 = _cluster(8, "c0")
        c1 = _cluster(16, "c1")
        pool = ClusterPool(initial_clusters=[c0, c1])

        usage = {"c0": 3600.0, "c1": 1800.0}
        cost = pool.total_cost(usage)

        expected = c0.cost_per_second * 3600.0 + c1.cost_per_second * 1800.0
        assert cost == pytest.approx(expected)

    def test_total_cost_ignores_unknown_clusters(self):
        pool = ClusterPool(initial_clusters=[_cluster(8, "c0")])
        # "unknown" is in usage but not in pool — silently skipped
        cost = pool.total_cost({"c0": 100.0, "unknown": 999.0})
        assert cost == pytest.approx(_cluster(8, "c0").cost_per_second * 100.0)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestClusterPoolThreadSafety:

    def test_concurrent_add_remove(self):
        """Concurrent add/remove operations don't corrupt internal state."""
        pool = ClusterPool()
        errors: list[Exception] = []

        def adder():
            try:
                for i in range(50):
                    name = f"add-{threading.current_thread().name}-{i}"
                    pool.add_cluster(_cluster(8, name))
            except Exception as e:
                errors.append(e)

        def remover():
            try:
                import time
                time.sleep(0.01)  # let adders get ahead
                for i in range(50):
                    name = f"add-{threading.current_thread().name}-{i}"
                    # may or may not exist — ignore errors
                    try:
                        pool.remove_cluster(name)
                    except KeyError:
                        pass
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=adder, name=f"t{i}")
            for i in range(4)
        ]
        threads.extend(
            threading.Thread(target=remover, name=f"t{i}")
            for i in range(4)
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # Pool should be in a consistent state — len should match
        # number of keys.
        assert len(pool) == len(pool.cluster_names)
