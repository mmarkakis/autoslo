"""
cluster_pool.py
---------------
Mutable cluster collection for dynamic provisioning.

Unlike :class:`~autoslo.blueprints.blueprint.Blueprint` (which is an
immutable, named set of pre-configured clusters), ``ClusterPool`` supports
adding and removing clusters at runtime — exactly what the capacity
controller and simulator need for dynamic spin-up / tear-down.
"""

from __future__ import annotations

import threading
from typing import Optional

from psycopg2.pool import ThreadedConnectionPool

from autoslo.blueprints.cluster import Cluster
from autoslo.blueprints.blueprint import Blueprint  


class ClusterPool:
    """A dynamic, mutable collection of clusters.

    Thread-safe: all mutations and reads are protected by an internal lock
    so that the capacity controller (which may spin clusters up/down from
    a background thread) can coexist with the router (which reads the
    cluster set on query-arrival threads).

    Parameters
    ----------
    initial_clusters :
        Clusters that are ready from the start.  May be *None* or empty.
    allowed_rpu_sizes :
        RPU sizes that the capacity controller is allowed to spin up.
        Defaults to ``Cluster.ALL_ALLOWED_RPU_SIZES`` (``[4, 8, 16, 32]``).
    """

    def __init__(
        self,
        initial_clusters: list[Cluster] | None = None,
        *,
        initial_rpus: list[int] | None = None,
        allowed_rpu_sizes: list[int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._clusters: dict[str, Cluster] = {}
        if initial_clusters:
            for c in initial_clusters:
                self._clusters[c.name] = c
        if initial_rpus:
            for rpu in initial_rpus:
                c = Cluster.new(rpu=rpu)
                self._clusters[c.name] = c
        self._allowed_rpu_sizes: list[int] = sorted(
            allowed_rpu_sizes
            if allowed_rpu_sizes is not None
            else Cluster.ALL_ALLOWED_RPU_SIZES
        )

    # ------------------------------------------------------------------
    # Cluster lifecycle
    # ------------------------------------------------------------------

    def add_cluster(self, cluster: Cluster) -> None:
        """Add a cluster to the pool.

        Raises
        ------
        ValueError
            If a cluster with the same name already exists.
        """
        with self._lock:
            if cluster.name in self._clusters:
                raise ValueError(
                    f"Cluster {cluster.name!r} already in pool."
                )
            self._clusters[cluster.name] = cluster

    def remove_cluster(self, cluster_name: str) -> Cluster:
        """Remove and return a cluster from the pool.

        Raises
        ------
        KeyError
            If the cluster name is not in the pool.
        """
        with self._lock:
            if cluster_name not in self._clusters:
                raise KeyError(
                    f"Cluster {cluster_name!r} not in pool."
                )
            return self._clusters.pop(cluster_name)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    @property
    def cluster_names(self) -> list[str]:
        """Sorted list of current cluster names."""
        with self._lock:
            return sorted(self._clusters.keys())

    @property
    def clusters(self) -> list[Cluster]:
        """List of current clusters (sorted by name)."""
        with self._lock:
            return [self._clusters[n] for n in sorted(self._clusters.keys())]

    def get_cluster(self, name: str) -> Cluster:
        """Look up a cluster by name.

        Raises
        ------
        KeyError
            If the cluster name is not in the pool.
        """
        with self._lock:
            return self._clusters[name]

    def get_cost_per_second(self, name: str) -> float:
        """Return the cost-per-second for the named cluster."""
        return self.get_cluster(name).cost_per_second

    @property
    def allowed_rpu_sizes(self) -> list[int]:
        """RPU sizes available for dynamic spin-up (sorted ascending)."""
        return list(self._allowed_rpu_sizes)

    def __len__(self) -> int:
        with self._lock:
            return len(self._clusters)

    def __contains__(self, cluster_name: str) -> bool:
        with self._lock:
            return cluster_name in self._clusters

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_blueprint(
        cls,
        blueprint: Blueprint,
        *,
        allowed_rpu_sizes: list[int] | None = None,
    ) -> ClusterPool:
        """Create a ``ClusterPool`` from an existing
        :class:`~autoslo.blueprints.blueprint.Blueprint`.

        This bridges the static (Blueprint) and dynamic (ClusterPool)
        worlds — useful when you want to start with a fixed set but
        allow the capacity controller to spin up additional clusters.
        """
        return cls(
            initial_clusters=list(blueprint.clusters),
            allowed_rpu_sizes=allowed_rpu_sizes,
        )

    @classmethod
    def from_rpu_list(
        cls,
        rpus: list[int],
        *,
        allowed_rpu_sizes: list[int] | None = None,
    ) -> ClusterPool:
        """Create a ``ClusterPool`` with one :meth:`Cluster.new()
        <autoslo.blueprints.cluster.Cluster.new>` per RPU value.

        Useful for simulation scenarios where clusters don't need config
        entries or connection info.
        """
        clusters = [Cluster.new(rpu=rpu) for rpu in rpus]
        return cls(
            initial_clusters=clusters,
            allowed_rpu_sizes=allowed_rpu_sizes,
        )

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------

    def total_cost(self, usage_s: dict[str, float]) -> float:
        """Compute total cost given per-cluster usage in seconds.

        Parameters
        ----------
        usage_s :
            ``{cluster_name: total_seconds_active}``

        Returns
        -------
        Total dollar cost across all clusters in *usage_s*.
        """
        total = 0.0
        with self._lock:
            for cn, secs in usage_s.items():
                if cn in self._clusters:
                    total += self._clusters[cn].cost_per_second * secs
        return total

    # ------------------------------------------------------------------
    # Live-execution helpers
    # ------------------------------------------------------------------

    def conn_pool_map(
        self,
        minconn: int = 1,
        maxconn: int = 1000,
        search_path: str = "public",
    ) -> dict[str, ThreadedConnectionPool]:
        """Return ``{cluster_name: ThreadedConnectionPool}`` for clusters
        that have connection info.

        Clusters without ``conn_info`` (e.g. simulated clusters) are
        silently skipped.
        """
        result: dict[str, ThreadedConnectionPool] = {}
        with self._lock:
            for cn, cluster in self._clusters.items():
                if cluster.conn_info is not None:
                    result[cn] = cluster.conn_pool(
                        minconn=minconn,
                        maxconn=maxconn,
                        search_path=search_path,
                    )
        return result
