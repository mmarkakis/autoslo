from typing import Optional

from psycopg2.pool import ThreadedConnectionPool

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster import Cluster


class Blueprint:
    """
    Represents a collection of clusters used for workload execution.

    Right now, blueprints are a thin wrapper around a single cluster, but they
    provide an abstraction layer that allows for future expansion to multiple
    clusters.
    """

    # Maps cluster names to their usage in seconds
    ClusterUsageMap = dict[str, float]

    def __init__(self, clusters: list[Cluster], name: str) -> None:
        """
        Initialize a Blueprint instance.

        Parameters:
            clusters: A list of Cluster instances that this blueprint contains.
            name: The name of the blueprint.

        Raises:
            ValueError: If the clusters list is empty.

        """
        if not clusters:
            raise ValueError("The clusters list cannot be empty.")
        self._clusters: dict[str, Cluster] = {}
        for cluster in clusters:
            self._clusters[cluster.name] = cluster
        self._name = name

    @staticmethod
    def from_config(blueprint_name: str) -> "Blueprint":
        """
        Read in a Blueprint instance from the configuration.

        Parameters:
            blueprint_name: The name of the blueprint to retrieve from the
            config.

        Returns:
            A Blueprint instance created from the configuration.

        Raises:
            KeyError: If the blueprint_name is not found in the configuration,
                or if the blueprint lacks a 'cluster_names' field.
        """
        blueprint_config = pu.get_blueprint_dicts_from_config()
        if blueprint_name not in blueprint_config:
            raise KeyError(
                f"Blueprint name '{blueprint_name}' not found in config."
            )
        if "cluster_names" not in blueprint_config[blueprint_name]:
            raise KeyError(
                f"Blueprint '{blueprint_name}' lacks a 'cluster_names' field."
            )
        cluster_names = blueprint_config[blueprint_name]["cluster_names"]
        clusters = [Cluster.from_config(name) for name in cluster_names]
        return Blueprint(clusters=clusters, name=blueprint_name)

    @staticmethod
    def one_cluster_with(cluster_rpu: float) -> "Blueprint":
        """
        Create a Blueprint instance with a single cluster of the specified RPU.

        Parameters:
            cluster_rpu: The RPU of the single cluster.

        Returns:
            A Blueprint instance with the specified configuration.
        """
        cluster = Cluster(
            rpu=int(cluster_rpu), name=f"cluster_{int(cluster_rpu)}"
        )
        return Blueprint(clusters=[cluster], name=f"single_{int(cluster_rpu)}")

    @property
    def name(self) -> str:
        """
        Get the name of the blueprint.

        Returns:
            The name of the blueprint.
        """
        return self._name

    @property
    def clusters(self) -> list[Cluster]:
        """
        Get all clusters of the blueprint.

        Returns:
            A list of Cluster instances in the clusters list.
        """
        return list(self._clusters.values())

    @property
    def cluster_names(self) -> list[str]:
        """
        Get the names of all clusters in the blueprint.

        Returns:
            A list of cluster names.
        """
        return list(self._clusters.keys())

    def total_cost(self, cluster_usage: ClusterUsageMap) -> float:
        """
        Calculate the total cost of running the blueprint based on cluster usage.

        Parameters:
            cluster_usage: A mapping of cluster names to their usage in seconds.

        Returns:
            The total cost of running the blueprint, in dollars.

        Raises:
            KeyError: If a cluster name in the cluster usage map does not match
                any cluster in the blueprint.
        """
        total_cost = 0.0
        for cluster_name, usage_s in cluster_usage.items():

            if cluster_name not in self._clusters:
                raise KeyError(
                    f"Cluster name '{cluster_name}' not found in blueprint."
                )
            total_cost += (
                self._clusters[cluster_name].cost_per_second * usage_s
            )

        return total_cost

    @staticmethod
    def simple_blueprints_up_to_32_rpu() -> list["Blueprint"]:
        """
        Generate simple blueprints with single clusters of sizes up to 32 RPU.

        Returns:
            A list of Blueprint instances, each containing a single Cluster with
            sizes 4, 8, 16, and 32 RPU.
        """
        blueprints = []
        for rpu in Cluster.UP_TO_32_RPU_SIZES:
            blueprint = Blueprint.one_cluster_with(cluster_rpu=rpu)
            blueprints.append(blueprint)
        return blueprints

    def conn_pool(
        self,
        cluster_name: str,
        minconn: int = 1,
        maxconn: int = 1000,
        search_path: str = "public",
    ) -> ThreadedConnectionPool:
        """
        Get a connection pool for a specific cluster in the blueprint.

        Parameters:
            cluster_name: The name of the cluster for which to get the
                connection pool.
            minconn: Minimum number of connections in the pool, if a new pool is
                created.
            maxconn: Maximum number of connections in the pool, if a new pool is
                created.
            search_path: The search path to set for the connections in the pool.

        Returns:
            A ThreadedConnectionPool instance for the specified cluster.

        Raises:
            KeyError: If the cluster_name is not found in the blueprint.
            ValueError: If no connection information has been provided for the
                specified cluster, in neither the conn_info parameter nor the
                cluster's own configuration.
        """
        if cluster_name not in self._clusters:
            raise KeyError(
                f"Cluster name '{cluster_name}' not found in blueprint."
            )
        cluster = self._clusters[cluster_name]
        return cluster.conn_pool(
            minconn=minconn, maxconn=maxconn, search_path=search_path
        )

    def conn_pool_map(
        self, minconn: int = 1, maxconn: int = 1000, search_path: str = "public"
    ) -> dict[str, ThreadedConnectionPool]:
        """
        Get a mapping of cluster names to their connection pools.

        Parameters:
            minconn: Minimum number of connections in each pool, if new pools
                are created.
            maxconn: Maximum number of connections in each pool, if new pools
                are created.
            search_path: The search path to set for the connections in each pool.

        Returns:
            A dictionary mapping cluster names to their ThreadedConnectionPool
            instances.

        Raises:
            ValueError: If no connection information has been provided for any
                cluster, in neither the conn_info parameter nor the cluster's
                own configuration.
        """
        conn_pool_map = {}
        for cluster_name, cluster in self._clusters.items():
            conn_pool_map[cluster_name] = cluster.conn_pool(
                minconn=minconn, maxconn=maxconn, search_path=search_path
            )
            assert conn_pool_map[cluster_name] is not None
        return conn_pool_map
