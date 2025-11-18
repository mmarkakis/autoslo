from typing import Optional

from psycopg2.pool import ThreadedConnectionPool

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from autoslo.workload_execution.conn_utils import ConnWithSetup


class Cluster:
    """
    Represents a computing cluster with a specified number of RPU (Redshift
    Processing Units).
    """

    US_EAST_1_COST_PER_RPU_HOUR = 0.375
    ONE_HOUR_S = 3600

    UP_TO_32_RPU_SIZES = [4, 8, 16, 32]
    ALL_ALLOWED_RPU_SIZES = UP_TO_32_RPU_SIZES

    def __init__(
        self,
        rpu: int,
        name: str,
        conn_info: Optional[ClusterConnInfo] = None,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
    ) -> None:
        """
        Initialize a Cluster instance.

        Parameters:
            rpu: The number of RPU (Redshift Processing Units) for the cluster.
            name: The name of the cluster.
            cost_per_rpu_hour: The cost per RPU per hour. Defaults to the
                US_EAST_1_COST_PER_RPU_HOUR constant.
            conn_info: Optional ClusterConnInfo instance containing connection
                details for the cluster.
        """
        self._rpu = rpu
        self._name = name
        self._cost_per_rpu_hour = cost_per_rpu_hour
        self._conn_info = conn_info
        self._conn_pool: Optional[ThreadedConnectionPool] = None

    def __del__(self):
        """
        Destructor to ensure the connection pool is closed.
        """
        self.destroy_conn_pool()

    @staticmethod
    def from_config(cluster_name: str) -> "Cluster":
        """
        Read in a Cluster instance from the configuration.

        Parameters:
            cluster_name: The name of the cluster to retrieve from the config.

        Returns:
            A Cluster instance created from the configuration.

        Raises:
            KeyError: If the cluster_name is not found in the configuration.
        """
        cluster_config = pu.get_cluster_dicts_from_config()
        if cluster_name not in cluster_config:
            raise KeyError(
                f"Cluster name '{cluster_name}' not found in config."
            )
        return Cluster.from_dict(cluster_config[cluster_name])

    @staticmethod
    def from_dict(d: dict) -> "Cluster":
        """
        Create a Cluster instance from a dictionary.

        Parameters:
            d: A dictionary containing the cluster configuration. Expected keys
                are 'rpu', 'cluster_name' (optional), 'cost_per_rpu_hour'
                (optional), and 'conn_info' (optional).

        Returns:
            A Cluster instance created from the dictionary.
        """
        rpu = d["rpu"]
        name = d.get("cluster_name", None)
        cost_per_rpu_hour = d.get(
            "cost_per_rpu_hour", Cluster.US_EAST_1_COST_PER_RPU_HOUR
        )
        conn_info = ClusterConnInfo.from_dict(d)
        return Cluster(
            rpu=rpu,
            name=name,
            cost_per_rpu_hour=cost_per_rpu_hour,
            conn_info=conn_info,
        )

    def cost(self, duration_s: float = ONE_HOUR_S) -> float:
        """
        Calculate the cost of running the cluster for a given duration.

        Parameters:
            duration_s: The duration in seconds for which the cluster is run. If
                not provided, defaults to one hour.

        Returns:
            The total cost of running the cluster for the specified duration, in
                dollars.
        """
        hours = duration_s / self.ONE_HOUR_S
        total_cost = self.rpu * self.cost_per_rpu_hour * hours
        return total_cost

    @property
    def name(self) -> str:
        """
        Get the name of the cluster.

        Returns:
            The name of the cluster.
        """
        return self._name

    @property
    def rpu(self) -> int:
        """
        Get the RPU size of the cluster.

        Returns:
            The RPU size of the cluster.
        """
        return self._rpu

    @property
    def cost_per_rpu_hour(self) -> float:
        """
        Get the cost per RPU per hour for the cluster.

        Returns:
            The cost per RPU per hour.
        """
        return self._cost_per_rpu_hour

    @property
    def conn_info(self) -> Optional[ClusterConnInfo]:
        """
        Get the ClusterConnInfo instance for the cluster.

        Returns:
            The ClusterConnInfo instance if available, otherwise None.
        """
        return self._conn_info

    def conn_pool(
        self,
        minconn: int = 1,
        maxconn: int = 1000,
        search_path: str = "public",
    ) -> ThreadedConnectionPool:
        """
        Get a connection pool for the cluster. If none exists yet, create one.

        Parameters:
            minconn: Minimum number of connections in the pool.
            maxconn: Maximum number of connections in the pool.
            search_path: The search path to set for the connections in the pool.

        Returns:
            A connection pool object for the cluster.

        Raises:
            ValueError: If no connection information has been provided for the
                cluster in the constructor.
        """
        if self._conn_pool is None:
            if self.conn_info is None:
                raise ValueError(
                    "No connection information provided for cluster."
                )

            self._conn_pool = ThreadedConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                host=self.conn_info.host,
                port=self.conn_info.port,
                user=self.conn_info.user,
                password=self.conn_info.password,
                dbname=self.conn_info.dbname,
                connection_factory=lambda *args, **kwargs: ConnWithSetup(
                    *args,
                    search_path=search_path,
                    **kwargs,
                ),
            )
        return self._conn_pool

    def destroy_conn_pool(self) -> None:
        """
        Close and destroy the connection pool for the cluster, if it exists.
        """
        if self._conn_pool is not None:
            self._conn_pool.closeall()
            self._conn_pool = None
