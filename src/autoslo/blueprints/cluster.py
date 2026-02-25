import itertools
from typing import Optional

from psycopg2.pool import ThreadedConnectionPool

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from autoslo.workload_execution.conn_utils import ConnWithSetup
from collections import defaultdict

from datetime import datetime


class Cluster:
    """
    Represents a computing cluster with a specified number of RPU (Redshift
    Processing Units).
    """

    US_EAST_1_COST_PER_RPU_HOUR = 0.375
    ONE_HOUR_S = 3600

    UP_TO_32_RPU_SIZES = [4, 8, 16, 32]
    ALL_ALLOWED_RPU_SIZES = UP_TO_32_RPU_SIZES

    _all_cluster_configs = None
    _new_counter = itertools.count()
    _ordered_cluster_names_per_rpu: dict[int, list[str]] | None = None
    _rpu_per_cluster_name: dict[str, int] | None = None

    @staticmethod
    def all_allowed_rpu_sizes() -> list[int]:
        return Cluster.ALL_ALLOWED_RPU_SIZES

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

    def __eq__(self, other: object) -> bool:
        """
        Check equality between this Cluster and another object.

        Parameters:
            other: The object to compare with.

        Returns:
            True if the other object is a Cluster with the same RPU, name,
            cost per RPU hour, and connection info, False otherwise.
        """
        if not isinstance(other, Cluster):
            return False
        if self._rpu != other._rpu:
            return False
        if self._name != other._name:
            return False
        if self._cost_per_rpu_hour != other._cost_per_rpu_hour:
            return False
        if self._conn_info != other._conn_info:
            return False
        return True

    def __del__(self):
        """
        Destructor to ensure the connection pool is closed.
        """
        self.destroy_conn_pool()

    @classmethod
    def all_cluster_configs(cls) -> dict[str, dict]:
        """
        Retrieve all cluster configurations from the configuration file.

        Returns:
            A dictionary mapping cluster names to their configuration
            dictionaries.
        """
        if cls._all_cluster_configs is None:
            cls._all_cluster_configs = pu.get_cluster_dicts_from_config()
        return cls._all_cluster_configs

    @classmethod
    def all_cluster_names(cls) -> list[str]:
        """
        Retrieve all cluster names from the configuration file.

        Returns:
            A list of all cluster names.
        """
        return list(cls.all_cluster_configs().keys())

    @classmethod
    def ordered_cluster_names_per_rpu(cls) -> dict[int, list[str]]:
        """
        Retrieve the cluster_names for each RPU size in a consistent order.

        Returns:
            A list of cluster names that have the specified RPU size.
        """
        if cls._ordered_cluster_names_per_rpu is None:
            d = defaultdict(list)
            for cluster_name, config in cls.all_cluster_configs().items():
                rpu = config["rpu"]
                d[rpu].append(cluster_name)
            cls._ordered_cluster_names_per_rpu = {rpu: sorted(cluster_names) for rpu, cluster_names in d.items()}
        return cls._ordered_cluster_names_per_rpu
    
    @classmethod
    def rpu_for_cluster_name(cls, cluster_name: str) -> int:
        """
        Retrieve the RPU size for a given cluster name.

        Parameters:
            cluster_name: The name of the cluster.

        Returns:
            The RPU size of the specified cluster.

        Raises:
            KeyError: If the cluster_name is not found in the configuration.
        """
        if cls._rpu_per_cluster_name is None:
            cls._rpu_per_cluster_name = {
                cluster_name: config["rpu"]
                for cluster_name, config in cls.all_cluster_configs().items()
            }
        if cluster_name not in cls._rpu_per_cluster_name:
            raise KeyError(
                f"Cluster name '{cluster_name}' not found in config."
            )
        return cls._rpu_per_cluster_name[cluster_name]

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
        cluster_configs = Cluster.all_cluster_configs()
        if cluster_name not in cluster_configs:
            raise KeyError(
                f"Cluster name '{cluster_name}' not found in config."
            )
        return Cluster.from_dict(cluster_configs[cluster_name])

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

    @staticmethod
    def new(
        rpu: int,
        name: Optional[str] = None,
        cost_per_rpu_hour: float = US_EAST_1_COST_PER_RPU_HOUR,
    ) -> "Cluster":
        """
        Create a spec-only cluster with no config lookup.

        This is the factory for dynamically provisioned clusters (used by
        the simulator and the capacity controller).  No connection info is
        attached — call :meth:`attach_conn_info` after the AWS workgroup
        is created.

        Parameters
        ----------
        rpu:
            The number of Redshift Processing Units for the cluster.
        name:
            Optional explicit name.  If *None*, auto-generated as
            ``"cluster_{rpu}_{unix_timestamp}"``.
        cost_per_rpu_hour:
            Cost per RPU per hour (defaults to US-East-1 pricing).

        Returns
        -------
        A new ``Cluster`` instance with no connection info.
        """
        if name is None:
            seq = next(Cluster._new_counter)
            name = f"cluster_{rpu}_{int(datetime.now().timestamp())}_{seq}"
        return Cluster(
            rpu=rpu,
            name=name,
            conn_info=None,
            cost_per_rpu_hour=cost_per_rpu_hour,
        )

    def attach_conn_info(self, conn_info: ClusterConnInfo) -> None:
        """Attach connection information after live provisioning.

        Parameters
        ----------
        conn_info:
            The connection info obtained after the AWS workgroup becomes
            available.

        Raises
        ------
        ValueError
            If connection info has already been set.
        """
        if self._conn_info is not None:
            raise ValueError(
                f"Cluster {self._name!r} already has connection info."
            )
        self._conn_info = conn_info

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
    def cost_per_second(self) -> float:
        """
        Get the cost per second for the cluster.

        Returns:
            The cost per second.
        """
        return self._cost_per_rpu_hour * self._rpu / Cluster.ONE_HOUR_S

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
