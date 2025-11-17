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

    def __init__(self, clusters: list[Cluster]) -> None:
        """
        Initialize a Blueprint instance.

        Parameters:
            clusters: A list of Cluster instances that this blueprint contains.

        Raises:
            ValueError: If the clusters list is empty.
            ValueError: (Temporary) If the clusters list contains more than one
            cluster.

        """
        if not clusters:
            raise ValueError("The clusters list cannot be empty.")
        if len(clusters) > 1:
            raise ValueError("Currently, only a single cluster is supported.")
        self._clusters = {}
        for cluster in clusters:
            self._clusters[cluster.name] = cluster

    @staticmethod
    def one_cluster_with(cluster_rpu: float) -> "Blueprint":
        """
        Create a Blueprint instance with a single cluster of the specified RPU.

        Parameters:
            cluster_rpu: The RPU of the single cluster.

        Returns:
            A Blueprint instance with the specified configuration.
        """
        cluster = Cluster(rpu=int(cluster_rpu))
        return Blueprint(clusters=[cluster])

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
            total_cost += self._clusters[cluster_name].cost(duration_s=usage_s)

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
            cluster = Cluster(rpu=rpu)
            blueprint = Blueprint(clusters=[cluster])
            blueprints.append(blueprint)
        return blueprints
