"""
autoslo.blueprints
==================
Cluster and blueprint abstractions.

Public API
----------
Cluster         Represents a computing cluster (config-based or spec-based).
ClusterPool     Mutable cluster collection for dynamic provisioning.
Blueprint       Immutable, named set of pre-configured clusters.
ClusterConnInfo Connection details for a cluster.
"""

from autoslo.blueprints.cluster import Cluster
from autoslo.blueprints.cluster_pool import ClusterPool

__all__ = [
    "Cluster",
    "ClusterPool",
]
