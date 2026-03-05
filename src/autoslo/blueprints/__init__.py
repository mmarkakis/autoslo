"""
autoslo.blueprints
==================
Cluster abstractions.

Public API
----------
Cluster         Lightweight descriptor for a compute cluster (frozen dataclass).
ClusterConnInfo Connection details for a cluster.
"""

from autoslo.blueprints.cluster import Cluster

__all__ = [
    "Cluster",
]
