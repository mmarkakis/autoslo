"""
cluster_provisioner.py
----------------------
Abstract interface and simulated implementation for cluster lifecycle.

The provisioner handles the *infra* side of spin-up / tear-down: actually
creating or destroying a cluster.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from autoslo.clusters.cluster import Cluster
from autoslo.config.component_configs import ProvisionerConfig
from autoslo.filesystem.logging import emit_structured
from autoslo.filesystem.structured_events import BaseStructuredEvent, EventType

logger = logging.getLogger(__name__)


class ClusterProvisioner(ABC):
    """Abstract interface for creating and destroying clusters."""

    @abstractmethod
    def __init__(self, config: ProvisionerConfig) -> None:
        """Initialize the provisioner with the given configuration.

        Parameters
        ----------
        config :
            The provisioner configuration.
        """

    @abstractmethod
    def spin_up(self, rpu: int, rel_time_s: float) -> Cluster:
        """Create a new cluster with the given RPU.

        Parameters
        ----------
        rpu :
            Redshift Processing Units for the new cluster.
        rel_time_s :
            Relative time in seconds since run start.

        Returns
        -------
        A ``Cluster`` instance representing the new cluster.
        For live provisioners, this method blocks until the cluster is
        available.  For simulated provisioners, it returns immediately.
        """

    @abstractmethod
    def tear_down(self, cluster_name: str, rel_time_s: float) -> None:
        """Destroy the named cluster.

        Parameters
        ----------
        cluster_name :
            Name of the cluster to destroy.
        rel_time_s :
            Relative time in seconds since run start.

        Raises
        ------
        KeyError if the cluster does not exist.
        """


class SimulatedProvisioner(ClusterProvisioner):
    """Provisioner for simulation: instant cluster creation via
    :meth:`Cluster.new`.

    The spin-up *delay* is not modelled here — it is the simulator's
    responsibility to insert a "cluster becomes ready" event at
    ``current_time + spin_up_delay_s`` and only add the cluster to the
    pool when that event fires.

    Parameters
    ----------
    spin_up_delay_s :
        The delay (in simulated seconds) between requesting a spin-up
        and the cluster becoming available.  Stored here so callers
        can query it; the provisioner itself does not sleep so that the
        simulator can proceed independently of wall clock time.
    """

    def __init__(self, config: ProvisionerConfig) -> None:
        self._config = config

    @property
    def spin_up_delay_s(self) -> float:
        """The configured spin-up delay (for the simulator to use)."""
        return self._config.spin_up_delay_s

    def spin_up(self, rpu: int, rel_time_s: float) -> Cluster:
        """Create a spec-only cluster instantly.

        Returns
        -------
        A new ``Cluster`` with auto-generated name and no connection info.
        """

        cluster = Cluster(
            rpu=rpu,
            creation_time_s=rel_time_s,
            cache_state=np.zeros(
                self._config.cluster_cache_state_dim, dtype=np.float32
            ),
        )
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.SPIN_UP_STARTED,
                source="SimulatedProvisioner",
                cluster_name=cluster.name,
            )
        )
        logger.debug(
            "SimulatedProvisioner: spun up %s (%d RPU) at time %.2f",
            cluster.name,
            rpu,
            rel_time_s,
        )
        return cluster

    def tear_down(self, cluster_name: str, rel_time_s: float) -> None:
        """Record a tear-down (no-op for simulation)."""
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.TEAR_DOWN_STARTED,
                source="SimulatedProvisioner",
                cluster_name=cluster_name,
            )
        )
        logger.debug(
            "SimulatedProvisioner: tore down %s at time %.2f",
            cluster_name,
            rel_time_s,
        )
