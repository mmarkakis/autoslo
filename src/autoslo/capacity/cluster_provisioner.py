"""
cluster_provisioner.py
----------------------
Abstract interface and simulated implementation for cluster lifecycle.

The provisioner handles the *infra* side of spin-up / tear-down: actually
creating or destroying a cluster.  The *decision* of when to spin up or
tear down lives in :class:`~autoslo.capacity.capacity_controller.CapacityController`.

Two concrete implementations:

* :class:`SimulatedProvisioner` — for the simulator.  Creates lightweight
  ``Cluster.new()`` objects instantly (the spin-up *delay* is modelled by
  the simulator's event loop, not by the provisioner itself).
* :class:`~autoslo.capacity.redshift_provisioner.RedshiftServerlessProvisioner`
  — for live execution.  Wraps AWS API calls (separate module).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from autoslo.blueprints.cluster import Cluster

logger = logging.getLogger(__name__)


class ClusterProvisioner(ABC):
    """Abstract interface for creating and destroying clusters."""

    @abstractmethod
    def spin_up(self, rpu: int, current_time_s: float) -> Cluster:
        """Create a new cluster with the given RPU.

        Parameters
        ----------
        rpu :
            Redshift Processing Units for the new cluster.
        current_time_s :
            The current time for bookkeeping.

        Returns
        -------
        A ``Cluster`` instance representing the new cluster.
        For live provisioners, this method blocks until the cluster is
        available.  For simulated provisioners, it returns immediately.
        """

    @abstractmethod
    def tear_down(self, cluster_name: str, current_time_s: float) -> None:
        """Destroy the named cluster.

        Parameters
        ----------
        cluster_name :
            Name of the cluster to destroy.
        current_time_s :
            The current time for bookkeeping.

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

    def __init__(self, spin_up_delay_s: float = 120.0) -> None:
        self._spin_up_delay_s = spin_up_delay_s
        self._spun_up: list[tuple[Cluster, float]] = []
        self._torn_down: list[tuple[str, float]] = []

    @property
    def spin_up_delay_s(self) -> float:
        """The configured spin-up delay (for the simulator to use)."""
        return self._spin_up_delay_s

    @property
    def spun_up(self) -> list[tuple[Cluster, float]]:
        """Clusters created so far (for test inspection)."""
        return list(self._spun_up)

    @property
    def torn_down(self) -> list[tuple[str, float]]:
        """Cluster names torn down so far (for test inspection)."""
        return list(self._torn_down)

    def spin_up(self, rpu: int, current_time_s: float) -> Cluster:
        """Create a spec-only cluster instantly.

        Returns
        -------
        A new ``Cluster`` with auto-generated name and no connection info.
        """
        cluster = Cluster.new(rpu=rpu)
        self._spun_up.append((cluster, current_time_s))
        logger.info(
            "SimulatedProvisioner: spun up %s (%d RPU) at time %.2f",
            cluster.name,
            rpu,
            current_time_s,
        )
        return cluster

    def tear_down(self, cluster_name: str, current_time_s: float) -> None:
        """Record a tear-down (no-op for simulation)."""
        self._torn_down.append((cluster_name, current_time_s))
        logger.info(
            "SimulatedProvisioner: tore down %s at time %.2f",
            cluster_name,
            current_time_s,
        )
