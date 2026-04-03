"""
autoslo.capacity
================
Capacity management: online autoscaling, offline policy tuning, and
cluster provisioning.

Public API
----------
Autoscaler                  Thin coordinator that delegates to a policy.
AutoscalingPolicy           ABC for pluggable autoscaling strategies.
AutoscalingAction           Dataclass returned by policy event handlers.
SpinUpRequest / TearDownRequest  Individual scaling directives.
NoOpPolicy                  Policy that never scales.
HeadroomPolicy              SLO-headroom-based policy (default).
ClusterProvisioner          ABC for cluster lifecycle operations.
SimulatedProvisioner        Instant provisioner for simulation.
RedshiftServerlessProvisioner   Live AWS provisioner.
"""

from autoslo.capacity.autoscaler import Autoscaler
from autoslo.capacity.autoscaling_policy import (
    AutoscalingAction,
    AutoscalingPolicy,
    NoOpPolicy,
    SpinUpRequest,
    TearDownRequest,
)
from autoslo.capacity.cluster_provisioner import (
    ClusterProvisioner,
    SimulatedProvisioner,
)
from autoslo.capacity.headroom_policy import HeadroomPolicy
from autoslo.capacity.redshift_provisioner import (
    RedshiftServerlessProvisioner,
)


__all__ = [
    "Autoscaler",
    "AutoscalingAction",
    "AutoscalingPolicy",
    "ClusterProvisioner",
    "HeadroomPolicy",
    "NoOpPolicy",
    "RedshiftServerlessProvisioner",
    "SimulatedProvisioner",
    "SpinUpRequest",
    "TearDownRequest",
]
