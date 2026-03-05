"""
autoslo.capacity
================
Capacity management: online autoscaling, offline policy tuning, and
cluster provisioning.

Public API
----------
CapacityController          Background autoscaler (spin-up / tear-down).
ClusterProvisioner          ABC for cluster lifecycle operations.
SimulatedProvisioner        Instant provisioner for simulation.
RedshiftServerlessProvisioner   Live AWS provisioner.
PolicyTuner                 Simulation-based Pareto policy sweep.
PolicyParams                Frozen policy-parameter pair.
SweepResult / SweepEntry    Sweep outputs.
ScenarioOutcome             Per-scenario cost/violation summary.
compute_pareto_front        Standalone O(N log N) Pareto extraction.
plot_pareto_front           Matplotlib scatter of the Pareto front.
print_pareto_summary        Rich / plain-text table of Pareto-optimal entries.
VALID_OBJECTIVES            Recognised objective names.
"""

from autoslo.capacity.capacity_controller import CapacityController
from autoslo.capacity.cluster_provisioner import (
    ClusterProvisioner,
    SimulatedProvisioner,
)
from autoslo.capacity.policy_tuner import (
    VALID_OBJECTIVES,
    PolicyParams,
    PolicyTuner,
    ScenarioOutcome,
    SweepEntry,
    SweepResult,
    compute_pareto_front,
    plot_pareto_front,
    print_pareto_summary,
)

__all__ = [
    "CapacityController",
    "ClusterProvisioner",
    "PolicyParams",
    "PolicyTuner",
    "ScenarioOutcome",
    "SimulatedProvisioner",
    "SweepEntry",
    "SweepResult",
    "VALID_OBJECTIVES",
    "compute_pareto_front",
    "plot_pareto_front",
    "print_pareto_summary",
]
