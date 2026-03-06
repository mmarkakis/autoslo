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
PolicyTuner                 Simulation-based Pareto policy sweep.
PolicyParams                Frozen policy-parameter pair.
SweepResult / SweepEntry    Sweep outputs.
ScenarioOutcome             Per-scenario cost/violation summary.
compute_pareto_front        Standalone O(N log N) Pareto extraction.
plot_pareto_front           Matplotlib scatter of the Pareto front.
print_pareto_summary        Rich / plain-text table of Pareto-optimal entries.
VALID_OBJECTIVES            Recognised objective names.
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
    "Autoscaler",
    "AutoscalingAction",
    "AutoscalingPolicy",
    "ClusterProvisioner",
    "HeadroomPolicy",
    "NoOpPolicy",
    "PolicyParams",
    "PolicyTuner",
    "ScenarioOutcome",
    "SimulatedProvisioner",
    "SpinUpRequest",
    "SweepEntry",
    "SweepResult",
    "TearDownRequest",
    "VALID_OBJECTIVES",
    "compute_pareto_front",
    "plot_pareto_front",
    "print_pareto_summary",
]
