from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingAction:
    reason: str


@dataclass(frozen=True)
class SpinUpAction(ScalingAction):
    rpu: int
    from_reserved_budget: bool = False # Whether this spin-up should draw from 
    # the reserved budget (e.g. for capacity checkpoints) or the regular budget.


@dataclass(frozen=True)
class TearDownAction(ScalingAction):
    cluster_name: str
