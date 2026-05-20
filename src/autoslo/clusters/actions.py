from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingAction:
    reason: str


@dataclass(frozen=True)
class SpinUpAction(ScalingAction):
    rpu: int
    from_reserved_budget: bool = False  # Whether this spin-up should draw from
    # the reserved budget (e.g. for scheduled spin-ups) or the regular budget.
    deferred_teardowns: tuple[str, ...] = ()  # Cluster names to tear down once
    # this cluster becomes READY (used by REPLACE_WITH_SINGLE_BEST_FORWARD policy).


@dataclass(frozen=True)
class TearDownAction(ScalingAction):
    cluster_name: str
