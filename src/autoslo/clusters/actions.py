from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingAction:
    reason: str


@dataclass(frozen=True)
class SpinUpAction(ScalingAction):
    rpu: int


@dataclass(frozen=True)
class TearDownAction(ScalingAction):
    cluster_name: str
