from enum import Enum


class AutoscalingTriggerPolicy(Enum):
    NOOP = "noop"
    QUEUE_DEPTH = "queue_depth"
    OBSERVED_VIOLATIONS = "observed_violations"
    PREDICTED_VIOLATIONS = "predicted_violations"
