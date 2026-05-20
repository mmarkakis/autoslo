from enum import Enum


class AutoscalingPolicy(Enum):
    NOOP = "noop"
    DUPLICATE_LARGEST = "duplicate_largest"
    ADD_SINGLE_BEST_FORWARD = "add_single_best_forward"
    REPLACE_WITH_SINGLE_BEST_FORWARD = "replace_with_single_best_forward"
