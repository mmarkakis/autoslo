from enum import Enum


class AutoscalingPolicy(Enum):
    ADD_SINGLE_BEST = "add_single_best"
    NOOP = "noop"
    DUPLICATE_LARGEST = "duplicate_largest"
    REPLACE_WITH_SINGLE_BEST = "replace_with_single_best"
