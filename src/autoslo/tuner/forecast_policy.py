from enum import Enum


class ForecastPolicy(Enum):
    GROUND_TRUTH = "ground_truth"  # For convenience; implemented higher up.
    ONE_DAY = "one_day"
    SEVEN_DAYS_FLAT = "seven_days_flat"
    SAME_DAY_ONCE = "same_day_once"
    SAME_DAY_EXPONENTIAL = "same_day_exponential"
