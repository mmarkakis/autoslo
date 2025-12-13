from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelPrediction:
    """
    A class representing a model prediction with mean and standard deviation.

    Attributes:
        mean_s: The mean runtime in seconds.
        std_s: The standard deviation of the runtime in seconds.
    """
    mean_s: float
    std_s: Optional[float] = None