from dataclasses import dataclass
from autoslo.config._partial_config import PartialConfig


@dataclass(frozen=True)
class AutoscalerConfig(PartialConfig):
    """
    Configuration for the autoscaler, including the target SLO and scaling
    parameters.
    """

    allowed_rpu_sizes: list[int]
    min_cluster_lifetime_s: float = 1200.0
    idle_time_before_tear_down_s: float = 300.0
    observation_window_s: float = 120.0
    min_observations_to_act: int = 5
    slo_tightening_factor: float = 1.0
