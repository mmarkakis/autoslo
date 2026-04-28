from dataclasses import dataclass
from enum import Enum

import numpy as np

from autoslo.config._partial_config import PartialConfig


class QueryRouterPolicy(Enum):
    USE_ICONQ_MODEL = "use_iconq_model"
    ROUND_ROBIN = "round_robin"
    UNIFORM_RANDOM = "uniform_random"
    CACHE_AWARE = "cache_aware"


@dataclass(frozen=True)
class QueryRouterConfig(PartialConfig):
    rel_time_s_to_forecasted_table_vecs: dict[float, np.ndarray]

    routing_policy: QueryRouterPolicy = QueryRouterPolicy.USE_ICONQ_MODEL
    cluster_cache_state_update_alpha: float = 0.7

    cache_risk_cost_multiplier: float = 0
    # adj_cost = real_cost * (1 + multiplier * risk)
    cache_risk_coverage: float = 0.9
    cache_risk_epsilon: float = 1e-6

    def __post_init__(self):
        if not (0 <= self.cache_risk_cost_multiplier):
            raise ValueError(
                f"cache_risk_cost_multiplier must be non-negative, "
                f"got {self.cache_risk_cost_multiplier}"
            )
        if not (0 <= self.cache_risk_coverage <= 1):
            raise ValueError(
                f"cache_risk_coverage must be in [0, 1], "
                f"got {self.cache_risk_coverage}"
            )
        if not (0 < self.cache_risk_epsilon):
            raise ValueError(
                f"cache_risk_epsilon must be positive, "
                f"got {self.cache_risk_epsilon}"
            )
