from typing import Optional

import numpy as np


class ClusterCacheStateUpdater:
    def __init__(self, state_dim: int, alpha: float = 0.7):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.state_dim = state_dim
        self.alpha = alpha

    def update(
        self,
        current_state: Optional[np.ndarray],
        table_vector: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if current_state is None or table_vector is None:
            return None
        return self.alpha * current_state + (1 - self.alpha) * table_vector
