"""lstm_state.py
---------------
Shared LSTM-state dataclass used by two separate layers:

  * ``autoslo.models.iconq_model`` — produces and consumes states (computation).
  * ``autoslo.clusters.cluster`` / ``ManagedClusterPool`` — stores and evicts
    states alongside ``predicted_latencies`` (lifecycle).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from autoslo.workload_definition.query import QueryTextId


@dataclass(frozen=True)
class LSTMTensorState:
    """Pure NN tensor state from a ``BayesianPinchLSTM`` inference pass.

    Produced by :meth:`RuntimeNet.forward_pinch_state` and
    :meth:`RuntimeNet.step_after_state`.  All tensors are detached (no
    gradient tape) and should not be mutated after creation.

    This is the lowest-level state representation: it contains only what
    the LSTM layer itself knows about.  Application-level metadata (RPU,
    query text ID, etc.) is added by :class:`AfterLSTMState`.
    """

    fwd_out: torch.Tensor
    """(1, 1d_hidden_size) — forward LSTM output at the pinch point."""
    after_h: torch.Tensor
    """(num_layers, 1, 1d_hidden_size) — after-LSTM hidden state."""
    after_c: torch.Tensor
    """(num_layers, 1, 1d_hidden_size) — after-LSTM cell state."""


@dataclass(frozen=True)
class AfterLSTMState:
    """Cached after-LSTM state for a single active base query.

    Owns a :class:`LSTMTensorState` for the LSTM tensor fields, and adds
    the application-level metadata needed by :class:`IconqModel` to
    featurize future incremental steps.
    """

    tensor_state: LSTMTensorState
    """Pure LSTM tensor fields (fwd_out, after_h, after_c)."""

    # Cluster information.
    rpu: int
    """Cluster RPU; needed to look up stage predictions for new neighbors."""

    # Base query information; needed for incremental featurization.
    qa_query_text_id: QueryTextId
    """Base query text ID."""
    qa_start_time_s: float
    """Base query start time (seconds)."""
    qa_latency_prediction: float
    """Stage-model prediction for the base query."""

    def with_tensor_state(
        self, tensor_state: LSTMTensorState
    ) -> AfterLSTMState:
        """Return a copy with an updated :class:`LSTMTensorState`; application
        metadata (rpu, query text ID, start time, latency prediction) is
        unchanged.
        """
        return replace(self, tensor_state=tensor_state)
