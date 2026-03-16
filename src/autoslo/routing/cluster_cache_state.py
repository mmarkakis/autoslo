"""
cluster_cache_state.py
----------------------
Per-cluster hypothesized cache state for cache-aware routing.

Tracks a cardinality-weighted table vector (length N, aligned with
:attr:`IconqQueryFeaturizer.top_table_names`) representing the tables
likely to be hot in a cluster's cache.  Three pluggable decay strategies
control how older entries age out.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Strategy enum (for config dispatch)
# ---------------------------------------------------------------------------

class DecayStrategyKind(Enum):
    EXPONENTIAL = "exponential"
    SLIDING_WINDOW = "sliding_window"
    LRU = "lru"


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------

class CacheDecayStrategy(ABC):
    """Protocol for cache-vector decay strategies."""

    @abstractmethod
    def update(
        self, table_vector: np.ndarray, timestamp_s: float
    ) -> None:
        """Incorporate a newly-routed query's table vector."""

    @abstractmethod
    def current_state(self, timestamp_s: float) -> np.ndarray:
        """Return the current cache-state vector (applying time-based eviction)."""

    @abstractmethod
    def hypothetical_state(
        self, table_vector: np.ndarray, timestamp_s: float
    ) -> np.ndarray:
        """Return the cache state *as if* ``table_vector`` were added, without mutating."""

    @abstractmethod
    def clone(self) -> "CacheDecayStrategy":
        """Deep-copy for safe what-if exploration."""


# ---------------------------------------------------------------------------
# Exponential decay
# ---------------------------------------------------------------------------

class ExponentialDecayStrategy(CacheDecayStrategy):
    """``state ← α·state + (1−α)·new_vector`` on each update.

    Parameters
    ----------
    alpha :
        Retention weight for existing state.  0 = memoryless, 1 = never decay.
    n_tables :
        Length of the table vector.
    """

    def __init__(self, alpha: float, n_tables: int) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha
        self._state = np.zeros(n_tables, dtype=np.float64)

    def update(self, table_vector: np.ndarray, timestamp_s: float) -> None:
        self._state = self._alpha * self._state + (1 - self._alpha) * table_vector

    def current_state(self, timestamp_s: float) -> np.ndarray:
        return self._state.copy()

    def hypothetical_state(
        self, table_vector: np.ndarray, timestamp_s: float
    ) -> np.ndarray:
        return self._alpha * self._state + (1 - self._alpha) * table_vector

    def clone(self) -> "ExponentialDecayStrategy":
        c = ExponentialDecayStrategy.__new__(ExponentialDecayStrategy)
        c._alpha = self._alpha
        c._state = self._state.copy()
        return c


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

@dataclass
class _WindowEntry:
    table_vector: np.ndarray
    timestamp_s: float


class SlidingWindowStrategy(CacheDecayStrategy):
    """Retain the last *max_queries* queries **or** queries within
    the last *window_s* seconds, whichever is more restrictive.

    At least one of ``max_queries`` or ``window_s`` must be set.

    Parameters
    ----------
    n_tables :
        Length of the table vector.
    max_queries :
        Maximum number of recent queries to keep (FIFO).
        ``None`` means unlimited.
    window_s :
        Maximum age (in seconds) for a query to remain in the window.
        ``None`` means time is ignored.
    """

    def __init__(
        self,
        n_tables: int,
        max_queries: int | None = None,
        window_s: float | None = None,
    ) -> None:
        if max_queries is None and window_s is None:
            raise ValueError("At least one of max_queries / window_s required.")
        self._n_tables = n_tables
        self._max_queries = max_queries
        self._window_s = window_s
        self._entries: deque[_WindowEntry] = deque()

    # -- helpers --

    def _evict(self, timestamp_s: float) -> None:
        """Remove entries that have expired by count or time."""
        if self._window_s is not None:
            cutoff = timestamp_s - self._window_s
            while self._entries and self._entries[0].timestamp_s < cutoff:
                self._entries.popleft()
        if self._max_queries is not None:
            while len(self._entries) > self._max_queries:
                self._entries.popleft()

    def _aggregate(self, entries: deque[_WindowEntry] | list[_WindowEntry]) -> np.ndarray:
        if not entries:
            return np.zeros(self._n_tables, dtype=np.float64)
        return np.mean([e.table_vector for e in entries], axis=0)

    # -- interface --

    def update(self, table_vector: np.ndarray, timestamp_s: float) -> None:
        self._entries.append(_WindowEntry(table_vector=table_vector.copy(), timestamp_s=timestamp_s))
        self._evict(timestamp_s)

    def current_state(self, timestamp_s: float) -> np.ndarray:
        self._evict(timestamp_s)
        return self._aggregate(self._entries)

    def hypothetical_state(
        self, table_vector: np.ndarray, timestamp_s: float
    ) -> np.ndarray:
        # Simulate adding without mutation.
        tmp: list[_WindowEntry] = list(self._entries)
        tmp.append(_WindowEntry(table_vector=table_vector, timestamp_s=timestamp_s))
        # Apply eviction rules on the tmp list.
        if self._window_s is not None:
            cutoff = timestamp_s - self._window_s
            tmp = [e for e in tmp if e.timestamp_s >= cutoff]
        if self._max_queries is not None:
            tmp = tmp[-self._max_queries:]
        return self._aggregate(tmp)

    def clone(self) -> "SlidingWindowStrategy":
        c = SlidingWindowStrategy.__new__(SlidingWindowStrategy)
        c._n_tables = self._n_tables
        c._max_queries = self._max_queries
        c._window_s = self._window_s
        c._entries = deque(
            _WindowEntry(table_vector=e.table_vector.copy(), timestamp_s=e.timestamp_s)
            for e in self._entries
        )
        return c


# ---------------------------------------------------------------------------
# LRU capacity
# ---------------------------------------------------------------------------

class LRUCapacityStrategy(CacheDecayStrategy):
    """Fixed capacity budget; tables evicted LRU-first.

    Each table occupies ``cardinality`` units of capacity (taken from its
    feature value in the incoming vector).  When the budget is exceeded the
    least-recently-used tables are evicted first.

    Parameters
    ----------
    n_tables :
        Length of the table vector.
    capacity :
        Total capacity budget (in the same cardinality units as the feature
        vector).
    """

    def __init__(self, n_tables: int, capacity: float) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._n_tables = n_tables
        self._capacity = capacity
        # Per-table current cardinality and last-access order.
        self._table_card = np.zeros(n_tables, dtype=np.float64)
        # Lower = older access.  We bump on every update.
        self._table_access_order = np.zeros(n_tables, dtype=np.float64)
        self._clock: float = 0.0

    def _apply_vector(
        self,
        state: np.ndarray,
        access_order: np.ndarray,
        table_vector: np.ndarray,
        clock: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Apply a table vector to a (state, access_order) pair, evicting as needed."""
        state = state.copy()
        access_order = access_order.copy()

        # Mark accessed tables.
        nonzero = table_vector > 0
        for idx in np.where(nonzero)[0]:
            state[idx] = table_vector[idx]
            clock += 1
            access_order[idx] = clock

        # Evict LRU while over capacity.
        while state.sum() > self._capacity and (state > 0).any():
            # Find the table with smallest access_order among those present.
            present = state > 0
            candidates = np.where(present)[0]
            lru_idx = candidates[np.argmin(access_order[candidates])]
            state[lru_idx] = 0.0
            access_order[lru_idx] = 0.0

        return state, access_order, clock

    def update(self, table_vector: np.ndarray, timestamp_s: float) -> None:
        self._table_card, self._table_access_order, self._clock = (
            self._apply_vector(
                self._table_card,
                self._table_access_order,
                table_vector,
                self._clock,
            )
        )

    def current_state(self, timestamp_s: float) -> np.ndarray:
        return self._table_card.copy()

    def hypothetical_state(
        self, table_vector: np.ndarray, timestamp_s: float
    ) -> np.ndarray:
        hyp, _, _ = self._apply_vector(
            self._table_card,
            self._table_access_order,
            table_vector,
            self._clock,
        )
        return hyp

    def clone(self) -> "LRUCapacityStrategy":
        c = LRUCapacityStrategy.__new__(LRUCapacityStrategy)
        c._n_tables = self._n_tables
        c._capacity = self._capacity
        c._table_card = self._table_card.copy()
        c._table_access_order = self._table_access_order.copy()
        c._clock = self._clock
        return c


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_decay_strategy(
    kind: str | DecayStrategyKind,
    n_tables: int,
    params: dict[str, Any] | None = None,
) -> CacheDecayStrategy:
    """Construct a decay strategy from config values.

    Parameters
    ----------
    kind :
        One of ``"exponential"``, ``"sliding_window"``, ``"lru"``
        (or the corresponding enum member).
    n_tables :
        Dimensionality of the table vector.
    params :
        Strategy-specific keyword arguments.

        * ``exponential``: ``{"alpha": float}``
        * ``sliding_window``: ``{"max_queries": int | None, "window_s": float | None}``
        * ``lru``: ``{"capacity": float}``
    """
    if isinstance(kind, str):
        kind = DecayStrategyKind(kind)
    params = params or {}

    if kind is DecayStrategyKind.EXPONENTIAL:
        return ExponentialDecayStrategy(
            alpha=params.get("alpha", 0.7),
            n_tables=n_tables,
        )
    elif kind is DecayStrategyKind.SLIDING_WINDOW:
        return SlidingWindowStrategy(
            n_tables=n_tables,
            max_queries=params.get("max_queries"),
            window_s=params.get("window_s"),
        )
    elif kind is DecayStrategyKind.LRU:
        return LRUCapacityStrategy(
            n_tables=n_tables,
            capacity=params.get("capacity", 1e9),
        )
    else:
        raise ValueError(f"Unknown decay strategy kind: {kind!r}")


# ---------------------------------------------------------------------------
# ClusterCacheState — per-cluster wrapper
# ---------------------------------------------------------------------------

class ClusterCacheState:
    """Hypothesized cache state for a single cluster.

    Thin wrapper around a :class:`CacheDecayStrategy` that also remembers
    the number of table dimensions and provides convenience helpers.

    Parameters
    ----------
    n_tables :
        Number of table dimensions (= ``IconqQueryFeaturizer._n``).
    strategy :
        A pre-built decay strategy, or ``None`` to use the default
        (exponential with α=0.7).
    """

    def __init__(
        self,
        n_tables: int,
        strategy: CacheDecayStrategy | None = None,
    ) -> None:
        self._n_tables = n_tables
        self._strategy = strategy or ExponentialDecayStrategy(
            alpha=0.7, n_tables=n_tables
        )

    @property
    def n_tables(self) -> int:
        return self._n_tables

    def update(self, table_vector: np.ndarray, timestamp_s: float) -> None:
        """Record a query that was just routed to this cluster."""
        self._strategy.update(table_vector, timestamp_s)

    def current_state(self, timestamp_s: float) -> np.ndarray:
        """Current cache-state vector (length *n_tables*)."""
        return self._strategy.current_state(timestamp_s)

    def hypothetical_state(
        self, table_vector: np.ndarray, timestamp_s: float
    ) -> np.ndarray:
        """What the cache state *would* look like if ``table_vector`` were
        routed here, without mutating."""
        return self._strategy.hypothetical_state(table_vector, timestamp_s)

    def clone(self) -> "ClusterCacheState":
        """Deep copy (e.g. for counterfactual exploration)."""
        return ClusterCacheState(
            n_tables=self._n_tables,
            strategy=self._strategy.clone(),
        )
