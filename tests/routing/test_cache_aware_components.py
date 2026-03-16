"""Tests for Phase 1: ClusterCacheState and CacheRiskScorer."""

from __future__ import annotations

import numpy as np
import pytest

from autoslo.routing.cluster_cache_state import (
    ClusterCacheState,
    DecayStrategyKind,
    ExponentialDecayStrategy,
    LRUCapacityStrategy,
    SlidingWindowStrategy,
    build_decay_strategy,
)
from autoslo.routing.cache_risk_scorer import (
    CacheRiskScorer,
    FutureQueryMix,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

N = 4  # small table dimension for tests


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


# =======================================================================
# ClusterCacheState — Exponential decay
# =======================================================================


class TestExponentialDecayStrategy:
    def test_initial_state_is_zeros(self):
        s = ExponentialDecayStrategy(alpha=0.5, n_tables=N)
        np.testing.assert_array_equal(s.current_state(0.0), np.zeros(N))

    def test_single_update(self):
        s = ExponentialDecayStrategy(alpha=0.5, n_tables=N)
        v = _vec(1, 0, 0, 0)
        s.update(v, timestamp_s=1.0)
        # state = 0.5*0 + 0.5*v = [0.5, 0, 0, 0]
        np.testing.assert_allclose(s.current_state(1.0), _vec(0.5, 0, 0, 0))

    def test_two_updates(self):
        s = ExponentialDecayStrategy(alpha=0.5, n_tables=N)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        s.update(_vec(0, 1, 0, 0), timestamp_s=2.0)
        # After first: [0.5, 0, 0, 0]
        # After second: 0.5*[0.5,0,0,0] + 0.5*[0,1,0,0] = [0.25, 0.5, 0, 0]
        np.testing.assert_allclose(
            s.current_state(2.0), _vec(0.25, 0.5, 0, 0)
        )

    def test_hypothetical_does_not_mutate(self):
        s = ExponentialDecayStrategy(alpha=0.5, n_tables=N)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        before = s.current_state(1.0).copy()
        hyp = s.hypothetical_state(_vec(0, 1, 0, 0), timestamp_s=2.0)
        # State should be unchanged.
        np.testing.assert_array_equal(s.current_state(1.0), before)
        # Hypothetical should reflect the would-be update.
        np.testing.assert_allclose(hyp, _vec(0.25, 0.5, 0, 0))

    def test_alpha_zero_is_memoryless(self):
        s = ExponentialDecayStrategy(alpha=0.0, n_tables=N)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        s.update(_vec(0, 0, 0, 1), timestamp_s=2.0)
        np.testing.assert_allclose(s.current_state(2.0), _vec(0, 0, 0, 1))

    def test_alpha_one_never_decays(self):
        s = ExponentialDecayStrategy(alpha=1.0, n_tables=N)
        s.update(_vec(1, 2, 3, 4), timestamp_s=1.0)
        s.update(_vec(99, 99, 99, 99), timestamp_s=2.0)
        # state stays zeros because alpha=1 means 1*state + 0*new
        np.testing.assert_allclose(s.current_state(2.0), np.zeros(N))

    def test_clone(self):
        s = ExponentialDecayStrategy(alpha=0.5, n_tables=N)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        c = s.clone()
        c.update(_vec(0, 1, 0, 0), timestamp_s=2.0)
        # Original should be unchanged.
        np.testing.assert_allclose(s.current_state(2.0), _vec(0.5, 0, 0, 0))
        np.testing.assert_allclose(c.current_state(2.0), _vec(0.25, 0.5, 0, 0))

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ExponentialDecayStrategy(alpha=-0.1, n_tables=N)
        with pytest.raises(ValueError):
            ExponentialDecayStrategy(alpha=1.1, n_tables=N)


# =======================================================================
# ClusterCacheState — Sliding window
# =======================================================================


class TestSlidingWindowStrategy:
    def test_max_queries_eviction(self):
        s = SlidingWindowStrategy(n_tables=N, max_queries=2)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        s.update(_vec(0, 1, 0, 0), timestamp_s=2.0)
        s.update(_vec(0, 0, 1, 0), timestamp_s=3.0)
        # Only last 2 should remain: [0,1,0,0] and [0,0,1,0].
        state = s.current_state(3.0)
        np.testing.assert_allclose(state, _vec(0, 0.5, 0.5, 0))

    def test_time_based_eviction(self):
        s = SlidingWindowStrategy(n_tables=N, window_s=2.0)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        s.update(_vec(0, 1, 0, 0), timestamp_s=2.5)
        # At t=3.5, cutoff=1.5 → first entry (t=1.0) is evicted.
        state = s.current_state(3.5)
        np.testing.assert_allclose(state, _vec(0, 1, 0, 0))

    def test_empty_state(self):
        s = SlidingWindowStrategy(n_tables=N, max_queries=5)
        np.testing.assert_array_equal(s.current_state(0.0), np.zeros(N))

    def test_hypothetical_does_not_mutate(self):
        s = SlidingWindowStrategy(n_tables=N, max_queries=2)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        before = s.current_state(1.0).copy()
        hyp = s.hypothetical_state(_vec(0, 1, 0, 0), timestamp_s=2.0)
        np.testing.assert_array_equal(s.current_state(1.0), before)
        # Hypothetical should have both entries (2 ≤ max_queries).
        np.testing.assert_allclose(hyp, _vec(0.5, 0.5, 0, 0))

    def test_hypothetical_evicts_oldest(self):
        s = SlidingWindowStrategy(n_tables=N, max_queries=1)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        hyp = s.hypothetical_state(_vec(0, 1, 0, 0), timestamp_s=2.0)
        # Only the hypothetical entry should survive.
        np.testing.assert_allclose(hyp, _vec(0, 1, 0, 0))

    def test_requires_at_least_one_limit(self):
        with pytest.raises(ValueError):
            SlidingWindowStrategy(n_tables=N)

    def test_clone(self):
        s = SlidingWindowStrategy(n_tables=N, max_queries=3)
        s.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        c = s.clone()
        c.update(_vec(0, 0, 0, 1), timestamp_s=2.0)
        # Original unchanged.
        np.testing.assert_allclose(s.current_state(2.0), _vec(1, 0, 0, 0))
        np.testing.assert_allclose(c.current_state(2.0), _vec(0.5, 0, 0, 0.5))


# =======================================================================
# ClusterCacheState — LRU capacity
# =======================================================================


class TestLRUCapacityStrategy:
    def test_within_capacity(self):
        s = LRUCapacityStrategy(n_tables=N, capacity=100)
        s.update(_vec(10, 20, 0, 0), timestamp_s=1.0)
        np.testing.assert_allclose(s.current_state(1.0), _vec(10, 20, 0, 0))

    def test_evicts_lru_when_over_capacity(self):
        s = LRUCapacityStrategy(n_tables=N, capacity=50)
        s.update(_vec(30, 0, 0, 0), timestamp_s=1.0)  # table 0 accessed
        s.update(_vec(0, 30, 0, 0), timestamp_s=2.0)  # table 1 accessed
        # Total = 60 > 50 → table 0 (LRU) evicted.
        np.testing.assert_allclose(s.current_state(2.0), _vec(0, 30, 0, 0))

    def test_hypothetical_does_not_mutate(self):
        s = LRUCapacityStrategy(n_tables=N, capacity=50)
        s.update(_vec(30, 0, 0, 0), timestamp_s=1.0)
        before = s.current_state(1.0).copy()
        hyp = s.hypothetical_state(_vec(0, 30, 0, 0), timestamp_s=2.0)
        np.testing.assert_array_equal(s.current_state(1.0), before)
        # Should have evicted table 0.
        np.testing.assert_allclose(hyp, _vec(0, 30, 0, 0))

    def test_reaccessing_table_updates_lru_order(self):
        s = LRUCapacityStrategy(n_tables=N, capacity=50)
        s.update(_vec(20, 0, 0, 0), timestamp_s=1.0)  # table 0
        s.update(_vec(0, 20, 0, 0), timestamp_s=2.0)  # table 1
        s.update(_vec(10, 0, 0, 0), timestamp_s=3.0)  # re-access table 0
        # Now table 1 is LRU.  Total = 10+20 = 30 ≤ 50, no eviction.
        np.testing.assert_allclose(s.current_state(3.0), _vec(10, 20, 0, 0))
        # Add table 2 to push over capacity.
        s.update(_vec(0, 0, 30, 0), timestamp_s=4.0)
        # Total = 10+20+30 = 60 > 50 → table 1 (LRU) evicted → 10+30=40 ≤ 50.
        np.testing.assert_allclose(s.current_state(4.0), _vec(10, 0, 30, 0))

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            LRUCapacityStrategy(n_tables=N, capacity=0)

    def test_clone(self):
        s = LRUCapacityStrategy(n_tables=N, capacity=100)
        s.update(_vec(10, 20, 0, 0), timestamp_s=1.0)
        c = s.clone()
        c.update(_vec(0, 0, 30, 40), timestamp_s=2.0)
        np.testing.assert_allclose(s.current_state(2.0), _vec(10, 20, 0, 0))
        np.testing.assert_allclose(c.current_state(2.0), _vec(10, 20, 30, 40))


# =======================================================================
# build_decay_strategy factory
# =======================================================================


class TestBuildDecayStrategy:
    def test_exponential_default(self):
        s = build_decay_strategy("exponential", n_tables=N)
        assert isinstance(s, ExponentialDecayStrategy)

    def test_sliding_window(self):
        s = build_decay_strategy("sliding_window", N, {"max_queries": 5})
        assert isinstance(s, SlidingWindowStrategy)

    def test_lru(self):
        s = build_decay_strategy(DecayStrategyKind.LRU, N, {"capacity": 100})
        assert isinstance(s, LRUCapacityStrategy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_decay_strategy("unknown_strategy", N)


# =======================================================================
# ClusterCacheState wrapper
# =======================================================================


class TestClusterCacheState:
    def test_default_strategy(self):
        cache = ClusterCacheState(n_tables=N)
        cache.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        state = cache.current_state(1.0)
        assert state.shape == (N,)
        assert state[0] > 0

    def test_custom_strategy(self):
        strat = SlidingWindowStrategy(n_tables=N, max_queries=1)
        cache = ClusterCacheState(n_tables=N, strategy=strat)
        cache.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        cache.update(_vec(0, 1, 0, 0), timestamp_s=2.0)
        # Only last query kept.
        np.testing.assert_allclose(cache.current_state(2.0), _vec(0, 1, 0, 0))

    def test_clone(self):
        cache = ClusterCacheState(n_tables=N)
        cache.update(_vec(1, 0, 0, 0), timestamp_s=1.0)
        cloned = cache.clone()
        cloned.update(_vec(0, 0, 0, 1), timestamp_s=2.0)
        # Original unchanged.
        np.testing.assert_allclose(
            cache.current_state(1.0), cache.current_state(2.0)
        )


# =======================================================================
# FutureQueryMix validation
# =======================================================================


class TestFutureQueryMix:
    def test_valid_construction(self):
        mix = FutureQueryMix(
            template_ids=["t1", "t2"],
            probabilities=np.array([0.6, 0.4]),
            table_vectors=np.eye(2, N),
            slo_tightness=np.array([0.5, 0.8]),
        )
        assert len(mix.template_ids) == 2

    def test_shape_mismatch_probabilities(self):
        with pytest.raises(ValueError, match="probabilities"):
            FutureQueryMix(
                template_ids=["t1"],
                probabilities=np.array([0.5, 0.5]),
                table_vectors=np.ones((1, N)),
                slo_tightness=np.array([0.5]),
            )

    def test_shape_mismatch_table_vectors(self):
        with pytest.raises(ValueError, match="table_vectors"):
            FutureQueryMix(
                template_ids=["t1"],
                probabilities=np.array([1.0]),
                table_vectors=np.ones((2, N)),
                slo_tightness=np.array([0.5]),
            )

    def test_shape_mismatch_slo_tightness(self):
        with pytest.raises(ValueError, match="slo_tightness"):
            FutureQueryMix(
                template_ids=["t1"],
                probabilities=np.array([1.0]),
                table_vectors=np.ones((1, N)),
                slo_tightness=np.array([0.5, 0.8]),
            )


# =======================================================================
# CacheRiskScorer
# =======================================================================


class TestCacheRiskScorer:
    def _make_future(
        self,
        table_vectors: np.ndarray,
        probabilities: np.ndarray | None = None,
        slo_tightness: np.ndarray | None = None,
    ) -> FutureQueryMix:
        k = table_vectors.shape[0]
        return FutureQueryMix(
            template_ids=[f"t{i}" for i in range(k)],
            probabilities=(
                probabilities
                if probabilities is not None
                else np.ones(k) / k
            ),
            table_vectors=table_vectors,
            slo_tightness=(
                slo_tightness if slo_tightness is not None else np.ones(k)
            ),
        )

    def test_no_degradation_returns_zero(self):
        """If the cache state doesn't change, risk is 0."""
        current = _vec(1, 0, 0, 0)
        hypothetical = _vec(1, 0, 0, 0)
        future = self._make_future(np.array([[1, 0, 0, 0]]))
        risk = CacheRiskScorer.score_cache_risk(current, hypothetical, future)
        assert risk == 0.0

    def test_improvement_returns_zero(self):
        """If the cache gets *better* for future queries, risk should be 0."""
        current = _vec(0, 1, 0, 0)
        hypothetical = _vec(1, 0, 0, 0)
        future = self._make_future(np.array([[1, 0, 0, 0]]))
        risk = CacheRiskScorer.score_cache_risk(current, hypothetical, future)
        assert risk == 0.0

    def test_degradation_returns_positive(self):
        """Switching cache from aligned to misaligned produces risk > 0."""
        current = _vec(1, 0, 0, 0)
        hypothetical = _vec(0, 1, 0, 0)
        # Future query wants table 0.
        future = self._make_future(
            np.array([[1, 0, 0, 0]]),
            probabilities=np.array([1.0]),
            slo_tightness=np.array([1.0]),
        )
        risk = CacheRiskScorer.score_cache_risk(current, hypothetical, future)
        assert risk > 0.0

    def test_risk_scales_with_probability(self):
        current = _vec(1, 0, 0, 0)
        hypothetical = _vec(0, 1, 0, 0)
        tv = np.array([[1, 0, 0, 0]])

        risk_high = CacheRiskScorer.score_cache_risk(
            current,
            hypothetical,
            self._make_future(tv, np.array([1.0]), np.array([1.0])),
        )
        risk_low = CacheRiskScorer.score_cache_risk(
            current,
            hypothetical,
            self._make_future(tv, np.array([0.1]), np.array([1.0])),
        )
        assert risk_high > risk_low > 0.0

    def test_risk_scales_with_slo_tightness(self):
        current = _vec(1, 0, 0, 0)
        hypothetical = _vec(0, 1, 0, 0)
        tv = np.array([[1, 0, 0, 0]])

        risk_tight = CacheRiskScorer.score_cache_risk(
            current,
            hypothetical,
            self._make_future(tv, np.array([1.0]), np.array([1.0])),
        )
        risk_loose = CacheRiskScorer.score_cache_risk(
            current,
            hypothetical,
            self._make_future(tv, np.array([1.0]), np.array([0.1])),
        )
        assert risk_tight > risk_loose > 0.0

    def test_multiple_templates(self):
        """Risk aggregates over all future templates."""
        current = _vec(1, 0, 0, 0)
        hypothetical = _vec(0, 0, 1, 0)
        future = self._make_future(
            # Template 0 wants table 0, template 1 wants table 2.
            np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),
            probabilities=np.array([0.5, 0.5]),
            slo_tightness=np.array([1.0, 1.0]),
        )
        risk = CacheRiskScorer.score_cache_risk(current, hypothetical, future)
        # Template 0: degraded (table 0 gone).  Template 1: improved (table 2 added).
        # Only template 0 contributes risk.
        assert risk > 0.0

    def test_zero_cache_state(self):
        """Empty cache — no degradation possible."""
        current = np.zeros(N)
        hypothetical = _vec(1, 0, 0, 0)
        future = self._make_future(np.array([[1, 0, 0, 0]]))
        risk = CacheRiskScorer.score_cache_risk(current, hypothetical, future)
        assert risk == 0.0

    def test_zero_norm_template(self):
        """Template with no table access should contribute zero risk."""
        current = _vec(1, 0, 0, 0)
        hypothetical = _vec(0, 1, 0, 0)
        future = self._make_future(
            np.array([[0, 0, 0, 0]]),
            probabilities=np.array([1.0]),
            slo_tightness=np.array([1.0]),
        )
        risk = CacheRiskScorer.score_cache_risk(current, hypothetical, future)
        assert risk == 0.0
