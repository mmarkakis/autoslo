"""
Tests for the autoscaling subsystem:
:mod:`autoslo.capacity.autoscaling_policy`,
:mod:`autoslo.capacity.headroom_policy`, and
:mod:`autoslo.capacity.autoscaler`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.capacity.autoscaler import Autoscaler
from autoslo.capacity.autoscaling_policy import (
    AutoscalingAction,
    AutoscalingPolicy,
    NoOpPolicy,
    SpinUpRequest,
    TearDownRequest,
)
from autoslo.capacity.headroom_policy import HeadroomPolicy
from autoslo.routing.routing_core import PlacementScore, RoutingResult
from autoslo.workload_definition.query import Query, QueryTextId, SloMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_query(
    query_id: str = "q1",
    template_id: int = 1,
    rel_start_time_s: float = 0.0,
) -> Query:
    """Create a minimal Query for testing."""
    return Query(
        query_id=query_id,
        query_text_id=QueryTextId(f"public#{template_id}#0"),
        rel_start_time_s=rel_start_time_s,
    )


def _make_score(
    cluster: str = "c0",
    violation: float = 0.0,
    cost: float = 1.0,
) -> PlacementScore:
    return PlacementScore(
        cluster_name=cluster,
        marginal_slo_violation=violation,
        marginal_cost=cost,
        latencies={},
    )


def _make_result(
    cluster: str = "c0",
    violation: float = 0.0,
    query: Query | None = None,
) -> RoutingResult:
    score = _make_score(cluster=cluster, violation=violation)
    return RoutingResult(
        cluster_name=cluster,
        score=score,
        query=query or _make_query(),
    )


def _mock_pool(
    active_queries: dict[str, list[Query]] | None = None,
    draining: set[str] | None = None,
    cluster_names: list[str] | None = None,
) -> MagicMock:
    pool = MagicMock()
    active = active_queries or {}
    pool.get_all_active_queries.return_value = active
    pool.draining_cluster_names = draining or set()
    pool.ready_cluster_names = cluster_names or list(active.keys())
    pool.cluster_names = cluster_names or list(active.keys())
    return pool


# ===================================================================
# NoOpPolicy
# ===================================================================


class TestNoOpPolicy:
    """NoOpPolicy returns empty actions for every event."""

    def test_on_routing_result(self):
        policy = NoOpPolicy()
        result = _make_result()
        action = policy.on_routing_result(result, current_time_s=0.0)
        assert action.spin_ups == []
        assert action.tear_downs == []

    def test_on_query_complete(self):
        policy = NoOpPolicy()
        action = policy.on_query_complete("q1", "c0", current_time_s=0.0)
        assert action == AutoscalingAction()

    def test_on_time_advance(self):
        policy = NoOpPolicy()
        action = policy.on_time_advance(current_time_s=10.0)
        assert action == AutoscalingAction()


# ===================================================================
# HeadroomPolicy – spin-up
# ===================================================================


class TestHeadroomSpinUp:
    """Spin-up conditions in HeadroomPolicy."""

    def _policy(self, eta_crit: float = 0.1, **kwargs) -> HeadroomPolicy:
        return HeadroomPolicy(
            slo_resolver=SloResolver.from_dict(
                default_slo_s=10.0, slo_dict={}
            ),
            slo_metric=SloMetric.RELATIVE,
            eta_crit=eta_crit,
            allowed_rpu_sizes=[8, 32],
            **kwargs,
        )

    def test_spin_up_on_low_headroom(self):
        """Pool has a query with latency 9.5 / SLO 10 → headroom 0.05 < 0.1."""
        q = _make_query()
        latencies = {q.query_id: 9.5}
        pool = _mock_pool(active_queries={"c0": [q]})
        result = _make_result(violation=0.0, query=q)

        policy = self._policy()
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        action = policy.on_routing_result(result, current_time_s=0.0, current_latencies=latencies)
        assert len(action.spin_ups) == 1
        assert action.spin_ups[0].rpu in (8, 32)

    def test_spin_up_on_capacity_pressure(self):
        """Marginal SLO violation > 0 ⇒ pressure ⇒ spin-up."""
        q = _make_query()  # low latency → high headroom
        latencies = {q.query_id: 1.0}
        pool = _mock_pool(active_queries={"c0": [q]})
        result = _make_result(violation=0.5, query=q)

        policy = self._policy(eta_crit=0.001)  # very low threshold
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        action = policy.on_routing_result(result, current_time_s=0.0, current_latencies=latencies)
        assert len(action.spin_ups) == 1

    def test_no_spin_up_when_headroom_sufficient(self):
        """Healthy headroom and no pressure ⇒ no spin-up."""
        q = _make_query()  # headroom = 0.9
        latencies = {q.query_id: 1.0}
        pool = _mock_pool(active_queries={"c0": [q]})
        result = _make_result(violation=0.0, query=q)

        policy = self._policy()
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        action = policy.on_routing_result(result, current_time_s=0.0, current_latencies=latencies)
        assert action.spin_ups == []
        assert action.tear_downs == []

    def test_no_double_spin_up(self):
        """While a spin-up is pending, no second spin-up is triggered."""
        q = _make_query()
        latencies = {q.query_id: 9.5}
        pool = _mock_pool(active_queries={"c0": [q]})
        result = _make_result(violation=0.0, query=q)

        policy = self._policy()
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        # First spin-up
        a1 = policy.on_routing_result(result, current_time_s=0.0, current_latencies=latencies)
        assert len(a1.spin_ups) == 1
        assert policy.pending_count == 1

        # Second request — should NOT spin up
        a2 = policy.on_routing_result(result, current_time_s=1.0, current_latencies=latencies)
        assert a2.spin_ups == []

    def test_spin_up_unblocked_after_cluster_ready(self):
        """After on_cluster_ready, pending_count returns to 0."""
        q = _make_query()
        latencies = {q.query_id: 9.5}
        pool = _mock_pool(active_queries={"c0": [q]})
        result = _make_result(violation=0.0, query=q)

        policy = self._policy()
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        # First spin-up
        a1 = policy.on_routing_result(result, current_time_s=0.0, current_latencies=latencies)
        assert len(a1.spin_ups) == 1

        # New cluster ready
        policy.on_cluster_ready("c1", 8, 100.0)
        assert policy.pending_count == 0

        # Second spin-up now allowed
        a2 = policy.on_routing_result(result, current_time_s=200.0, current_latencies=latencies)
        assert len(a2.spin_ups) == 1

    def test_no_active_queries_high_headroom(self):
        """Empty pool → headroom defaults to 1.0 → no spin-up."""
        pool = _mock_pool(active_queries={"c0": []})
        result = _make_result(violation=0.0)

        policy = self._policy()
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        action = policy.on_routing_result(result, current_time_s=0.0)
        assert action.spin_ups == []


# ===================================================================
# HeadroomPolicy – tear-down
# ===================================================================


class TestHeadroomTearDown:
    """Tear-down conditions in HeadroomPolicy."""

    def _policy(
        self,
        idle_periods: int = 3,
        min_lifetime: float = 0.0,
    ) -> HeadroomPolicy:
        return HeadroomPolicy(
            slo_resolver=SloResolver.from_dict(
                default_slo_s=10.0, slo_dict={}
            ),
            slo_metric=SloMetric.RELATIVE,
            idle_periods_before_tear_down=idle_periods,
            min_cluster_lifetime_s=min_lifetime,
        )

    def test_tear_down_after_idle_periods(self):
        """Cluster idle for enough periods ⇒ tear-down."""
        pool = _mock_pool(active_queries={"c0": [], "c1": []})

        policy = self._policy(idle_periods=2, min_lifetime=0.0)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)
        policy.on_cluster_ready("c1", 8, 0.0)

        # Tick 1 — not enough idle periods yet
        a1 = policy.on_time_advance(current_time_s=60.0)
        assert a1.tear_downs == []

        # Tick 2 — should trigger tear-down for both
        a2 = policy.on_time_advance(current_time_s=120.0)
        assert len(a2.tear_downs) == 2

    def test_tear_down_deferred_by_min_lifetime(self):
        """Cluster is idle but hasn't reached min_cluster_lifetime_s."""
        pool = _mock_pool(active_queries={"c0": []})

        policy = self._policy(idle_periods=1, min_lifetime=600.0)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, ready_time_s=100.0)

        # Idle for 1 period but age = 200-100 = 100 < 600
        a = policy.on_time_advance(current_time_s=200.0)
        assert a.tear_downs == []

        # Now age = 800-100 = 700 ≥ 600 (but idle counter was reset
        # only if queries ran; re-idle from tick 2 onwards).
        # Advance a few more ticks to re-accumulate idle count.
        policy.on_time_advance(current_time_s=700.0)
        a2 = policy.on_time_advance(current_time_s=800.0)
        # counter is now at 3, age is 700 — should tear down
        assert len(a2.tear_downs) == 1
        assert a2.tear_downs[0].cluster_name == "c0"

    def test_active_cluster_not_torn_down(self):
        """Cluster with active queries never accumulates idle periods."""
        q = _make_query()
        pool = _mock_pool(active_queries={"c0": [q]})

        policy = self._policy(idle_periods=1, min_lifetime=0.0)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        for t in range(10):
            a = policy.on_time_advance(current_time_s=float(t * 60))
            assert a.tear_downs == []

    def test_draining_clusters_excluded(self):
        """Draining clusters should not be considered for tear-down."""
        pool = _mock_pool(
            active_queries={"c0": []},
            draining={"c0"},
        )

        policy = self._policy(idle_periods=1, min_lifetime=0.0)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        a = policy.on_time_advance(current_time_s=60.0)
        assert a.tear_downs == []


# ===================================================================
# HeadroomPolicy – on_attach reset
# ===================================================================


class TestHeadroomOnAttach:
    """on_attach should clear all mutable state."""

    def test_resets_pending_count(self):
        pool = _mock_pool(active_queries={"c0": []})
        policy = HeadroomPolicy(
            slo_resolver=SloResolver.from_dict(
                default_slo_s=10.0, slo_dict={}
            ),
        )
        policy.on_attach(pool)
        # Artificially bump pending
        policy._pending_count = 5
        # Re-attach should reset
        policy.on_attach(pool)
        assert policy.pending_count == 0

    def test_resets_idle_counts(self):
        pool = _mock_pool(active_queries={"c0": []})
        policy = HeadroomPolicy(
            slo_resolver=SloResolver.from_dict(
                default_slo_s=10.0, slo_dict={}
            ),
        )
        policy.on_attach(pool)
        policy._idle_counts["c0"] = 99
        policy.on_attach(pool)
        assert policy._idle_counts == {}


# ===================================================================
# HeadroomPolicy – property tunability
# ===================================================================


class TestHeadroomProperties:
    """Tunable properties should update correctly."""

    def test_eta_crit_setter(self):
        policy = HeadroomPolicy(
            slo_resolver=SloResolver.from_dict(
                default_slo_s=10.0, slo_dict={}
            ),
            eta_crit=0.1,
        )
        assert policy.eta_crit == 0.1
        policy.eta_crit = 0.5
        assert policy.eta_crit == 0.5

    def test_idle_periods_setter(self):
        policy = HeadroomPolicy(
            slo_resolver=SloResolver.from_dict(
                default_slo_s=10.0, slo_dict={}
            ),
            idle_periods_before_tear_down=5,
        )
        assert policy.idle_periods_before_tear_down == 5
        policy.idle_periods_before_tear_down = 10
        assert policy.idle_periods_before_tear_down == 10

    def test_allowed_rpu_sizes_sorted(self):
        policy = HeadroomPolicy(
            slo_resolver=SloResolver.from_dict(
                default_slo_s=10.0, slo_dict={}
            ),
            allowed_rpu_sizes=[32, 8, 128],
        )
        assert policy.allowed_rpu_sizes == [8, 32, 128]


# ===================================================================
# Autoscaler – coordinator
# ===================================================================


class TestAutoscaler:
    """Autoscaler dispatches events and executes returned actions."""

    def test_on_attach_called_at_construction(self):
        policy = MagicMock(spec=AutoscalingPolicy)
        pool = _mock_pool()
        spin_up = MagicMock()
        tear_down = MagicMock()

        Autoscaler(policy=policy, pool=pool,
                   on_spin_up=spin_up, on_tear_down=tear_down)
        policy.on_attach.assert_called_once_with(pool)

    def test_executes_spin_up_actions(self):
        policy = MagicMock(spec=AutoscalingPolicy)
        policy.on_routing_result.return_value = AutoscalingAction(
            spin_ups=[SpinUpRequest(rpu=32, reason="test")],
        )
        pool = _mock_pool()
        spin_up = MagicMock()
        tear_down = MagicMock()

        scaler = Autoscaler(
            policy=policy, pool=pool,
            on_spin_up=spin_up, on_tear_down=tear_down,
        )
        result = _make_result()
        scaler.on_routing_result(result, current_time_s=0.0)

        spin_up.assert_called_once_with("test", 32)
        tear_down.assert_not_called()

    def test_executes_tear_down_actions(self):
        policy = MagicMock(spec=AutoscalingPolicy)
        policy.on_time_advance.return_value = AutoscalingAction(
            tear_downs=[
                TearDownRequest(cluster_name="c0", reason="idle"),
                TearDownRequest(cluster_name="c1", reason="idle"),
            ],
        )
        pool = _mock_pool()
        spin_up = MagicMock()
        tear_down = MagicMock()

        scaler = Autoscaler(
            policy=policy, pool=pool,
            on_spin_up=spin_up, on_tear_down=tear_down,
        )
        scaler.on_time_advance(current_time_s=600.0)

        assert tear_down.call_count == 2
        tear_down.assert_any_call("c0")
        tear_down.assert_any_call("c1")

    def test_notify_cluster_ready_forwards_to_policy(self):
        policy = MagicMock(spec=AutoscalingPolicy)
        pool = _mock_pool()

        scaler = Autoscaler(
            policy=policy, pool=pool,
            on_spin_up=MagicMock(), on_tear_down=MagicMock(),
        )
        scaler.notify_cluster_ready("c2", 64, 100.0)
        policy.on_cluster_ready.assert_called_once_with("c2", 64, 100.0)

    def test_callback_exception_does_not_propagate(self):
        """If a spin-up callback throws, _execute logs and continues."""
        policy = MagicMock(spec=AutoscalingPolicy)
        policy.on_routing_result.return_value = AutoscalingAction(
            spin_ups=[
                SpinUpRequest(rpu=8, reason="first"),
                SpinUpRequest(rpu=32, reason="second"),
            ],
        )
        pool = _mock_pool()

        spin_up = MagicMock(side_effect=[RuntimeError("boom"), None])
        tear_down = MagicMock()

        scaler = Autoscaler(
            policy=policy, pool=pool,
            on_spin_up=spin_up, on_tear_down=tear_down,
        )
        # Should not raise
        scaler.on_routing_result(_make_result(), 0.0)
        assert spin_up.call_count == 2

    def test_policy_property(self):
        policy = NoOpPolicy()
        pool = _mock_pool()
        scaler = Autoscaler(
            policy=policy, pool=pool,
            on_spin_up=MagicMock(), on_tear_down=MagicMock(),
        )
        assert scaler.policy is policy


# ===================================================================
# Data types
# ===================================================================


class TestDataTypes:
    def test_spin_up_request_fields(self):
        req = SpinUpRequest(rpu=32, reason="headroom_low")
        assert req.rpu == 32
        assert req.reason == "headroom_low"

    def test_tear_down_request_fields(self):
        req = TearDownRequest(cluster_name="c0", reason="idle")
        assert req.cluster_name == "c0"

    def test_action_defaults_empty(self):
        action = AutoscalingAction()
        assert action.spin_ups == []
        assert action.tear_downs == []

    def test_action_with_entries(self):
        action = AutoscalingAction(
            spin_ups=[SpinUpRequest(rpu=8, reason="r")],
            tear_downs=[TearDownRequest(cluster_name="c0", reason="x")],
        )
        assert len(action.spin_ups) == 1
        assert len(action.tear_downs) == 1
