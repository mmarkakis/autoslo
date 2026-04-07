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
    CapacityCheckpoint,
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
    predicted_latency_s: float = -1.0,
) -> RoutingResult:
    score = _make_score(cluster=cluster, violation=violation)
    return RoutingResult(
        cluster_name=cluster,
        score=score,
        query=query or _make_query(),
        predicted_latency_s=predicted_latency_s,
    )


def _mock_pool(
    active_queries: dict[str, list[Query]] | None = None,
    draining: set[str] | None = None,
    cluster_names: list[str] | None = None,
    cluster_rpu_multiset: dict[int, int] | None = None,
) -> MagicMock:
    pool = MagicMock()
    active = active_queries or {}
    pool.get_all_active_queries.return_value = active
    pool.draining_cluster_names = draining or set()
    pool.ready_cluster_names = cluster_names or list(active.keys())
    pool.cluster_names = cluster_names or list(active.keys())
    pool.cluster_rpu_multiset = cluster_rpu_multiset or {}
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
        """Window has queries with latency 9.5 / SLO 10 → headroom 0.05 < 0.1."""
        pool = _mock_pool(active_queries={"c0": []})

        policy = self._policy(min_window_observations=1)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        result = _make_result(violation=0.0, predicted_latency_s=9.5)
        action = policy.on_routing_result(result, current_time_s=0.0)
        assert len(action.spin_ups) == 1
        assert action.spin_ups[0].rpu in (8, 32)

    def test_spin_up_on_capacity_pressure(self):
        """Marginal SLO violation > 0 ⇒ pressure ⇒ spin-up."""
        pool = _mock_pool(active_queries={"c0": []})
        result = _make_result(violation=0.5, predicted_latency_s=1.0)

        policy = self._policy(eta_crit=0.001)  # very low threshold
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        # Pressure needs window_size >= 1 (one post-ready routing).
        action = policy.on_routing_result(result, current_time_s=0.0)
        assert len(action.spin_ups) == 1

    def test_no_spin_up_when_headroom_sufficient(self):
        """Healthy headroom and no pressure ⇒ no spin-up."""
        pool = _mock_pool(active_queries={"c0": []})
        result = _make_result(violation=0.0, predicted_latency_s=1.0)

        policy = self._policy(min_window_observations=1)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        action = policy.on_routing_result(result, current_time_s=0.0)
        assert action.spin_ups == []
        assert action.tear_downs == []

    def test_no_double_spin_up(self):
        """While a spin-up is pending, no second spin-up is triggered."""
        pool = _mock_pool(active_queries={"c0": []})
        result = _make_result(violation=0.0, predicted_latency_s=9.5)

        policy = self._policy(min_window_observations=1)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        # First spin-up
        a1 = policy.on_routing_result(result, current_time_s=0.0)
        assert len(a1.spin_ups) == 1
        assert policy.pending_count == 1

        # Second request — should NOT spin up (window frozen)
        a2 = policy.on_routing_result(result, current_time_s=1.0)
        assert a2.spin_ups == []

    def test_spin_up_unblocked_after_cluster_ready(self):
        """After on_cluster_ready, window resets; fresh evidence can trigger."""
        pool = _mock_pool(active_queries={"c0": []})
        result = _make_result(violation=0.0, predicted_latency_s=9.5)

        policy = self._policy(min_window_observations=1)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        # First spin-up
        a1 = policy.on_routing_result(result, current_time_s=0.0)
        assert len(a1.spin_ups) == 1

        # New cluster ready — window clears, pending drops to 0
        policy.on_cluster_ready("c1", 8, 100.0)
        assert policy.pending_count == 0
        assert policy.get_routing_window() == []  # window was cleared

        # Second spin-up requires fresh evidence in the new window
        a2 = policy.on_routing_result(result, current_time_s=200.0)
        assert len(a2.spin_ups) == 1

    def test_no_active_queries_high_headroom(self):
        """Empty window → headroom defaults to 1.0 → no spin-up."""
        pool = _mock_pool(active_queries={"c0": []})
        result = _make_result(violation=0.0, predicted_latency_s=1.0)

        policy = self._policy(min_window_observations=1)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        action = policy.on_routing_result(result, current_time_s=0.0)
        assert action.spin_ups == []

    def test_no_spurious_spin_up_right_after_cluster_ready(self):
        """The core scenario: right after cluster_ready, the window is empty
        so neither headroom nor pressure can trigger a second spin-up, even
        though the underlying system is still stressed."""
        pool = _mock_pool(active_queries={"c0": []})

        policy = self._policy(min_window_observations=3)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        # Build up 3 bad routing results → triggers spin-up.
        bad = _make_result(violation=0.0, predicted_latency_s=9.5)
        policy.on_routing_result(bad, current_time_s=1.0)
        policy.on_routing_result(bad, current_time_s=2.0)
        a = policy.on_routing_result(bad, current_time_s=3.0)
        assert len(a.spin_ups) == 1
        assert policy.pending_count == 1

        # New cluster comes online — window clears.
        policy.on_cluster_ready("c1", 8, 100.0)
        assert policy.pending_count == 0

        # Immediately send a bad routing result.  Because the window has
        # only 1 entry (< min_window_observations=3), headroom alone
        # cannot trigger.  Pressure is also gated (violation=0).
        a2 = policy.on_routing_result(bad, current_time_s=100.1)
        assert a2.spin_ups == []

        # Two more bad results → window reaches 3 → now it fires.
        policy.on_routing_result(bad, current_time_s=100.2)
        a3 = policy.on_routing_result(bad, current_time_s=100.3)
        assert len(a3.spin_ups) == 1

    def test_pressure_fires_after_single_post_ready_routing(self):
        """Pressure (marginal_slo_violation > 0) fires with just 1 entry
        in the fresh window, since it represents acute routing failure."""
        pool = _mock_pool(active_queries={"c0": []})
        pressure_result = _make_result(
            violation=0.5, predicted_latency_s=1.0,
        )

        policy = self._policy(eta_crit=0.001, min_window_observations=5)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        # Single routing with pressure → fires despite min_window=5.
        action = policy.on_routing_result(
            pressure_result, current_time_s=0.0,
        )
        assert len(action.spin_ups) == 1

    def test_window_frozen_while_pending(self):
        """Routing results while a spin-up is pending do NOT grow the window."""
        pool = _mock_pool(active_queries={"c0": []})

        policy = self._policy(min_window_observations=1)
        policy.on_attach(pool)
        policy.on_cluster_ready("c0", 8, 0.0)

        bad = _make_result(violation=0.0, predicted_latency_s=9.5)
        a1 = policy.on_routing_result(bad, current_time_s=0.0)
        assert len(a1.spin_ups) == 1
        window_size_at_trigger = len(policy.get_routing_window())

        # More routing while pending — window should not grow.
        for t in range(1, 10):
            policy.on_routing_result(bad, current_time_s=float(t))
        assert len(policy.get_routing_window()) == window_size_at_trigger


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

    def test_capacity_checkpoint_frozen(self):
        cp = CapacityCheckpoint(rel_time_s=180.0, min_rpus=(4, 32, 32))
        assert cp.rel_time_s == 180.0
        assert cp.min_rpus == (4, 32, 32)
        with pytest.raises(AttributeError):
            cp.rel_time_s = 999.0  # type: ignore[misc]


# ===================================================================
# CapacityCheckpoints (Autoscaler reconciliation)
# ===================================================================


class TestCapacityCheckpoints:
    """Tests for the Autoscaler's capacity checkpoint reconciliation."""

    def _make_scaler(
        self,
        checkpoints: list[CapacityCheckpoint],
        cluster_rpu_multiset: dict[int, int] | None = None,
    ) -> tuple[Autoscaler, MagicMock, MagicMock]:
        pool = _mock_pool(
            active_queries={"c0": []},
            cluster_rpu_multiset=cluster_rpu_multiset or {},
        )
        spin_up = MagicMock()
        tear_down = MagicMock()
        scaler = Autoscaler(
            policy=NoOpPolicy(),
            pool=pool,
            on_spin_up=spin_up,
            on_tear_down=tear_down,
            capacity_checkpoints=checkpoints,
        )
        return scaler, spin_up, pool

    def test_no_checkpoints_is_noop(self):
        scaler, spin_up, _ = self._make_scaler([])
        scaler.reconcile_checkpoints_up_to(999.0)
        spin_up.assert_not_called()

    def test_checkpoint_already_satisfied(self):
        """Pool already has the desired RPU set — no spin-ups."""
        cp = CapacityCheckpoint(rel_time_s=100.0, min_rpus=(4, 32))
        scaler, spin_up, _ = self._make_scaler(
            [cp], cluster_rpu_multiset={4: 1, 32: 1}
        )
        scaler.reconcile_checkpoints_up_to(100.0)
        spin_up.assert_not_called()

    def test_checkpoint_spins_up_gap(self):
        """Pool has {4, 32} but checkpoint wants {4, 32, 32} — gap is one 32."""
        cp = CapacityCheckpoint(rel_time_s=100.0, min_rpus=(4, 32, 32))
        scaler, spin_up, _ = self._make_scaler(
            [cp], cluster_rpu_multiset={4: 1, 32: 1}
        )
        scaler.reconcile_checkpoints_up_to(100.0)
        spin_up.assert_called_once()
        reason, rpu = spin_up.call_args[0]
        assert rpu == 32
        assert "capacity_checkpoint" in reason

    def test_checkpoint_multiple_gaps(self):
        """Need 2x32 + 1x64, have nothing."""
        cp = CapacityCheckpoint(rel_time_s=50.0, min_rpus=(32, 32, 64))
        scaler, spin_up, _ = self._make_scaler(
            [cp], cluster_rpu_multiset={}
        )
        scaler.reconcile_checkpoints_up_to(50.0)
        assert spin_up.call_count == 3
        rpus_requested = sorted([c[0][1] for c in spin_up.call_args_list])
        assert rpus_requested == [32, 32, 64]

    def test_checkpoint_not_triggered_before_time(self):
        """Checkpoint at t=200 is not triggered by reconcile at t=100."""
        cp = CapacityCheckpoint(rel_time_s=200.0, min_rpus=(4, 32, 32))
        scaler, spin_up, _ = self._make_scaler(
            [cp], cluster_rpu_multiset={4: 1, 32: 1}
        )
        scaler.reconcile_checkpoints_up_to(100.0)
        spin_up.assert_not_called()
        # Now advance past 200 — should trigger.
        scaler.reconcile_checkpoints_up_to(200.0)
        spin_up.assert_called_once()

    def test_multiple_checkpoints_in_order(self):
        """Two checkpoints at different times, both triggered in one call."""
        cp1 = CapacityCheckpoint(rel_time_s=100.0, min_rpus=(8,))
        cp2 = CapacityCheckpoint(rel_time_s=200.0, min_rpus=(8, 32))
        scaler, spin_up, pool = self._make_scaler(
            [cp1, cp2], cluster_rpu_multiset={}
        )
        # After cp1 fires, the pool still reports {} (we'd need to
        # update the mock dynamically for full fidelity, but the
        # Autoscaler correctly processes them in order).
        scaler.reconcile_checkpoints_up_to(250.0)
        # cp1 gap: {8: 1}. cp2 gap: {8: 1, 32: 1} (pool still empty in mock).
        assert spin_up.call_count == 3

    def test_checkpoints_idempotent(self):
        """Calling reconcile twice at the same time doesn't re-trigger."""
        cp = CapacityCheckpoint(rel_time_s=100.0, min_rpus=(4, 32, 32))
        scaler, spin_up, _ = self._make_scaler(
            [cp], cluster_rpu_multiset={4: 1, 32: 1}
        )
        scaler.reconcile_checkpoints_up_to(100.0)
        spin_up.assert_called_once()
        spin_up.reset_mock()
        scaler.reconcile_checkpoints_up_to(100.0)
        spin_up.assert_not_called()

    def test_checkpoints_sorted_regardless_of_input_order(self):
        """Even if checkpoints are given out of order, earlier one fires first."""
        cp_late = CapacityCheckpoint(rel_time_s=300.0, min_rpus=(64,))
        cp_early = CapacityCheckpoint(rel_time_s=100.0, min_rpus=(32,))
        scaler, spin_up, _ = self._make_scaler(
            [cp_late, cp_early], cluster_rpu_multiset={}
        )
        scaler.reconcile_checkpoints_up_to(150.0)
        # Only cp_early should have fired.
        spin_up.assert_called_once()
        _, rpu = spin_up.call_args[0]
        assert rpu == 32

    def test_checkpoint_excess_capacity_no_action(self):
        """Pool has MORE than desired — no spin-ups."""
        cp = CapacityCheckpoint(rel_time_s=100.0, min_rpus=(4,))
        scaler, spin_up, _ = self._make_scaler(
            [cp], cluster_rpu_multiset={4: 2, 32: 1}
        )
        scaler.reconcile_checkpoints_up_to(100.0)
        spin_up.assert_not_called()

    def test_checkpoints_property(self):
        cp = CapacityCheckpoint(rel_time_s=100.0, min_rpus=(4, 32))
        scaler, _, _ = self._make_scaler([cp])
        assert scaler.checkpoints == [cp]
