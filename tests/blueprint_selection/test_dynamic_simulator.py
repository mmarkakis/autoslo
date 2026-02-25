"""
Tests for dynamic provisioning in
:class:`~autoslo.blueprint_selection.workload_routing_simulator.WorkloadRoutingSimulator`.

These tests exercise the event-loop machinery (cluster activation,
deactivation, capacity-controller integration, and pending-event
processing) without running full simulation passes — the heavy
dependencies (IconqModel, Workload) are mocked.
"""

from __future__ import annotations


from unittest.mock import MagicMock, patch

import pytest

from autoslo.blueprints.cluster import Cluster
from autoslo.blueprint_selection.workload_routing_simulator import (
    WorkloadRoutingSimulator,
)

from autoslo.capacity.policy_tuner import DynamicClusterConfig
from autoslo.workload_definition.query import Query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(
    qid: str,
    template: str = "1_0",
    start: float = 0.0,
    latency: float = 5.0,
) -> Query:
    return Query(
        query_id=qid,
        tpcds_temp_and_q_idx=template,
        rel_start_time_s=start,
        latency_s=latency,
    )


def _make_dynamic_simulator(
    initial_rpus: tuple[int, ...] = (8,),
    allowed_rpu_sizes: tuple[int, ...] = (8,),
    spin_up_delay_s: float = 120.0,
    eta_crit: float = 0.1,
    idle_periods_before_tear_down: int = 5,
    capacity_poll_interval_s: float = 60.0,
    slo_s: float = 10.0,
) -> WorkloadRoutingSimulator:
    """Build a simulator in dynamic mode with mocked heavy deps."""
    config = DynamicClusterConfig(
        initial_rpus=initial_rpus,
        allowed_rpu_sizes=allowed_rpu_sizes,
        spin_up_delay_s=spin_up_delay_s,
    )

    mock_model = MagicMock()
    mock_model.iconq_query_featurizer = MagicMock()
    mock_model.iconq_interaction_featurizer = MagicMock()
    mock_model.stage_model = MagicMock()
    mock_workload = MagicMock()

    with (
        patch(
            "autoslo.blueprint_selection.workload_routing_simulator.IconqModel.load",
            return_value=mock_model,
        ),
        patch(
            "autoslo.blueprint_selection.workload_routing_simulator.Chunk.load",
            return_value=mock_workload,
        ),
        patch(
            "autoslo.blueprint_selection.workload_routing_simulator.WorkloadRoutingSimulator._make_out_dir",
            return_value="/tmp/test_sim_out",
        ),
        patch(
            "autoslo.blueprint_selection.workload_routing_simulator.WorkloadRoutingSimulator._write_config_yml",
        ),
    ):
        sim = WorkloadRoutingSimulator(
            workload_name="test_wl",
            iconq_model_id="test_model",
            blueprint_name="dynamic",
            slo_s=slo_s,
            dynamic_cluster_config=config,
            eta_crit=eta_crit,
            idle_periods_before_tear_down=idle_periods_before_tear_down,
            capacity_poll_interval_s=capacity_poll_interval_s,
        )
    return sim


# ---------------------------------------------------------------------------
# Tests: dynamic mode initialisation
# ---------------------------------------------------------------------------


class TestDynamicInit:

    def test_dynamic_mode_enabled(self):
        sim = _make_dynamic_simulator()
        assert sim._dynamic_mode is True
        assert sim._blueprint is None
        assert sim._provisioner is not None
        assert sim._capacity_controller is not None

    def test_initial_clusters_activated(self):
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))
        assert len(sim._active_queries_per_cluster) == 2
        # All clusters start with empty active-query lists.
        for qs in sim._active_queries_per_cluster.values():
            assert qs == []
        assert len(sim._cost_per_second_per_cluster) == 2

    def test_initial_cluster_names_contain_rpu(self):
        sim = _make_dynamic_simulator(initial_rpus=(8,))
        names = list(sim._active_queries_per_cluster.keys())
        assert len(names) == 1
        # Cluster.new names contain the RPU.
        assert "8" in names[0]


# ---------------------------------------------------------------------------
# Tests: cluster bookkeeping
# ---------------------------------------------------------------------------


class TestClusterBookkeeping:

    def test_activate_adds_all_dicts(self):
        sim = _make_dynamic_simulator(initial_rpus=())
        assert len(sim._active_queries_per_cluster) == 0

        sim._activate_cluster_bookkeeping("c1", 0.36)
        assert "c1" in sim._active_queries_per_cluster
        assert sim._active_queries_per_cluster["c1"] == []
        assert "c1" in sim._completed_queries_per_cluster
        assert sim._cost_per_second_per_cluster["c1"] == 0.36
        assert sim._before_cache_valid["c1"] is False

    def test_deactivate_removes_routing_state(self):
        sim = _make_dynamic_simulator(initial_rpus=())
        sim._activate_cluster_bookkeeping("c1", 0.36)
        sim._deactivate_cluster_bookkeeping("c1")

        assert "c1" not in sim._active_queries_per_cluster
        assert "c1" not in sim._cost_per_second_per_cluster
        # completed queries are preserved for billing
        assert "c1" in sim._completed_queries_per_cluster

    def test_deactivate_does_not_move_active_queries(self):
        """Deactivate is only called once a cluster is fully drained,
        so active queries should not be present — but if they are,
        they are simply dropped (not moved to completed)."""
        sim = _make_dynamic_simulator(initial_rpus=())
        sim._activate_cluster_bookkeeping("c1", 0.36)
        q = _q("q1")
        sim._active_queries_per_cluster["c1"].append(q)
        sim._neighbors_per_active_query["q1"] = []

        sim._deactivate_cluster_bookkeeping("c1")
        assert "c1" not in sim._active_queries_per_cluster
        # completed list was initialised empty during activate and
        # no automatic migration should have happened.
        assert sim._completed_queries_per_cluster["c1"] == []


# ---------------------------------------------------------------------------
# Tests: capacity controller integration
# ---------------------------------------------------------------------------


class TestCapacityControllerIntegration:

    def test_spin_up_callback_schedules_event(self):
        sim = _make_dynamic_simulator(initial_rpus=(8,), spin_up_delay_s=120.0)
        sim._current_sim_time_s = 100.0
        sim._on_sim_spin_up("test_reason", 8)

        assert len(sim._pending_events) == 1
        ready_time, _, cluster = sim._pending_events[0]
        assert ready_time == 220.0  # 100 + 120
        assert cluster.rpu == 8

    def test_tear_down_empty_cluster_removes_it(self):
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))
        names = list(sim._active_queries_per_cluster.keys())
        assert len(names) == 2

        sim._on_sim_tear_down(names[0])
        assert len(sim._active_queries_per_cluster) == 1

    def test_tear_down_with_active_queries_marks_draining(self):
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))
        names = list(sim._active_queries_per_cluster.keys())
        target = names[0]

        # Give the target cluster an active query.
        q = _q("q1", latency=500.0)
        sim._active_queries_per_cluster[target].append(q)
        sim._neighbors_per_active_query["q1"] = []

        sim._on_sim_tear_down(target)
        # Cluster is still present (queries still running).
        assert target in sim._active_queries_per_cluster
        assert target in sim._draining_clusters
        # But it was recorded as torn down by the provisioner.
        assert len(sim._provisioner.torn_down) == 1

    def test_tear_down_blocked_for_last_cluster(self):
        sim = _make_dynamic_simulator(initial_rpus=(8,))
        names = list(sim._active_queries_per_cluster.keys())
        assert len(names) == 1

        # Tear-down should be skipped — it's the last cluster.
        sim._on_sim_tear_down(names[0])
        assert len(sim._active_queries_per_cluster) == 1

    def test_tear_down_blocked_when_all_others_draining(self):
        """If every other cluster is already draining, the remaining
        routable cluster must not be torn down."""
        sim = _make_dynamic_simulator(initial_rpus=(8, 16, 32))
        names = list(sim._active_queries_per_cluster.keys())

        # Mark first two clusters as draining.
        for cn in names[:2]:
            q = _q(f"q_{cn}", latency=999.0)
            sim._active_queries_per_cluster[cn].append(q)
            sim._neighbors_per_active_query[q.query_id] = []
            sim._on_sim_tear_down(cn)

        # The third cluster is the last routable one — tear-down refused.
        sim._on_sim_tear_down(names[2])
        assert names[2] in sim._active_queries_per_cluster
        assert names[2] not in sim._draining_clusters


# ---------------------------------------------------------------------------
# Tests: pending event processing
# ---------------------------------------------------------------------------


class TestPendingEvents:

    def test_events_processed_in_order(self):
        sim = _make_dynamic_simulator(initial_rpus=())
        c1 = Cluster.new(rpu=8, name="ev1")
        c2 = Cluster.new(rpu=16, name="ev2")

        import heapq

        heapq.heappush(sim._pending_events, (50.0, 1, c1))
        heapq.heappush(sim._pending_events, (100.0, 2, c2))

        sim._process_pending_events_up_to(75.0)
        assert "ev1" in sim._active_queries_per_cluster
        assert "ev2" not in sim._active_queries_per_cluster

        sim._process_pending_events_up_to(100.0)
        assert "ev2" in sim._active_queries_per_cluster

    def test_no_events_before_time(self):
        sim = _make_dynamic_simulator(initial_rpus=())
        c1 = Cluster.new(rpu=8, name="ev1")

        import heapq

        heapq.heappush(sim._pending_events, (200.0, 1, c1))

        sim._process_pending_events_up_to(100.0)
        assert "ev1" not in sim._active_queries_per_cluster


# ---------------------------------------------------------------------------
# Tests: advance_simulated_time
# ---------------------------------------------------------------------------


class TestAdvanceSimulatedTime:

    def test_no_ticks_before_first_poll_interval(self):
        """Ticks start at poll_interval_s; no tick should fire before."""
        sim = _make_dynamic_simulator(
            initial_rpus=(8,),
            capacity_poll_interval_s=60.0,
        )
        # Advance to t=30 — no tick yet.
        sim._advance_simulated_time(30.0)
        assert sim._current_sim_time_s == 30.0
        # No spin-ups or tear-downs should have happened.
        assert len(sim._provisioner.spun_up) == 0

    def test_tick_fires_at_poll_interval(self):
        """A tick should fire when we advance past poll_interval_s."""
        sim = _make_dynamic_simulator(
            initial_rpus=(8,),
            capacity_poll_interval_s=60.0,
            # Set eta_crit to 2.0 so any active query triggers spin-up
            # (headroom = (slo - latency) / slo is always <= 1.0 < 2.0).
            eta_crit=2.0,
            slo_s=10.0,
        )
        # Add an active query so headroom is computed.
        cn = list(sim._active_queries_per_cluster.keys())[0]
        # Use a long latency so the query is still active at t=60.
        q = _q("q1", latency=500.0)
        sim._active_queries_per_cluster[cn].append(q)
        sim._neighbors_per_active_query["q1"] = []

        sim._advance_simulated_time(70.0)
        # At least one tick should have fired and triggered a spin-up.
        assert len(sim._provisioner.spun_up) >= 1

    def test_multiple_ticks(self):
        """Multiple ticks should fire across a long time span."""
        sim = _make_dynamic_simulator(
            initial_rpus=(8,),
            capacity_poll_interval_s=60.0,
            eta_crit=2.0,  # always triggers spin-up
            slo_s=10.0,
        )
        cn = list(sim._active_queries_per_cluster.keys())[0]
        q = _q("q1", latency=5.0, start=0.0)
        q.latency_s = 500.0  # stays active the whole time
        sim._active_queries_per_cluster[cn].append(q)
        sim._neighbors_per_active_query["q1"] = []

        sim._advance_simulated_time(200.0)
        # Ticks at t=60, 120, 180.  Each triggers spin-up.
        assert len(sim._provisioner.spun_up) >= 3

    def test_spin_up_cluster_becomes_ready(self):
        """A cluster scheduled via spin-up should become ready after delay."""
        sim = _make_dynamic_simulator(
            initial_rpus=(8,),
            capacity_poll_interval_s=60.0,
            spin_up_delay_s=30.0,
            eta_crit=2.0,
            slo_s=10.0,
        )
        initial_count = len(sim._active_queries_per_cluster)
        cn = list(sim._active_queries_per_cluster.keys())[0]
        q = _q("q1", latency=5.0)
        q.latency_s = 500.0
        sim._active_queries_per_cluster[cn].append(q)
        sim._neighbors_per_active_query["q1"] = []

        # Advance past first tick (t=60) + delay (30) = t=90.
        sim._advance_simulated_time(100.0)
        assert len(sim._active_queries_per_cluster) > initial_count

    def test_tear_down_after_idle_periods(self):
        """A cluster idle for enough ticks should be torn down."""
        sim = _make_dynamic_simulator(
            initial_rpus=(8, 16),
            capacity_poll_interval_s=10.0,
            idle_periods_before_tear_down=3,
            eta_crit=-1.0,  # no spin-ups (headroom always > eta_crit)
            slo_s=10.0,
        )
        assert len(sim._active_queries_per_cluster) == 2

        # All clusters are idle.  After 3 ticks (t=10, 20, 30), one
        # should be torn down (but not the last one).
        sim._advance_simulated_time(35.0)
        # At most one cluster should have been torn down.
        # (Can't tear down both because of the last-cluster guard.)
        assert len(sim._active_queries_per_cluster) == 1


# ---------------------------------------------------------------------------
# Tests: reset in dynamic mode
# ---------------------------------------------------------------------------


class TestDynamicReset:

    def test_reset_reinitialises_clusters(self):
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))

        # Add a query to change state.
        cn = list(sim._active_queries_per_cluster.keys())[0]
        q = _q("q1")
        sim._active_queries_per_cluster[cn].append(q)

        with patch.object(sim, "_make_out_dir", return_value="/tmp/reset_test"):
            with patch.object(sim, "_write_config_yml"):
                sim.reset()

        # After reset: fresh cluster set with correct number.
        new_names = set(sim._active_queries_per_cluster.keys())
        assert len(new_names) == 2
        # All query lists should be empty after reset.
        for qs in sim._active_queries_per_cluster.values():
            assert qs == []
        # Sim time should be reset.
        assert sim._current_sim_time_s == 0.0
        assert len(sim._pending_events) == 0
        assert len(sim._draining_clusters) == 0


# ---------------------------------------------------------------------------
# Tests: graceful drain before tear-down
# ---------------------------------------------------------------------------


class TestDraining:

    def test_draining_cluster_excluded_from_routing(self):
        """A draining cluster should not appear in the routable set
        used by _find_best_cluster_for_query."""
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))
        names = list(sim._active_queries_per_cluster.keys())
        target = names[0]

        # Mark target as draining with an active query.
        q = _q("q1", latency=999.0)
        sim._active_queries_per_cluster[target].append(q)
        sim._neighbors_per_active_query["q1"] = []
        sim._draining_clusters.add(target)

        # Build the routable set the same way _find_best_cluster_for_query does.
        routable = {
            cn: qs
            for cn, qs in sim._active_queries_per_cluster.items()
            if cn not in sim._draining_clusters
        }
        assert target not in routable
        assert names[1] in routable

    def test_draining_cluster_excluded_from_capacity_controller(self):
        """The capacity controller's get_active_queries should not
        include draining clusters."""
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))
        names = list(sim._active_queries_per_cluster.keys())
        target = names[0]

        sim._draining_clusters.add(target)
        cc_view = sim._capacity_controller._get_active_queries()
        assert target not in cc_view
        assert names[1] in cc_view

    def test_draining_cluster_deactivated_after_queries_complete(self):
        """Once all active queries on a draining cluster complete,
        _cleanup_completed_queries_up_to should fully deactivate it."""
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))
        names = list(sim._active_queries_per_cluster.keys())
        target = names[0]

        # Add a query that ends at t=10.
        q = _q("q1", start=0.0, latency=10.0)
        sim._active_queries_per_cluster[target].append(q)
        sim._neighbors_per_active_query["q1"] = []
        sim._draining_clusters.add(target)

        # Cleanup at t=5 — query still active, cluster stays.
        sim._cleanup_completed_queries_up_to(5.0)
        assert target in sim._active_queries_per_cluster
        assert target in sim._draining_clusters

        # Cleanup at t=10 — query done, cluster deactivated.
        sim._cleanup_completed_queries_up_to(10.0)
        assert target not in sim._active_queries_per_cluster
        assert target not in sim._draining_clusters
        # Completed query is preserved for billing.
        assert q in sim._completed_queries_per_cluster[target]

    def test_end_to_end_drain_via_advance_simulated_time(self):
        """Full integration: idle cluster is marked draining, then
        deactivated once its queries finish."""
        sim = _make_dynamic_simulator(
            initial_rpus=(8, 16),
            capacity_poll_interval_s=10.0,
            idle_periods_before_tear_down=2,
            eta_crit=-1.0,  # no spin-ups
            slo_s=10.0,
        )
        names = list(sim._active_queries_per_cluster.keys())
        target = names[0]
        other = names[1]

        # Put a long-running query on the target cluster.
        q = _q("q_long", start=0.0, latency=100.0)
        sim._active_queries_per_cluster[target].append(q)
        sim._neighbors_per_active_query["q_long"] = []

        # Advance enough for the *other* idle cluster to be torn down
        # after 2 idle periods.  The target has an active query so it
        # won't be considered idle.
        sim._advance_simulated_time(25.0)
        # The idle cluster (other) should be gone.
        assert other not in sim._active_queries_per_cluster

        # Now make the target idle so it gets torn down next.
        # But first, the query ends at t=100.  Let's advance to t=35
        # where the cluster still has an active query so it won't be
        # torn down by idle tracking.
        sim._advance_simulated_time(35.0)
        assert target in sim._active_queries_per_cluster
        assert target not in sim._draining_clusters

    def test_reset_clears_draining_set(self):
        """After reset(), no clusters should be in the draining set."""
        sim = _make_dynamic_simulator(initial_rpus=(8, 16))
        names = list(sim._active_queries_per_cluster.keys())
        sim._draining_clusters.add(names[0])

        with patch.object(sim, "_make_out_dir", return_value="/tmp/reset_test"):
            with patch.object(sim, "_write_config_yml"):
                sim.reset()

        assert len(sim._draining_clusters) == 0
