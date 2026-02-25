"""
Tests for :mod:`autoslo.capacity.capacity_controller`.

The controller's core logic is deterministic and synchronous via
``tick_once()``, so no background threads are needed for unit testing.
"""

from __future__ import annotations

import pytest

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.capacity.capacity_controller import CapacityController
from autoslo.routing.routing_core import RoutingCore
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


def _resolver(default: float = 10.0) -> SloResolver:
    return SloResolver.from_dict(default_slo_s=default, slo_dict={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCapacityControllerSpinUp:

    def test_no_spin_up_when_healthy(self):
        """Plenty of headroom → no spin-up."""
        slo_s = 10.0
        free_frac = 0.5
        eta = 0.2

        active = {"c0": [_q("a", latency=slo_s * (1 - free_frac))]}
        spin_ups: list[tuple[str, int]] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(slo_s),
            on_spin_up=lambda reason, rpu: spin_ups.append((reason, rpu)),
            eta_crit=eta,
        )
        ctrl.tick_once()
        assert spin_ups == []

    def test_spin_up_when_headroom_critical(self):
        """Headroom below η_crit → spin-up fires."""
        slo_s = 10.0
        free_frac = 0.1
        eta = 0.2

        active = {"c0": [_q("a", latency=slo_s * (1 - free_frac))]}
        spin_ups: list[tuple[str, int]] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(slo_s),
            on_spin_up=lambda reason, rpu: spin_ups.append((reason, rpu)),
            eta_crit=eta,
        )
        ctrl.tick_once()
        assert len(spin_ups) == 1
        assert "headroom" in spin_ups[0][0]

    def test_spin_up_on_pressure_signal(self):
        """Pressure flag → spin-up even if headroom is fine."""
        slo_s = 10.0
        free_frac = 0.5
        eta = 0.2

        active = {"c0": [_q("a", latency=slo_s * (1 - free_frac))]}
        spin_ups: list[tuple[str, int]] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(slo_s),
            on_spin_up=lambda reason, rpu: spin_ups.append((reason, rpu)),
            eta_crit=eta,
        )
        ctrl.tick_once(pressure=True)
        assert len(spin_ups) == 1
        assert "pressure" in spin_ups[0][0]

    def test_spin_up_when_no_queries(self):
        """Empty system → headroom=1.0 → no spin-up."""
        slo_s = 10.0
        eta = 0.2

        active: dict[str, list[Query]] = {"c0": []}
        spin_ups: list[tuple[str, int]] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(slo_s),
            on_spin_up=lambda reason, rpu: spin_ups.append((reason, rpu)),
            eta_crit=eta,
        )
        ctrl.tick_once()
        assert spin_ups == []


class TestCapacityControllerTearDown:

    def test_no_tear_down_when_active(self):
        """Cluster with running queries → not idle → no tear-down."""
        idle_periods_before_tear_down = 3

        active = {"c0": [_q("a")]}
        tear_downs: list[str] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(),
            on_tear_down=lambda cn: tear_downs.append(cn),
            idle_periods_before_tear_down=idle_periods_before_tear_down,
        )
        for _ in range(2 * idle_periods_before_tear_down):
            ctrl.tick_once()
        assert tear_downs == []

    def test_tear_down_after_idle_periods(self):
        """Cluster idle for N consecutive ticks → tear-down fires once."""
        idle_periods_before_tear_down = 3

        active: dict[str, list[Query]] = {"c0": [], "c1": [_q("a")]}
        tear_downs: list[str] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(),
            on_tear_down=lambda cn: tear_downs.append(cn),
            idle_periods_before_tear_down=idle_periods_before_tear_down,
        )
        # Tick: c0 is idle each time.
        for _ in range(idle_periods_before_tear_down):
            ctrl.tick_once()
        assert tear_downs == ["c0"]

    def test_idle_counter_resets_on_activity(self):
        """If a cluster gets work mid-way, idle counter resets."""

        idle_periods_before_tear_down = 3

        call_count = [0]
        active: dict[str, list[Query]] = {"c0": []}
        tear_downs: list[str] = []

        def get_qs() -> dict[str, list[Query]]:
            call_count[0] += 1
            if call_count[0] == idle_periods_before_tear_down:
                return {"c0": [_q("a")]}
            return {"c0": []}

        ctrl = CapacityController(
            get_active_queries=get_qs,
            slo_resolver=_resolver(),
            on_tear_down=lambda cn: tear_downs.append(cn),
            idle_periods_before_tear_down=idle_periods_before_tear_down,
        )
        for _ in range(2 * idle_periods_before_tear_down):
            ctrl.tick_once()
        # Should fire once: idle ticks 4,5,6 (3 consecutive).
        assert tear_downs == ["c0"]

    def test_tear_down_resets_after_firing(self):
        """After firing, the idle counter resets so it doesn't fire
        immediately on the next tick."""
        idle_periods_before_tear_down = 2

        active: dict[str, list[Query]] = {"c0": []}
        tear_downs: list[str] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(),
            on_tear_down=lambda cn: tear_downs.append(cn),
            idle_periods_before_tear_down=idle_periods_before_tear_down,
        )
        # 4 ticks → should fire at tick 2, reset, fire again at tick 4
        for _ in range(2 * idle_periods_before_tear_down):
            ctrl.tick_once()
        assert tear_downs == ["c0", "c0"]


class TestCapacityControllerParameters:

    def test_eta_crit_settable(self):
        ctrl = CapacityController(
            get_active_queries=lambda: {},
            slo_resolver=_resolver(),
            eta_crit=0.2,
        )
        assert ctrl.eta_crit == 0.2
        ctrl.eta_crit = 0.05
        assert ctrl.eta_crit == 0.05

    def test_idle_periods_settable(self):
        ctrl = CapacityController(
            get_active_queries=lambda: {},
            slo_resolver=_resolver(),
            idle_periods_before_tear_down=10,
        )
        assert ctrl.idle_periods_before_tear_down == 10
        ctrl.idle_periods_before_tear_down = 3
        assert ctrl.idle_periods_before_tear_down == 3


class TestCapacityControllerBackgroundThread:

    def test_start_stop(self):
        """Controller starts and stops cleanly."""
        ctrl = CapacityController(
            get_active_queries=lambda: {},
            slo_resolver=_resolver(),
            poll_interval_s=0.05,
        )
        ctrl.start()
        assert ctrl.is_running
        ctrl.stop(timeout=2.0)
        assert not ctrl.is_running

    def test_signal_wakes_controller(self):
        """Calling signal_capacity_pressure wakes the controller
        before the poll interval elapses."""
        import time

        spin_ups: list[tuple[str, int]] = []
        # Very long poll interval — should not fire within 1 second
        # unless the pressure signal wakes it.
        ctrl = CapacityController(
            get_active_queries=lambda: {"c0": [_q("a", latency=9.9)]},
            slo_resolver=_resolver(10.0),
            on_spin_up=lambda reason, rpu: spin_ups.append((reason, rpu)),
            poll_interval_s=600.0,  # 10 minutes
            eta_crit=0.1,
        )
        ctrl.start()
        time.sleep(0.05)  # let thread start
        ctrl.signal_capacity_pressure()
        time.sleep(0.2)  # let tick process
        ctrl.stop(timeout=2.0)
        # Should have fired because of the pressure signal.
        assert len(spin_ups) >= 1


class TestCapacityControllerRPUSelection:

    def test_spin_up_passes_smallest_rpu(self):
        """on_spin_up receives the smallest allowed RPU."""
        active = {"c0": [_q("a", latency=9.5)]}
        spin_ups: list[tuple[str, int]] = []

        ctrl = CapacityController(
            get_active_queries=lambda: active,
            slo_resolver=_resolver(10.0),
            on_spin_up=lambda reason, rpu: spin_ups.append((reason, rpu)),
            eta_crit=0.2,
            allowed_rpu_sizes=[32, 16, 8],
        )
        ctrl.tick_once()
        assert len(spin_ups) == 1
        assert spin_ups[0][1] == 8  # smallest

    def test_default_rpu_sizes(self):
        ctrl = CapacityController(
            get_active_queries=lambda: {},
            slo_resolver=_resolver(),
        )
        assert ctrl.allowed_rpu_sizes == [8]

    def test_allowed_rpu_sizes_settable(self):
        ctrl = CapacityController(
            get_active_queries=lambda: {},
            slo_resolver=_resolver(),
            allowed_rpu_sizes=[4, 16],
        )
        assert ctrl.allowed_rpu_sizes == [4, 16]
        ctrl.allowed_rpu_sizes = [32, 8]
        assert ctrl.allowed_rpu_sizes == [8, 32]  # sorted
