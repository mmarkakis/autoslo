"""
Tests for scheduled spin-up execution.

Replaces the old CapacityCheckpoint reconciliation tests with tests for the
new ScheduledSpinUp imperative model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autoslo.clusters.actions import SpinUpAction
from autoslo.clusters.scheduled_spinup import ScheduledSpinUp


# ===================================================================
# ScheduledSpinUp data-type tests
# ===================================================================


class TestScheduledSpinUpDataType:
    def test_fields_accessible(self):
        su = ScheduledSpinUp(rel_time_s=180.0, rpu=8)
        assert su.rel_time_s == 180.0
        assert su.rpu == 8

    def test_frozen(self):
        su = ScheduledSpinUp(rel_time_s=180.0, rpu=8)
        with pytest.raises(AttributeError):
            su.rel_time_s = 999.0  # type: ignore[misc]

    def test_equality(self):
        a = ScheduledSpinUp(rel_time_s=100.0, rpu=32)
        b = ScheduledSpinUp(rel_time_s=100.0, rpu=32)
        assert a == b

    def test_inequality_different_rpu(self):
        a = ScheduledSpinUp(rel_time_s=100.0, rpu=8)
        b = ScheduledSpinUp(rel_time_s=100.0, rpu=32)
        assert a != b


# ===================================================================
# ScheduledSpinUp.from_config
# ===================================================================


class TestScheduledSpinUpFromConfig:
    def test_empty_config(self):
        assert ScheduledSpinUp.from_config({}) == []

    def test_parses_list(self):
        cfg = {
            "scheduled_spinups": [
                {"rel_time_s": 100.0, "rpu": 8},
                {"rel_time_s": 200.0, "rpu": 32},
            ]
        }
        result = ScheduledSpinUp.from_config(cfg)
        assert result == [
            ScheduledSpinUp(rel_time_s=100.0, rpu=8),
            ScheduledSpinUp(rel_time_s=200.0, rpu=32),
        ]

    def test_returns_frozen_instances(self):
        cfg = {"scheduled_spinups": [{"rel_time_s": 50.0, "rpu": 4}]}
        result = ScheduledSpinUp.from_config(cfg)
        with pytest.raises(AttributeError):
            result[0].rpu = 999  # type: ignore[misc]


# ===================================================================
# ScheduledSpinUp.total_spinups
# ===================================================================


class TestScheduledSpinUpTotalSpinups:
    def test_empty(self):
        assert ScheduledSpinUp.total_spinups([]) == 0

    def test_single(self):
        su = ScheduledSpinUp(rel_time_s=100.0, rpu=8)
        assert ScheduledSpinUp.total_spinups([su]) == 1

    def test_multiple(self):
        spinups = [
            ScheduledSpinUp(rel_time_s=100.0, rpu=8),
            ScheduledSpinUp(rel_time_s=200.0, rpu=32),
            ScheduledSpinUp(rel_time_s=300.0, rpu=16),
        ]
        assert ScheduledSpinUp.total_spinups(spinups) == 3


# ===================================================================
# ScheduledSpinUp.execute
# ===================================================================


class TestScheduledSpinUpExecute:
    def test_execute_calls_on_spin_up_once(self):
        """execute() must call on_spin_up exactly once."""
        su = ScheduledSpinUp(rel_time_s=100.0, rpu=8)
        on_spin_up = MagicMock()

        su.execute(source="test", on_spin_up=on_spin_up)

        on_spin_up.assert_called_once()

    def test_execute_passes_correct_rpu(self):
        """The SpinUpAction passed to on_spin_up must carry the correct rpu."""
        su = ScheduledSpinUp(rel_time_s=100.0, rpu=32)
        on_spin_up = MagicMock()

        su.execute(source="test", on_spin_up=on_spin_up)

        action: SpinUpAction = on_spin_up.call_args[0][0]
        assert action.rpu == 32

    def test_execute_marks_from_reserved_budget(self):
        """Spin-ups from scheduled spin-ups must draw from the reserved budget."""
        su = ScheduledSpinUp(rel_time_s=50.0, rpu=16)
        on_spin_up = MagicMock()

        su.execute(source="test", on_spin_up=on_spin_up)

        action: SpinUpAction = on_spin_up.call_args[0][0]
        assert action.from_reserved_budget is True

    def test_execute_reason_contains_time(self):
        """The SpinUpAction reason must mention the scheduled rel_time_s."""
        su = ScheduledSpinUp(rel_time_s=999.5, rpu=8)
        on_spin_up = MagicMock()

        su.execute(source="test", on_spin_up=on_spin_up)

        action: SpinUpAction = on_spin_up.call_args[0][0]
        assert "999.5" in action.reason

    def test_two_spinups_execute_independently(self):
        """Two ScheduledSpinUp instances each call on_spin_up once with their rpu."""
        su1 = ScheduledSpinUp(rel_time_s=100.0, rpu=8)
        su2 = ScheduledSpinUp(rel_time_s=200.0, rpu=32)
        on_spin_up = MagicMock()

        su1.execute(source="test", on_spin_up=on_spin_up)
        su2.execute(source="test", on_spin_up=on_spin_up)

        assert on_spin_up.call_count == 2
        rpus = [on_spin_up.call_args_list[i][0][0].rpu for i in range(2)]
        assert sorted(rpus) == [8, 32]
