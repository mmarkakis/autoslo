"""Tests for ``SpinUpBudget`` (see docs/max_clusters_design.md §5.2)."""

from __future__ import annotations

import pytest

from autoslo.clusters.spin_up_budget import SpinUpBudget


class TestSpinUpBudgetInit:
    def test_initial_state(self):
        b = SpinUpBudget(max_clusters=10)
        assert b.max_clusters == 10
        assert b.used == 0
        assert b.reserved == 0
        assert b.available == 10

    def test_zero_budget(self):
        b = SpinUpBudget(max_clusters=0)
        assert b.available == 0
        assert b.try_consume() is False

    def test_negative_max_clusters_raises(self):
        with pytest.raises(ValueError):
            SpinUpBudget(max_clusters=-1)


class TestSpinUpBudgetReserveRelease:
    def test_reserve_moves_available_to_reserved(self):
        b = SpinUpBudget(max_clusters=10)
        b.reserve(3)
        assert b.reserved == 3
        assert b.available == 7

    def test_reserve_too_much_raises(self):
        b = SpinUpBudget(max_clusters=5)
        with pytest.raises(ValueError):
            b.reserve(6)
        # State unchanged.
        assert b.reserved == 0
        assert b.available == 5

    def test_release_reservation_moves_back_to_available(self):
        b = SpinUpBudget(max_clusters=10)
        b.reserve(4)
        b.release_reservation(3)
        assert b.reserved == 1
        assert b.available == 9

    def test_release_reservation_caps_at_remaining(self):
        b = SpinUpBudget(max_clusters=10)
        b.reserve(2)
        b.release_reservation(100)  # idempotent over-release
        assert b.reserved == 0
        assert b.available == 10

    def test_negative_args_raise(self):
        b = SpinUpBudget(max_clusters=10)
        with pytest.raises(ValueError):
            b.reserve(-1)
        with pytest.raises(ValueError):
            b.release_reservation(-1)


class TestSpinUpBudgetConsume:
    def test_try_consume_succeeds_until_exhausted(self):
        b = SpinUpBudget(max_clusters=3)
        assert b.try_consume() is True
        assert b.try_consume() is True
        assert b.try_consume() is True
        assert b.try_consume() is False
        assert b.used == 3
        assert b.available == 0

    def test_try_consume_does_not_draw_from_reserved(self):
        b = SpinUpBudget(max_clusters=5)
        b.reserve(3)  # available=2, reserved=3
        assert b.try_consume() is True
        assert b.try_consume() is True
        assert (
            b.try_consume() is False
        )  # available exhausted; reserved untouched
        assert b.reserved == 3

    def test_try_consume_reserved_only_draws_from_reserved(self):
        b = SpinUpBudget(max_clusters=5)
        b.reserve(2)  # available=3, reserved=2
        assert b.try_consume_reserved() is True
        assert b.try_consume_reserved() is True
        assert b.try_consume_reserved() is False
        assert b.available == 3  # untouched

    def test_invariant_holds_throughout(self):
        b = SpinUpBudget(max_clusters=10)
        snap = b.snapshot()
        assert snap["used"] + snap["reserved"] + snap["available"] == 10
        b.reserve(4)
        snap = b.snapshot()
        assert snap["used"] + snap["reserved"] + snap["available"] == 10
        b.try_consume()
        b.try_consume_reserved()
        b.release_reservation(1)
        snap = b.snapshot()
        assert snap["used"] + snap["reserved"] + snap["available"] == 10
        assert snap["used"] == 2
        assert snap["reserved"] == 2
        assert snap["available"] == 6
