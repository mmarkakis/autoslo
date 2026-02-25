"""
Tests for :mod:`autoslo.forecasting.recurring_pattern`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from autoslo.forecasting.recurring_pattern import RecurringPatternExtractor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log(
    template_id: str,
    datetimes: list[datetime],
) -> pd.DataFrame:
    """Build a minimal query log DataFrame."""
    return pd.DataFrame({"timestamp": datetimes, "template_id": template_id})


def _concat(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True)


def _repeated_weekly(
    template_id: str,
    base: datetime,
    n_weeks: int,
    *,
    hour: int | None = None,
    minute: int = 0,
) -> pd.DataFrame:
    """Create a template that appears at the same day-of-week & time each week.

    Uses ``base`` to determine the day-of-week; overrides hour/minute
    if provided.
    """
    if hour is not None:
        base = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    times = [base + timedelta(weeks=w) for w in range(n_weeks)]
    return _make_log(template_id, times)


# ---------------------------------------------------------------------------
# Tests: weekly_slot helper
# ---------------------------------------------------------------------------


class TestWeeklySlot:

    def test_monday_midnight_15min(self):
        """Monday 00:00 with 15-min slots → slot 0."""
        dt = datetime(2026, 3, 2, 0, 0)  # Monday
        assert dt.weekday() == 0
        extractor = RecurringPatternExtractor(slot_minutes=15)
        assert extractor._weekly_slot(dt) == 0

    def test_monday_midnight_60min(self):
        """Monday 00:00 with 60-min slots → slot 0."""
        dt = datetime(2026, 3, 2, 0, 0)
        extractor = RecurringPatternExtractor(slot_minutes=60)
        assert extractor._weekly_slot(dt) == 0

    def test_monday_0015_15min(self):
        """Monday 00:15 with 15-min slots → slot 1."""
        dt = datetime(2026, 3, 2, 0, 15)
        extractor = RecurringPatternExtractor(slot_minutes=15)
        assert extractor._weekly_slot(dt) == 1

    def test_tuesday_0800_15min(self):
        """Tuesday 08:00 → slot = 1*96 + 8*4 = 128."""
        dt = datetime(2026, 3, 3, 8, 0)  # Tuesday
        assert dt.weekday() == 1
        extractor = RecurringPatternExtractor(slot_minutes=15)
        assert extractor._weekly_slot(dt) == 96 + 32

    def test_sunday_2345_15min(self):
        """Sunday 23:45 → last slot = 6*96 + 23*4 + 3 = 671."""
        dt = datetime(2026, 3, 8, 23, 45)  # Sunday
        assert dt.weekday() == 6
        extractor = RecurringPatternExtractor(slot_minutes=15)
        assert extractor._weekly_slot(dt) == 671

    def test_n_weekly_slots_15min(self):
        extractor = RecurringPatternExtractor(slot_minutes=15)
        assert extractor._n_weekly_slots(15) == 672

    def test_n_weekly_slots_60min(self):
        extractor = RecurringPatternExtractor(slot_minutes=60)
        assert extractor._n_weekly_slots(60) == 168


# ---------------------------------------------------------------------------
# Tests: RecurringPatternExtractor construction
# ---------------------------------------------------------------------------


class TestRecurringPatternExtractorInit:

    def test_bad_slot_minutes(self):
        with pytest.raises(ValueError, match="integer number of slots"):
            RecurringPatternExtractor(slot_minutes=7)

    def test_bad_threshold_zero(self):
        with pytest.raises(ValueError, match="must be in"):
            RecurringPatternExtractor(reliability_threshold=0.0)

    def test_bad_threshold_over_one(self):
        with pytest.raises(ValueError, match="must be in"):
            RecurringPatternExtractor(reliability_threshold=1.5)

    def test_valid_construction(self):
        ext = RecurringPatternExtractor(
            slot_minutes=30, reliability_threshold=0.9
        )
        assert ext.slot_minutes == 30
        assert ext.reliability_threshold == 0.9

    def test_result_before_fit_raises(self):
        ext = RecurringPatternExtractor()
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = ext.result


# ---------------------------------------------------------------------------
# Tests: fit()
# ---------------------------------------------------------------------------


class TestRecurringPatternExtractorFit:

    def test_perfectly_reliable_template(self):
        """Template appearing every week at the same time → reliability 1.0."""
        base = datetime(2026, 3, 2, 8, 0)  # Monday
        log = _repeated_weekly("q42", base, n_weeks=5)

        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        result = ext.fit(log)

        assert "q42" in result.recurring_templates
        rt = result.recurring_templates["q42"]
        expected_slot = ext._weekly_slot(base)
        assert expected_slot in rt.reliable_slots
        assert rt.reliable_slots[expected_slot] == pytest.approx(1.0)
        assert rt.expected_count_per_slot[expected_slot] == pytest.approx(1.0)

    def test_unreliable_template_excluded(self):
        """Template appearing only 1 of 5 weeks → reliability 0.2 < 0.8."""
        base = datetime(2026, 3, 2, 8, 0)
        log = _repeated_weekly("q99", base, n_weeks=1)
        filler = _repeated_weekly("q1", base, n_weeks=5)
        combined = _concat(log, filler)

        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        result = ext.fit(combined)

        assert "q99" not in result.recurring_templates
        assert "q1" in result.recurring_templates

    def test_residual_excludes_recurring_events(self):
        """Recurring events are removed from the residual log."""
        base = datetime(2026, 3, 2, 8, 0)
        recurring_log = _repeated_weekly("q42", base, n_weeks=5)
        sporadic_time = datetime(2026, 3, 4, 14, 30)  # Wednesday
        sporadic_log = _make_log("q99", [sporadic_time])
        combined = _concat(recurring_log, sporadic_log)

        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        result = ext.fit(combined)

        # Residual should contain only the sporadic query.
        assert len(result.residual_log) == 1
        assert result.residual_log["template_id"].iloc[0] == "q99"

    def test_multiple_queries_per_slot(self):
        """Template that fires 3 times each Monday 08:00 slot."""
        base = datetime(2026, 3, 2, 8, 0)
        n_weeks = 4
        rows = []
        for w in range(n_weeks):
            t = base + timedelta(weeks=w)
            for i in range(3):
                rows.append(
                    {
                        "timestamp": t + timedelta(seconds=60 * i),
                        "template_id": "q7",
                    }
                )
        log = pd.DataFrame(rows)

        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        result = ext.fit(log)

        assert "q7" in result.recurring_templates
        rt = result.recurring_templates["q7"]
        slot = ext._weekly_slot(base)
        assert rt.expected_count_per_slot[slot] == pytest.approx(3.0)

    def test_reliability_matrix_shape(self):
        """Reliability matrix has shape (n_templates, n_weekly_slots)."""
        base = datetime(2026, 3, 2, 8, 0)
        log = _concat(
            _repeated_weekly("q1", base, n_weeks=3),
            _repeated_weekly("q2", base.replace(hour=10), n_weeks=3),
        )

        ext = RecurringPatternExtractor(
            slot_minutes=60, reliability_threshold=0.8
        )
        result = ext.fit(log)

        assert result.reliability_matrix.shape == (2, 168)
        assert len(result.template_ids) == 2

    def test_template_ids_sorted(self):
        """template_ids array is sorted."""
        base = datetime(2026, 3, 2, 8, 0)
        log = _concat(
            _repeated_weekly("qZ", base, n_weeks=3),
            _repeated_weekly("qA", base.replace(hour=10), n_weeks=3),
        )
        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        result = ext.fit(log)
        assert list(result.template_ids) == ["qA", "qZ"]

    def test_edge_reliability_equals_threshold(self):
        """Exactly at threshold → included."""
        base = datetime(2026, 3, 2, 8, 0)
        log = _repeated_weekly("q42", base, n_weeks=4)
        fifth = _make_log("q1", [base + timedelta(weeks=4)])
        combined = _concat(log, fifth)

        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        result = ext.fit(combined)

        assert "q42" in result.recurring_templates
        slot = ext._weekly_slot(base)
        assert result.recurring_templates["q42"].reliable_slots[
            slot
        ] == pytest.approx(0.8)

    def test_template_id_coerced_to_str(self):
        """Integer template_ids in the log are coerced to strings."""
        base = datetime(2026, 3, 2, 8, 0)
        log = pd.DataFrame(
            {
                "timestamp": [base + timedelta(weeks=w) for w in range(5)],
                "template_id": [42] * 5,
            }
        )

        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        result = ext.fit(log)

        assert "42" in result.recurring_templates


# ---------------------------------------------------------------------------
# Tests: generate_recurring()
# ---------------------------------------------------------------------------


class TestGenerateRecurring:

    def _fit_simple(self) -> RecurringPatternExtractor:
        """Fit with one template at Monday 08:00, count=2."""
        base = datetime(2026, 3, 2, 8, 0)
        n_weeks = 5
        rows = []
        for w in range(n_weeks):
            t = base + timedelta(weeks=w)
            for i in range(2):
                rows.append(
                    {
                        "timestamp": t + timedelta(seconds=30 * i),
                        "template_id": "q42",
                    }
                )
        log = pd.DataFrame(rows)
        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        ext.fit(log)
        return ext

    def test_generates_expected_count(self):
        """Generates the expected number of queries per slot."""
        ext = self._fit_simple()
        start = datetime(2026, 4, 6, 8, 0)  # Monday
        arrivals = ext.generate_recurring(start, n_slots=1, random_state=0)
        assert len(arrivals) == 2
        assert (arrivals["template_id"] == "q42").all()

    def test_no_arrivals_outside_slot(self):
        """Slots not matching a recurring template produce no arrivals."""
        ext = self._fit_simple()
        start = datetime(2026, 4, 7, 8, 0)  # Tuesday
        arrivals = ext.generate_recurring(start, n_slots=1, random_state=0)
        assert len(arrivals) == 0

    def test_timestamps_within_slot(self):
        """All timestamps are within [0, slot_seconds)."""
        ext = self._fit_simple()
        start = datetime(2026, 4, 6, 8, 0)
        arrivals = ext.generate_recurring(start, n_slots=1, random_state=0)
        slot_s = 15 * 60
        assert (arrivals["timestamp"] >= 0).all()
        assert (arrivals["timestamp"] < slot_s).all()

    def test_multi_slot_forecast(self):
        """Recurring pattern across multiple slots."""
        ext = self._fit_simple()
        # 4 slots starting at Monday 07:45 — only the 2nd slot
        # (08:00–08:15) matches the recurring pattern.
        start = datetime(2026, 4, 6, 7, 45)  # Monday 07:45
        arrivals = ext.generate_recurring(start, n_slots=4, random_state=0)
        assert len(arrivals) == 2
        slot_s = 15 * 60
        assert (arrivals["timestamp"] >= 1 * slot_s).all()
        assert (arrivals["timestamp"] < 2 * slot_s).all()

    def test_reproducible_with_seed(self):
        """Same seed → same output."""
        ext = self._fit_simple()
        start = datetime(2026, 4, 6, 8, 0)
        a1 = ext.generate_recurring(start, n_slots=1, random_state=42)
        a2 = ext.generate_recurring(start, n_slots=1, random_state=42)
        pd.testing.assert_frame_equal(a1, a2)

    def test_empty_recurring_generates_empty(self):
        """If no templates pass threshold, generate_recurring returns empty."""
        log = _make_log("q99", [datetime(2026, 3, 2, 8, 0)])
        ext = RecurringPatternExtractor(
            slot_minutes=15, reliability_threshold=0.8
        )
        ext.fit(log)
        arrivals = ext.generate_recurring(
            datetime(2026, 4, 6, 8, 0), n_slots=1, random_state=0
        )
        assert len(arrivals) == 0
