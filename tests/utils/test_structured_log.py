"""Tests for autoslo.utils.structured_log."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from autoslo.filesystem.structured_events import (
    BaseStructuredEvent,
    EventType,
    QueryRelatedEvent,
    wall_clock_utc,
)
from autoslo.filesystem.structured_log import (
    LOGGER_NAME,
    REQUIRED_KEYS,
    REQUIRED_KEYS_LIST,
    StructuredLog,
    StructuredLogHandler,
    emit_structured,
    setup_structured_logging,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(**overrides) -> QueryRelatedEvent:
    """Build a minimal valid arrival QueryRelatedEvent with optional overrides."""
    defaults = dict(
        rel_time_s=0.0,
        event_type=EventType.ARRIVAL,
        source="test_harness",
        query_id="q0",
        query_text_id=1,
    )
    defaults.update(overrides)
    return QueryRelatedEvent(**defaults)


def _make_record(**overrides) -> dict:
    """Build a minimal valid record dict for handler-level tests."""
    base = {
        "wall_clock_s": 1.0,
        "rel_time_s": 0.0,
        "event_type": "test",
        "source": "test_harness",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# emit_structured validation
# ---------------------------------------------------------------------------


class TestEmitStructuredValidation:
    """Tests for emit_structured behavior."""

    def test_short_circuits_without_handler(self):
        """emit_structured should silently return when no handler is attached."""
        # Ensure no handler.
        logger = logging.getLogger(LOGGER_NAME)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        # Should not raise.
        emit_structured(_make_event())

    def test_wall_clock_utc_returns_float(self):
        ts = wall_clock_utc()
        assert isinstance(ts, float)
        assert ts > 0

    def test_wall_clock_s_auto_populated(self):
        """wall_clock_s should be auto-filled by wall_clock_utc()."""
        event = _make_event()
        assert isinstance(event.wall_clock_s, float)
        assert event.wall_clock_s > 0

    def test_event_to_dict_has_required_keys(self):
        event = _make_event()
        d = event.to_dict()
        assert REQUIRED_KEYS <= d.keys()

    def test_all_event_types_have_event_type(self):
        """Every EventType member should produce a valid event."""
        from autoslo.filesystem.structured_events import REQUIRED_DETAILS

        for et in EventType:
            details = {k: None for k in REQUIRED_DETAILS.get(et, [])}
            ev = BaseStructuredEvent(
                rel_time_s=0.0,
                event_type=et,
                source="test",
                details=details,
            )
            assert ev.event_type == et
            d = ev.to_dict()
            assert "event_type" in d
            assert d["event_type"] == et.value


# ---------------------------------------------------------------------------
# StructuredLogHandler unit tests
# ---------------------------------------------------------------------------


class TestStructuredLogHandler:
    """Unit tests for buffering, flushing, and finalization."""

    def test_emit_ignores_non_dict(self, tmp_path):
        handler = StructuredLogHandler(out_dir=str(tmp_path))
        lr = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="plain string",
            args=None,
            exc_info=None,
        )
        handler.emit(lr)
        assert handler.buffered_count == 0

    def test_emit_buffers_dict(self, tmp_path):
        handler = StructuredLogHandler(out_dir=str(tmp_path))
        lr = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=_make_record(),
            args=None,
            exc_info=None,
        )
        handler.emit(lr)
        assert handler.buffered_count == 1

    def test_auto_flush_at_threshold(self, tmp_path):
        handler = StructuredLogHandler(
            out_dir=str(tmp_path),
            flush_threshold=3,
        )
        for i in range(3):
            lr = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=_make_record(wall_clock_s=float(i)),
                args=None,
                exc_info=None,
            )
            handler.emit(lr)
        # Buffer should be drained after the 3rd emit.
        assert handler.buffered_count == 0
        shard = tmp_path / "_shard_0000.parquet"
        assert shard.exists()

    def test_finalize_consolidates_shards(self, tmp_path):
        handler = StructuredLogHandler(
            out_dir=str(tmp_path),
            flush_threshold=2,
        )
        for i in range(5):
            lr = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=_make_record(wall_clock_s=float(i), idx=i),
                args=None,
                exc_info=None,
            )
            handler.emit(lr)
        log = handler.finalize()
        assert log is not None
        assert log.path.name == "structured_log.parquet"

        df = log.df
        assert len(df) == 5
        # Required columns should appear first in defined order.
        assert list(df.columns[: len(REQUIRED_KEYS)]) == REQUIRED_KEYS_LIST
        # No leftover shard files.
        shards = list(tmp_path.glob("_shard_*.parquet"))
        assert len(shards) == 0

    def test_finalize_returns_none_when_empty(self, tmp_path):
        handler = StructuredLogHandler(out_dir=str(tmp_path))
        assert handler.finalize() is None

    def test_reset_clears_state(self, tmp_path):
        handler = StructuredLogHandler(
            out_dir=str(tmp_path),
            flush_threshold=100,
        )
        lr = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=_make_record(),
            args=None,
            exc_info=None,
        )
        handler.emit(lr)
        assert handler.buffered_count == 1
        handler.reset()
        assert handler.buffered_count == 0

    def test_reset_switches_out_dir(self, tmp_path):
        handler = StructuredLogHandler(out_dir=str(tmp_path / "a"))
        new_dir = str(tmp_path / "b")
        handler.reset(out_dir=new_dir)
        assert handler.out_dir == new_dir

    def test_dynamic_columns_discovery(self, tmp_path):
        handler = StructuredLogHandler(
            out_dir=str(tmp_path),
            flush_threshold=100,
        )
        # Record 1: only required keys
        lr1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=_make_record(),
            args=None,
            exc_info=None,
        )
        handler.emit(lr1)
        # Record 2: has extra 'cluster_name' and 'rpu' keys
        lr2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=_make_record(cluster_name="cluster-0", rpu=8),
            args=None,
            exc_info=None,
        )
        handler.emit(lr2)
        log = handler.finalize()
        df = log.df
        assert "cluster_name" in df.columns
        assert "rpu" in df.columns
        # First row should have NaN for the extra columns.
        assert pd.isna(df.loc[0, "cluster_name"])
        assert df.loc[1, "cluster_name"] == "cluster-0"


# ---------------------------------------------------------------------------
# setup_structured_logging integration
# ---------------------------------------------------------------------------


class TestSetupStructuredLogging:
    """Integration tests for the setup helper + emit_structured flow."""

    def test_roundtrip(self, tmp_path):
        handler = setup_structured_logging(out_dir=str(tmp_path))
        emit_structured(_make_event())
        emit_structured(
            QueryRelatedEvent(
                rel_time_s=0.5,
                event_type=EventType.COMPLETION,
                source="test_harness",
                cluster_name="c0",
                details={"success": True, "latency_s": 0.5},
                query_id="q1",
                query_text_id=1,
            )
        )
        out = handler.finalize()
        df = out.df
        assert len(df) == 2
        assert set(df["event_type"]) == {"arrival", "completion"}

    def test_replaces_previous_handler(self, tmp_path):
        h1 = setup_structured_logging(out_dir=str(tmp_path / "a"))
        h2 = setup_structured_logging(out_dir=str(tmp_path / "b"))
        logger = logging.getLogger(LOGGER_NAME)
        structured_handlers = [
            h for h in logger.handlers if isinstance(h, StructuredLogHandler)
        ]
        assert len(structured_handlers) == 1
        assert structured_handlers[0] is h2

    def test_no_propagation_to_root(self, tmp_path):
        setup_structured_logging(out_dir=str(tmp_path))
        logger = logging.getLogger(LOGGER_NAME)
        assert logger.propagate is False

    def teardown_method(self):
        """Clean up the canonical logger after each test."""
        logger = logging.getLogger(LOGGER_NAME)
        for h in list(logger.handlers):
            if isinstance(h, StructuredLogHandler):
                logger.removeHandler(h)


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_emits(self, tmp_path):
        import threading

        handler = setup_structured_logging(
            out_dir=str(tmp_path),
            flush_threshold=50,
        )
        errors = []
        n_per_thread = 200

        def _emit(thread_idx: int):
            try:
                for i in range(n_per_thread):
                    emit_structured(
                        _make_event(
                            source=f"thread_{thread_idx}",
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_emit, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        out = handler.finalize()
        df = out.df
        assert len(df) == 4 * n_per_thread

    def teardown_method(self):
        logger = logging.getLogger(LOGGER_NAME)
        for h in list(logger.handlers):
            if isinstance(h, StructuredLogHandler):
                logger.removeHandler(h)


# ---------------------------------------------------------------------------
# StructuredLog.query_latencies
# ---------------------------------------------------------------------------


def _make_log(*rows: dict) -> pd.DataFrame:
    """Build a minimal structured-log DataFrame from explicit row dicts."""
    defaults = {"wall_clock_s": 0.0, "source": "test", "query_text_id": "1"}
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


class TestStructuredLogQueryLatencies:

    def test_happy_path_dataframe(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 1.0},
            {"event_type": "completion", "query_id": "q0", "rel_time_s": 3.0},
        )
        result = StructuredLog.load(log).query_latencies()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["arrival_s"] == pytest.approx(1.0)
        assert row["completion_s"] == pytest.approx(3.0)
        assert row["latency_s"] == pytest.approx(2.0)
        assert set(result.columns) == {
            "query_id",
            "query_text_id",
            "arrival_s",
            "completion_s",
            "latency_s",
        }

    def test_happy_path_parquet(self, tmp_path):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 2.0},
            {"event_type": "completion", "query_id": "q0", "rel_time_s": 5.0},
        )
        path = tmp_path / "structured_log.parquet"
        log.to_parquet(path, index=False)
        result = StructuredLog.load(path).query_latencies()
        assert len(result) == 1
        assert result.iloc[0]["latency_s"] == pytest.approx(3.0)

    def test_multi_query(self):
        log = _make_log(
            {
                "event_type": "arrival",
                "query_id": "q0",
                "rel_time_s": 0.0,
                "query_text_id": "1",
            },
            {
                "event_type": "completion",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "query_text_id": "1",
            },
            {
                "event_type": "arrival",
                "query_id": "q1",
                "rel_time_s": 0.5,
                "query_text_id": "2",
            },
            {
                "event_type": "completion",
                "query_id": "q1",
                "rel_time_s": 2.5,
                "query_text_id": "2",
            },
            {
                "event_type": "arrival",
                "query_id": "q2",
                "rel_time_s": 1.0,
                "query_text_id": "3",
            },
            {
                "event_type": "completion",
                "query_id": "q2",
                "rel_time_s": 4.0,
                "query_text_id": "3",
            },
        )
        result = StructuredLog.load(log).query_latencies().set_index("query_id")
        assert result.loc["q0", "latency_s"] == pytest.approx(1.0)
        assert result.loc["q1", "latency_s"] == pytest.approx(2.0)
        assert result.loc["q2", "latency_s"] == pytest.approx(3.0)

    def test_drop_incomplete_true_default(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0},
        )
        result = StructuredLog.load(log).query_latencies()
        assert len(result) == 0

    def test_drop_incomplete_partial(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0},
            {"event_type": "completion", "query_id": "q0", "rel_time_s": 1.0},
            {"event_type": "arrival", "query_id": "q1", "rel_time_s": 0.0},
        )
        result = StructuredLog.load(log).query_latencies()
        assert len(result) == 1
        assert result.iloc[0]["query_id"] == "q0"

    def test_drop_incomplete_false_keeps_nan(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0},
        )
        result = StructuredLog.load(log).query_latencies(drop_incomplete=False)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["completion_s"])
        assert pd.isna(result.iloc[0]["latency_s"])

    def test_empty_log_returns_empty_dataframe(self):
        log = _make_log()
        result = StructuredLog.load(log).query_latencies()
        assert len(result) == 0
        assert set(result.columns) == {
            "query_id",
            "query_text_id",
            "arrival_s",
            "completion_s",
            "latency_s",
        }

    def test_no_completion_events(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0},
            {"event_type": "arrival", "query_id": "q1", "rel_time_s": 1.0},
        )
        assert len(StructuredLog.load(log).query_latencies()) == 0
        result_full = StructuredLog.load(log).query_latencies(
            drop_incomplete=False
        )
        assert len(result_full) == 2
        assert result_full["latency_s"].isna().all()

    def test_duplicate_event_raises(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0},
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.1},
            {"event_type": "completion", "query_id": "q0", "rel_time_s": 1.0},
        )
        with pytest.raises(ValueError, match="q0"):
            StructuredLog.load(log).query_latencies()

    def test_extra_columns_ignored(self):
        log = _make_log(
            {
                "event_type": "arrival",
                "query_id": "q0",
                "rel_time_s": 0.0,
                "cluster_name": "c1",
                "details": "{}",
            },
            {
                "event_type": "completion",
                "query_id": "q0",
                "rel_time_s": 2.0,
                "cluster_name": "c1",
                "details": "{}",
            },
        )
        result = StructuredLog.load(log).query_latencies()
        assert "cluster_name" not in result.columns
        assert result.iloc[0]["latency_s"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# StructuredLog.flat_df
# ---------------------------------------------------------------------------


def _make_log_with_details(*rows: dict) -> pd.DataFrame:
    """Build a log DataFrame with a JSON-string ``details`` column."""
    import json as _json

    defaults = {"wall_clock_s": 0.0, "source": "test", "query_text_id": "1"}
    records = []
    for r in rows:
        rec = {**defaults, **r}
        if "details" in rec and isinstance(rec["details"], dict):
            rec["details"] = _json.dumps(rec["details"])
        records.append(rec)
    return pd.DataFrame(records)


class TestFlatDf:
    def test_expands_details_fields(self):
        log = _make_log_with_details(
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "details": {"slo_violation": 0.25, "cost": 0.4},
            },
        )
        result = StructuredLog.load(log).flat_df()
        assert "details" not in result.columns
        assert "slo_violation" in result.columns
        assert result.iloc[0]["slo_violation"] == pytest.approx(0.25)

    def test_coerces_numeric_details_fields(self):
        log = _make_log_with_details(
            {
                "event_type": "query_execution_finish",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "details": {"latency_s": "12.5"},
            },
        )
        result = StructuredLog.load(log).flat_df(coerce_numerics=True)
        assert pd.api.types.is_numeric_dtype(result["latency_s"])
        assert result.iloc[0]["latency_s"] == pytest.approx(12.5)

    def test_does_not_coerce_mixed_string_field(self):
        log = _make_log_with_details(
            {
                "event_type": "spin_up_requested",
                "query_id": None,
                "rel_time_s": 0.0,
                "details": {"reason": "initial"},
            },
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "details": {"reason": "overload"},
            },
        )
        result = StructuredLog.load(log).flat_df(coerce_numerics=True)
        # "reason" has string values; should stay as object dtype
        assert result["reason"].dtype == object

    def test_drops_fwd_queries_by_default(self):
        log = _make_log_with_details(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 1.0, "details": {}},
            {"event_type": "arrival", "query_id": "fwd_q1", "rel_time_s": 2.0, "details": {}},
            {"event_type": "arrival", "query_id": "fwd_q2", "rel_time_s": 3.0, "details": {}},
        )
        result = StructuredLog.load(log).flat_df()
        assert len(result) == 1
        assert result.iloc[0]["query_id"] == "q0"

    def test_keeps_fwd_queries_when_disabled(self):
        log = _make_log_with_details(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 1.0, "details": {}},
            {"event_type": "arrival", "query_id": "fwd_q1", "rel_time_s": 2.0, "details": {}},
        )
        result = StructuredLog.load(log).flat_df(drop_fwd_queries=False)
        assert len(result) == 2

    def test_no_details_column_returns_base(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 1.0},
        )
        result = StructuredLog.load(log).flat_df()
        assert "details" not in result.columns
        assert "event_type" in result.columns

    def test_resets_index(self):
        log = _make_log_with_details(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0, "details": {}},
            {"event_type": "arrival", "query_id": "fwd_q1", "rel_time_s": 1.0, "details": {}},
            {"event_type": "arrival", "query_id": "q2", "rel_time_s": 2.0, "details": {}},
        )
        result = StructuredLog.load(log).flat_df()
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# StructuredLog.query_slo_outcomes
# ---------------------------------------------------------------------------


class _FakeSloResolver:
    """Minimal stand-in for SloResolver used in tests."""

    def __init__(self, mapping: dict[str, float], default: float = 10.0):
        self._mapping = mapping
        self._default = default

    def resolve(self, qid) -> float:
        if qid is None:
            return self._default
        return self._mapping.get(str(qid), self._default)


class TestQuerySloOutcomes:
    def _make_slog(self, arrival: float, completion: float, qid: str = "q0", qtid: str = "t1"):
        log = _make_log(
            {"event_type": "arrival", "query_id": qid, "rel_time_s": arrival, "query_text_id": qtid},
            {"event_type": "completion", "query_id": qid, "rel_time_s": completion, "query_text_id": qtid},
        )
        return StructuredLog.load(log)

    def test_columns_present(self):
        slog = self._make_slog(0.0, 5.0)
        resolver = _FakeSloResolver({"t1": 10.0})
        result = slog.query_slo_outcomes(resolver)
        assert set(result.columns) >= {
            "query_id", "latency_s", "slo_s", "slo_violated",
            "slo_overshoot_s", "relative_violation",
        }

    def test_no_violation(self):
        slog = self._make_slog(0.0, 5.0)  # latency = 5s
        resolver = _FakeSloResolver({"t1": 10.0})  # SLO = 10s
        result = slog.query_slo_outcomes(resolver)
        row = result.iloc[0]
        assert row["slo_s"] == pytest.approx(10.0)
        assert row["slo_violated"] == 0
        assert row["slo_overshoot_s"] == pytest.approx(0.0)
        assert row["relative_violation"] == pytest.approx(0.0)

    def test_violation(self):
        slog = self._make_slog(0.0, 15.0)  # latency = 15s
        resolver = _FakeSloResolver({"t1": 10.0})  # SLO = 10s
        result = slog.query_slo_outcomes(resolver)
        row = result.iloc[0]
        assert row["slo_violated"] == 1
        assert row["slo_overshoot_s"] == pytest.approx(5.0)
        assert row["relative_violation"] == pytest.approx(0.5)

    def test_exact_slo_boundary(self):
        slog = self._make_slog(0.0, 10.0)  # latency == SLO exactly
        resolver = _FakeSloResolver({"t1": 10.0})
        result = slog.query_slo_outcomes(resolver)
        assert result.iloc[0]["slo_violated"] == 0
        assert result.iloc[0]["slo_overshoot_s"] == pytest.approx(0.0)

    def test_per_template_slo(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0, "query_text_id": "fast"},
            {"event_type": "completion", "query_id": "q0", "rel_time_s": 3.0, "query_text_id": "fast"},
            {"event_type": "arrival", "query_id": "q1", "rel_time_s": 0.0, "query_text_id": "slow"},
            {"event_type": "completion", "query_id": "q1", "rel_time_s": 3.0, "query_text_id": "slow"},
        )
        resolver = _FakeSloResolver({"fast": 2.0, "slow": 5.0})
        result = StructuredLog.load(log).query_slo_outcomes(resolver).set_index("query_id")
        assert result.loc["q0", "slo_violated"] == 1  # 3s > 2s
        assert result.loc["q1", "slo_violated"] == 0  # 3s < 5s

    def test_empty_log_returns_correct_columns(self):
        log = _make_log()
        resolver = _FakeSloResolver({})
        result = StructuredLog.load(log).query_slo_outcomes(resolver)
        assert len(result) == 0
        assert set(result.columns) >= {
            "slo_s", "slo_violated", "slo_overshoot_s", "relative_violation",
        }

    def test_zero_slo_does_not_raise(self):
        slog = self._make_slog(0.0, 5.0)
        # slo_s = 0 would cause divide-by-zero; should produce NaN, not raise.
        resolver = _FakeSloResolver({"t1": 0.0})
        result = slog.query_slo_outcomes(resolver)
        assert pd.isna(result.iloc[0]["relative_violation"])


# ---------------------------------------------------------------------------
# StructuredLog.logos_df
# ---------------------------------------------------------------------------


class TestLogosDf:
    def _make_full_slog(self):
        """Log with arrival, query_execution_finish, query_routed, latency_update, completion."""
        log = _make_log_with_details(
            {
                "event_type": "arrival",
                "query_id": "q0",
                "rel_time_s": 0.0,
                "query_text_id": "t1",
                "details": {},
            },
            {
                "event_type": "query_routed",
                "query_id": "q0",
                "rel_time_s": 0.1,
                "query_text_id": "t1",
                "details": {"latency_s": "8.0"},
            },
            {
                "event_type": "latency_update",
                "query_id": "q0",
                "rel_time_s": 0.5,
                "query_text_id": "t1",
                "details": {"latency_s": "9.0", "old_latency_s": "8.0"},
            },
            {
                "event_type": "query_execution_finish",
                "query_id": "q0",
                "rel_time_s": 10.0,
                "query_text_id": "t1",
                "details": {"latency_s": "10.0"},
            },
            {
                "event_type": "completion",
                "query_id": "q0",
                "rel_time_s": 10.0,
                "query_text_id": "t1",
                "details": {"success": True},
            },
            # Non-query row (no query_id)
            {
                "event_type": "cluster_ready",
                "query_id": None,
                "rel_time_s": 0.05,
                "query_text_id": None,
                "details": {},
            },
        )
        return StructuredLog.load(log)

    def test_details_expanded(self):
        slog = self._make_full_slog()
        result = slog.logos_df()
        assert "details" not in result.columns
        # latency_s is split into event-type-specific columns by logos_df
        assert "actual_execution_latency_s" in result.columns
        assert "predicted_latency_s" in result.columns

    def test_without_resolver_no_slo_columns(self):
        slog = self._make_full_slog()
        result = slog.logos_df(slo_resolver=None)
        assert "slo_violated" not in result.columns
        assert "slo_s" not in result.columns

    def test_slo_columns_broadcast_to_all_query_rows(self):
        slog = self._make_full_slog()
        resolver = _FakeSloResolver({"t1": 15.0})  # latency=10s < 15s → no violation
        result = slog.logos_df(slo_resolver=resolver)
        query_rows = result[result["query_id"] == "q0"]
        assert query_rows["slo_violated"].dropna().unique().tolist() == [0]
        assert (query_rows["slo_s"].dropna() == 15.0).all()

    def test_non_query_rows_have_nan_slo(self):
        slog = self._make_full_slog()
        resolver = _FakeSloResolver({"t1": 15.0})
        result = slog.logos_df(slo_resolver=resolver)
        non_query = result[result["query_id"].isna()]
        assert non_query["slo_violated"].isna().all()

    def test_actual_execution_latency_only_on_execution_rows(self):
        slog = self._make_full_slog()
        result = slog.logos_df()
        exe_rows = result[result["event_type"] == "query_execution_finish"]
        non_exe_rows = result[result["event_type"] != "query_execution_finish"]
        assert exe_rows["actual_execution_latency_s"].notna().all()
        assert non_exe_rows["actual_execution_latency_s"].isna().all()

    def test_predicted_latency_only_on_routing_rows(self):
        slog = self._make_full_slog()
        result = slog.logos_df()
        routing_types = {"query_routed", "latency_update"}
        routing_rows = result[result["event_type"].isin(routing_types)]
        other_rows = result[~result["event_type"].isin(routing_types)]
        assert routing_rows["predicted_latency_s"].notna().all()
        assert other_rows["predicted_latency_s"].isna().all()

    def test_fwd_queries_excluded_by_default(self):
        log = _make_log_with_details(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0, "details": {}},
            {"event_type": "arrival", "query_id": "fwd_sim_q1", "rel_time_s": 1.0, "details": {}},
        )
        result = StructuredLog.load(log).logos_df()
        assert "fwd_sim_q1" not in result["query_id"].astype(str).values

    def test_violation_broadcast_correctly(self):
        slog = self._make_full_slog()
        resolver = _FakeSloResolver({"t1": 5.0})  # latency=10s > 5s → violated
        result = slog.logos_df(slo_resolver=resolver)
        query_rows = result[result["query_id"] == "q0"]
        assert (query_rows["slo_violated"].dropna() == 1).all()
        assert (query_rows["slo_overshoot_s"].dropna() == 5.0).all()


# ---------------------------------------------------------------------------
# StructuredLog.query_cluster_assignments
# ---------------------------------------------------------------------------


def _make_routing_log(*rows: dict) -> pd.DataFrame:
    """Build a log with routing events that have details as JSON strings."""
    import json as _json

    defaults = {"wall_clock_s": 0.0, "source": "QueryRouter", "query_text_id": "t1"}
    records = []
    for r in rows:
        rec = {**defaults, **r}
        if "details" in rec and isinstance(rec["details"], dict):
            rec["details"] = _json.dumps(rec["details"])
        records.append(rec)
    return pd.DataFrame(records)


class TestQueryClusterAssignments:
    def test_basic(self):
        log = _make_routing_log(
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "cluster_name": "autoslo-32-run1-0",
                "details": {"latency_s_for_routing": 10.5, "slo_violation": 0.0, "cost": 0.0, "cache_risk": 0.0},
            },
        )
        result = StructuredLog.load(log).query_cluster_assignments()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["query_id"] == "q0"
        assert row["cluster_name"] == "autoslo-32-run1-0"
        assert row["rpu"] == 32
        assert row["latency_s_for_routing"] == pytest.approx(10.5)
        assert row["slo_violation_at_routing"] == pytest.approx(0.0)

    def test_columns_present(self):
        log = _make_routing_log(
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "cluster_name": "autoslo-16-run1-1",
                "details": {"latency_s_for_routing": 5.0, "slo_violation": 0.25, "cost": 0.2, "cache_risk": 0.0},
            },
        )
        result = StructuredLog.load(log).query_cluster_assignments()
        assert set(result.columns) == {
            "query_id", "cluster_name", "rpu",
            "latency_s_for_routing", "slo_violation_at_routing",
        }

    def test_multiple_queries(self):
        log = _make_routing_log(
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "cluster_name": "autoslo-32-run1-0",
                "details": {"latency_s_for_routing": 10.0, "slo_violation": 0.0, "cost": 0.0, "cache_risk": 0.0},
            },
            {
                "event_type": "routing",
                "query_id": "q1",
                "rel_time_s": 2.0,
                "cluster_name": "autoslo-16-run1-1",
                "details": {"latency_s_for_routing": 5.0, "slo_violation": 0.5, "cost": 0.2, "cache_risk": 0.0},
            },
        )
        result = StructuredLog.load(log).query_cluster_assignments().set_index("query_id")
        assert result.loc["q0", "rpu"] == 32
        assert result.loc["q1", "rpu"] == 16
        assert result.loc["q1", "slo_violation_at_routing"] == pytest.approx(0.5)

    def test_fwd_queries_excluded(self):
        log = _make_routing_log(
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "cluster_name": "autoslo-32-run1-0",
                "details": {"latency_s_for_routing": 10.0, "slo_violation": 0.0, "cost": 0.0, "cache_risk": 0.0},
            },
            {
                "event_type": "routing",
                "query_id": "fwd_sim_q1",
                "rel_time_s": 2.0,
                "cluster_name": "autoslo-32-run1-0",
                "details": {"latency_s_for_routing": 8.0, "slo_violation": 0.0, "cost": 0.0, "cache_risk": 0.0},
            },
        )
        result = StructuredLog.load(log).query_cluster_assignments()
        assert len(result) == 1
        assert result.iloc[0]["query_id"] == "q0"

    def test_no_routing_events_returns_empty(self):
        log = _make_log(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0},
        )
        result = StructuredLog.load(log).query_cluster_assignments()
        assert len(result) == 0
        assert set(result.columns) == {
            "query_id", "cluster_name", "rpu",
            "latency_s_for_routing", "slo_violation_at_routing",
        }

    def test_unparseable_cluster_name_gives_none_rpu(self):
        log = _make_routing_log(
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 1.0,
                "cluster_name": "no-spinup-baseline",
                "details": {"latency_s_for_routing": 10.0, "slo_violation": 0.0, "cost": 0.0, "cache_risk": 0.0},
            },
        )
        result = StructuredLog.load(log).query_cluster_assignments()
        assert result.iloc[0]["rpu"] is None

    def test_logos_df_adds_cluster_columns(self):
        log = _make_log_with_details(
            {"event_type": "arrival", "query_id": "q0", "rel_time_s": 0.0, "query_text_id": "t1", "details": {}},
            {
                "event_type": "routing",
                "query_id": "q0",
                "rel_time_s": 0.1,
                "cluster_name": "autoslo-32-run1-0",
                "query_text_id": "t1",
                "details": {"latency_s_for_routing": "10.0", "slo_violation": "0.0", "cost": "0.0", "cache_risk": "0.0"},
            },
            {"event_type": "completion", "query_id": "q0", "rel_time_s": 12.0, "query_text_id": "t1", "details": {"success": True}},
        )
        resolver = _FakeSloResolver({"t1": 15.0})
        result = StructuredLog.load(log).logos_df(slo_resolver=resolver)
        query_rows = result[result["query_id"] == "q0"]
        assert (query_rows["selected_cluster_name"].dropna() == "autoslo-32-run1-0").all()
        assert (query_rows["selected_rpu"].dropna() == 32).all()
        # prediction_error = final_latency_s - latency_s_for_routing = 12 - 10 = 2
        assert query_rows["prediction_error"].dropna().iloc[0] == pytest.approx(2.0)
