"""Tests for autoslo.utils.structured_log."""

from __future__ import annotations

import logging
import os

import pandas as pd

from autoslo.utils.logging import (
    LOGGER_NAME,
    REQUIRED_KEYS,
    REQUIRED_KEYS_LIST,
    StructuredLogHandler,
    emit_structured,
    setup_structured_logging,
)
from autoslo.utils.structured_events import (
    ArrivalEvent,
    BaseStructuredEvent,
    CompletionEvent,
    RoutingDecisionEvent,
    RunStartEvent,
    ScenarioResultEvent,
    wall_clock_utc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(**overrides) -> BaseStructuredEvent:
    """Build a minimal valid ArrivalEvent with optional overrides."""
    defaults = dict(
        rel_time_s=0.0,
        source="test_harness",
        query_id="q0",
        query_text_id=1,
    )
    defaults.update(overrides)
    return ArrivalEvent(**defaults)


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
        """Each event subclass should set event_type in __post_init__."""
        for cls in [
            ArrivalEvent,
            CompletionEvent,
            RoutingDecisionEvent,
            RunStartEvent,
            ScenarioResultEvent,
        ]:
            ev = cls(source="test")
            assert ev.event_type != ""
            assert "event_type" in ev.to_dict()


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
        out_path = handler.finalize()
        assert out_path is not None
        assert os.path.basename(out_path) == "structured_log.parquet"

        df = pd.read_parquet(out_path)
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
        out_path = handler.finalize()
        df = pd.read_parquet(out_path)
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
            CompletionEvent(
                rel_time_s=0.5,
                source="test_harness",
                query_id="q1",
                query_text_id=1,
                cluster_name="c0",
                latency_s=0.5,
            )
        )
        out = handler.finalize()
        df = pd.read_parquet(out)
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
        df = pd.read_parquet(out)
        assert len(df) == 4 * n_per_thread

    def teardown_method(self):
        logger = logging.getLogger(LOGGER_NAME)
        for h in list(logger.handlers):
            if isinstance(h, StructuredLogHandler):
                logger.removeHandler(h)
