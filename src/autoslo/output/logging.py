"""
structured_log.py
-----------------
Centralized structured-logging facility for autoslo.

Components (``WorkloadSimulator``, ``QueryRouter``, ``Autoscaler``,
``HeadroomPolicy``, ``WorkloadRunner``, …) emit structured *dict* records
through Python's standard :mod:`logging` framework.  A custom handler
(``StructuredLogHandler``) buffers those records and periodically flushes
them to Parquet shard files.  At the end of a run the caller invokes
``finalize()`` to consolidate all shards into one file.

Design
------
* **Required columns** — every record must include ``timestamp``,
  ``event_type``, and ``source``.  These are validated on ``emit()``.
* **Dynamic columns** — additional keys in the dict are discovered
  automatically.  The final Parquet file has the union of all columns
  seen across all shards (missing values become ``None``).
* **Thread safety** — the buffer is guarded by a :class:`threading.Lock`
  so that async runners can safely emit from multiple threads.
* **Python logging integration** — ``StructuredLogHandler`` is a
  :class:`logging.Handler` subclass.  Components call
  ``logging.getLogger("autoslo.structured").info(dict_record)`` **or**
  use the thin helper ``emit_structured(...)`` which does the same but
  with a friendlier API.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import pandas as pd

from autoslo.output.structured_events import BaseStructuredEvent

# ---------------------------------------------------------------------------
# Required keys every record must carry
# ---------------------------------------------------------------------------

REQUIRED_KEYS_LIST = ["wall_clock_s", "rel_time_s", "source", "event_type"]
REQUIRED_KEYS = frozenset(REQUIRED_KEYS_LIST)

# ---------------------------------------------------------------------------
# Canonical logger name used by all components
# ---------------------------------------------------------------------------

LOGGER_NAME = "autoslo.structured"

# ---------------------------------------------------------------------------
# Public helper: emit a structured record
# ---------------------------------------------------------------------------


def emit_structured(event: BaseStructuredEvent) -> None:
    """Convenience: emit a structured event through the canonical logger.

    Parameters
    ----------
    event :
        A :class:`BaseStructuredEvent` subclass instance.

    Short-circuits silently when no handler is attached to the
    structured logger, eliminating the need for per-call-site guards.
    """
    _logger = logging.getLogger(LOGGER_NAME)
    if not _logger.handlers:
        return
    record = event.to_dict()
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(
            f"Structured log record missing required key(s): {sorted(missing)}"
        )
    _logger.info(record)


# ---------------------------------------------------------------------------
# Structured-log handler
# ---------------------------------------------------------------------------


class StructuredLogHandler(logging.Handler):
    """Buffering :class:`logging.Handler` that collects structured dicts
    and flushes them to Parquet shard files.

    Parameters
    ----------
    out_dir :
        Directory where shard and consolidated files are written.
    flush_threshold :
        Number of buffered records that triggers an automatic flush.
    filename :
        Base name for the consolidated output file.
    """

    def __init__(
        self,
        out_dir: str,
        flush_threshold: int = 10_000,
        filename: str = "structured_log.parquet",
    ) -> None:
        super().__init__(level=logging.DEBUG)
        self._out_dir = out_dir
        self._flush_threshold = flush_threshold
        self._filename = filename

        self._buffer: list[dict[str, Any]] = []
        self._shard_idx: int = 0
        self._all_columns: set[str] = set(REQUIRED_KEYS)
        self._lock = threading.Lock()

    # -- properties --------------------------------------------------------

    @property
    def out_dir(self) -> str:
        return self._out_dir

    @out_dir.setter
    def out_dir(self, value: str) -> None:
        self._out_dir = value

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    # -- logging.Handler interface -----------------------------------------

    def emit(self, log_record: logging.LogRecord) -> None:
        """Accept a :class:`logging.LogRecord` whose ``msg`` is a dict.

        Non-dict messages are silently ignored so that the handler can
        coexist with regular text loggers on the same logger hierarchy.
        """
        msg = log_record.msg
        if not isinstance(msg, dict):
            return

        with self._lock:
            self._all_columns.update(msg.keys())
            self._buffer.append(msg)
            if len(self._buffer) >= self._flush_threshold:
                self._flush_locked()

    # -- flush / finalize --------------------------------------------------

    def flush(self) -> None:  # noqa: D102  (overrides Handler.flush)
        with self._lock:
            if self._buffer:
                self._flush_locked()

    def _flush_locked(self) -> None:
        """Write the current buffer to a numbered shard file.

        Caller must hold ``self._lock``.
        """
        os.makedirs(self._out_dir, exist_ok=True)
        cols = sorted(self._all_columns)
        df = pd.DataFrame(self._buffer, columns=cols)
        shard_path = os.path.join(
            self._out_dir,
            f"_shard_{self._shard_idx:04d}.parquet",
        )
        df.to_parquet(shard_path, index=False)
        self._shard_idx += 1
        self._buffer.clear()

    def finalize(self) -> str | None:
        """Flush remaining records, consolidate shards, and return the
        path to the consolidated file (or ``None`` if nothing was logged).

        Shard files are deleted after consolidation.
        """
        with self._lock:
            if self._buffer:
                self._flush_locked()

            if self._shard_idx == 0:
                return None

            dfs: list[pd.DataFrame] = []
            for idx in range(self._shard_idx):
                shard_path = os.path.join(
                    self._out_dir,
                    f"_shard_{idx:04d}.parquet",
                )
                dfs.append(pd.read_parquet(shard_path))
                os.remove(shard_path)

            consolidated = pd.concat(dfs, ignore_index=True)

            # Ensure required columns appear first, then alphabetical.
            additional_important_cols = [
                "query_id",
                "query_text_id",
                "cluster_name",
            ]
            ordered_cols = (
                REQUIRED_KEYS_LIST
                + additional_important_cols
                + sorted(
                    c
                    for c in self._all_columns
                    if c not in REQUIRED_KEYS
                    and c not in additional_important_cols
                )
            )
            consolidated = consolidated.reindex(columns=ordered_cols)

            out_path = os.path.join(self._out_dir, self._filename)
            consolidated.to_parquet(out_path, index=False)
            self._shard_idx = 0
            return out_path

    def reset(self, out_dir: str | None = None) -> None:
        """Drop all buffered records and shard state for a new run.

        If *out_dir* is provided the handler switches to that directory.
        """
        with self._lock:
            self._buffer.clear()
            self._shard_idx = 0
            self._all_columns = set(REQUIRED_KEYS)
            if out_dir is not None:
                self._out_dir = out_dir


# ---------------------------------------------------------------------------
# Setup helper: attach a StructuredLogHandler to the canonical logger
# ---------------------------------------------------------------------------


def setup_structured_logging(
    out_dir: str,
    flush_threshold: int = 10_000,
    filename: str = "structured_log.parquet",
) -> StructuredLogHandler:
    """Create and attach a :class:`StructuredLogHandler` to the canonical
    ``autoslo.structured`` logger.

    If a ``StructuredLogHandler`` is already attached, it is replaced so
    that callers can safely call this multiple times (e.g. on
    ``WorkloadSimulator.reset()``).

    Returns the handler instance so the caller can later call
    ``handler.finalize()`` or ``handler.reset()``.
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Remove any pre-existing StructuredLogHandler.
    for h in list(logger.handlers):
        if isinstance(h, StructuredLogHandler):
            logger.removeHandler(h)

    handler = StructuredLogHandler(
        out_dir=out_dir,
        flush_threshold=flush_threshold,
        filename=filename,
    )
    logger.addHandler(handler)
    # Don't propagate structured dicts to the root logger
    # (they aren't human-readable).
    logger.propagate = False
    # Ensure the logger level is permissive enough.
    if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)

    return handler

def setup_run_logging(
    out_dir: str,
    write_text_log: bool,
) -> StructuredLogHandler:
    """Configure file-based text logging and structured logging for a run.

    Returns the StructuredLogHandler so the caller can call
    .finalize() at the end of the run.
    """
    if write_text_log:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        fh = logging.FileHandler(os.path.join(out_dir, "run.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(fh)
        logger.propagate = False
        logging.info(f"Run directory created at {out_dir}")
    return setup_structured_logging(out_dir=out_dir)