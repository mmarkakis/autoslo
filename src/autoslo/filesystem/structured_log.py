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

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import Cluster
from autoslo.filesystem.structured_events import BaseStructuredEvent, EventType

# ---------------------------------------------------------------------------
# Public helper: emit a structured record
# ---------------------------------------------------------------------------

LOGGER_NAME = "autoslo.structured"

# Module-level reference to the active handler.  Set by
# setup_structured_logging so that emit_structured can call the handler
# directly without going through Python's logging machinery.
_active_handler: "StructuredLogHandler | None" = None


def emit_structured(event: BaseStructuredEvent) -> None:
    """Convenience: emit a structured event through the canonical logger.

    Parameters
    ----------
    event :
        A :class:`BaseStructuredEvent` subclass instance.

    Short-circuits silently when no handler is active, eliminating the
    need for per-call-site guards.
    """
    if _active_handler is None:
        return
    record = event.to_dict()
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(
            f"Structured log record missing required key(s): {sorted(missing)}"
        )
    _active_handler._emit_direct(record)


# ---------------------------------------------------------------------------
# Structured-log handler
# ---------------------------------------------------------------------------

REQUIRED_KEYS_LIST = ["wall_clock_s", "rel_time_s", "source", "event_type"]
REQUIRED_KEYS = frozenset(REQUIRED_KEYS_LIST)


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
        out_dir: str | Path,
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
    def out_dir(self) -> str | Path:
        return self._out_dir

    @out_dir.setter
    def out_dir(self, value: str | Path) -> None:
        self._out_dir = value

    @property
    def buffered_count(self) -> int:
        with self._lock:
            return len(self._buffer)

    # -- fast direct path (used by emit_structured) -----------------------

    def _emit_direct(self, record: dict[str, Any]) -> None:
        """Append *record* to the buffer without Python logging overhead.

        Called by :func:`emit_structured`.  The ``details`` value must be
        a plain ``dict`` (serialisation to JSON happens in
        :meth:`_flush_locked`).
        """
        with self._lock:
            self._all_columns.update(record.keys())
            self._buffer.append(record)
            if len(self._buffer) >= self._flush_threshold:
                self._flush_locked()

    # -- logging.Handler interface (fallback for direct logger.info usage) --

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
        # Serialise details dicts to JSON strings for Parquet storage.
        # Done here in bulk rather than per-event in the hot path.
        if "details" in df.columns:
            df["details"] = [
                json.dumps(d, default=str) if isinstance(d, dict) and d else (d or "")
                for d in df["details"]
            ]
        shard_path = os.path.join(
            self._out_dir,
            f"_shard_{self._shard_idx:04d}.parquet",
        )
        df.to_parquet(shard_path, index=False)
        self._shard_idx += 1
        self._buffer.clear()

    def finalize(self) -> StructuredLog | None:
        """Flush remaining records, consolidate shards, and return a
        :class:`StructuredLog` for the consolidated file (or ``None`` if
        nothing was logged).

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
            return StructuredLog.load(Path(out_path))

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
    out_dir: str | Path,
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

    global _active_handler
    _active_handler = handler
    return handler


def setup_run_logging(
    out_dir: str | Path,
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


# ---------------------------------------------------------------------------
# StructuredLog — read-only analytical view of a consolidated log
# ---------------------------------------------------------------------------

_LATENCY_COLS = ["rel_time_s", "event_type", "query_id", "query_text_id"]


class StructuredLog:
    """Read-only view of a consolidated structured log Parquet file.

    Analogous to :class:`~autoslo.workload_execution.trace.Trace` for
    Redshift query history, but for the autoslo structured event log.

    Obtain an instance via :meth:`load` (from a run ID or explicit path)
    or directly from :meth:`StructuredLogHandler.finalize`.
    """

    def __init__(self, parquet_path: Path) -> None:
        self._parquet_path = parquet_path
        self._df: pd.DataFrame | None = None

    @classmethod
    def load(cls, source: str | Path | pd.DataFrame) -> StructuredLog:
        """Load a consolidated structured log.

        Parameters
        ----------
        source :
            One of:

            * A ``pd.DataFrame`` — used directly as the in-memory log.
            * A run ID string (resolved to
              ``data/runs/<run_id>/structured_log.parquet``).
            * A :class:`~pathlib.Path` / path-like string pointing to the
              Parquet file or the directory that contains it.
        """
        if isinstance(source, pd.DataFrame):
            instance = cls.__new__(cls)
            instance._parquet_path = None  # type: ignore[assignment]
            instance._df = source
            return instance

        path = Path(source)
        if isinstance(source, str) and not path.exists():
            # Treat as a run ID.
            path = Path(pu.get_runs_path()) / source / "structured_log.parquet"
        elif path.is_dir():
            path = path / "structured_log.parquet"

        if not path.exists():
            raise FileNotFoundError(f"No structured log found at {path}")

        return cls(path)

    @property
    def path(self) -> Path | None:
        """
        Path to the consolidated Parquet file, or ``None`` for in-memory
        instances.
        """
        return self._parquet_path

    @property
    def df(self) -> pd.DataFrame:
        """Full consolidated log, lazy-loaded and cached."""
        if self._df is None:
            self._df = pd.read_parquet(self._parquet_path)
            if any(col not in self._df.columns for col in REQUIRED_KEYS):
                missing = [
                    col for col in REQUIRED_KEYS if col not in self._df.columns
                ]
                raise ValueError(
                    f"Structured log at {self._parquet_path} is missing "
                    f"required column(s): {missing}"
                )
            if "details" in self._df.columns:
                # Parse string into dict for easier downstream access.
                def _parse_details(raw: Any) -> dict:
                    if isinstance(raw, dict):
                        return raw
                    if isinstance(raw, str) and raw:
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                return parsed
                        except (TypeError, json.JSONDecodeError):
                            return {}
                    return {}

                self._df["details"] = self._df["details"].apply(_parse_details)
        return self._df

    def query_latencies(self, *, drop_incomplete: bool = True) -> pd.DataFrame:
        """Return per-query end-to-end latency (COMPLETION − ARRIVAL rel_time_s).

        Columns: ``query_id``, ``query_text_id``, ``arrival_s``,
        ``completion_s``, ``latency_s``.
        Rows where either event is missing are dropped when
        *drop_incomplete* is ``True`` (default) or kept as ``NaN`` when
        ``False``.
        Raises ``ValueError`` on duplicate (query_id, event_type) pairs.
        """
        df = (
            self.df[_LATENCY_COLS]
            if set(_LATENCY_COLS).issubset(self.df.columns)
            else self.df
        )

        if df.empty or "event_type" not in df.columns:
            return pd.DataFrame(
                columns=[
                    "query_id",
                    "query_text_id",
                    "arrival_s",
                    "completion_s",
                    "latency_s",
                ]
            )

        filtered = df[
            df["event_type"].isin(
                {EventType.ARRIVAL.value, EventType.COMPLETION.value}
            )
        ][_LATENCY_COLS].copy()

        if filtered.empty:
            return pd.DataFrame(
                columns=[
                    "query_id",
                    "query_text_id",
                    "arrival_s",
                    "completion_s",
                    "latency_s",
                ]
            )

        dupes = filtered.groupby(["query_id", "event_type"]).size()
        dupes = dupes[dupes > 1]
        if not dupes.empty:
            bad_ids = dupes.index.get_level_values("query_id").unique().tolist()
            raise ValueError(
                f"Duplicate (query_id, event_type) pairs in structured log for "
                f"query_id(s): {bad_ids!r}"
            )

        pivoted = (
            filtered.pivot(
                index=["query_id", "query_text_id"],
                columns="event_type",
                values="rel_time_s",
            )
            .rename_axis(columns=None)
            .reset_index()
        )

        for col in (EventType.ARRIVAL.value, EventType.COMPLETION.value):
            if col not in pivoted.columns:
                pivoted[col] = float("nan")

        result = pd.DataFrame(
            {
                "query_id": pivoted["query_id"],
                "query_text_id": pivoted["query_text_id"].astype(str),
                "arrival_s": pivoted[EventType.ARRIVAL.value],
                "completion_s": pivoted[EventType.COMPLETION.value],
                "latency_s": (
                    pivoted[EventType.COMPLETION.value]
                    - pivoted[EventType.ARRIVAL.value]
                ),
            }
        )

        if drop_incomplete:
            result = result.dropna(
                subset=["arrival_s", "completion_s"]
            ).reset_index(drop=True)

        return result

    def query_success(self) -> pd.Series:
        """Return a boolean Series indexed by query_id indicating whether each
        completed query succeeded.

        Parses the ``details["success"]`` flag from every ``COMPLETION`` event.
        Only queries that have a ``COMPLETION`` row with a parseable ``success``
        field are included.  Analogous to
        :meth:`~autoslo.workload_execution.trace.Trace.was_aborted`.
        """
        df = self.df

        completions = df[df["event_type"] == EventType.COMPLETION.value].copy()
        if completions.empty or "details" not in completions.columns:
            return pd.Series(dtype=bool, name="success")

        # Take the first COMPLETION row per query_id.
        completions = completions.drop_duplicates(
            subset=["query_id"], keep="first"
        )
        completions = completions[completions["query_id"].notna()]

        completions["_success"] = completions["details"].apply(
            lambda d: d.get("success", False)
        )
        valid = completions[completions["_success"].notna()]
        return (
            valid.set_index("query_id")["_success"]
            .astype(bool)
            .rename("success")
        )

    def prediction_accuracy_df(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per query containing actual latency,
        predicted latency, RPU, and error metrics.
        """
        

        query_data: dict[Any, dict[str, float | int | None]] = {}
        for _, row in self.df.iterrows():
            query_id = row.get("query_id")
            event_type = row.get("event_type")
            if query_id is None or event_type is None:
                continue

            if query_id not in query_data:
                query_data[query_id] = {
                    "arrival_time": None,
                    "completion_time": None,
                    "predicted_latency": None,
                    "rpu": None,
                    "is_censored_target": False,
                }

            if event_type == "arrival":
                query_data[query_id]["arrival_time"] = float(row["rel_time_s"])
            elif event_type == "completion":
                query_data[query_id]["completion_time"] = float(
                    row["rel_time_s"]
                )

                query_data[query_id]["is_censored_target"] = not row[
                    "details"
                ].get("success", False)
            elif event_type in {"query_routed", "latency_update"}:
                pred = row["details"].get("latency_s")
                if pred is not None:
                    query_data[query_id]["predicted_latency"] = pred
                if (
                    event_type == "query_routed"
                    and row.get("cluster_name") is not None
                ):
                    query_data[query_id]["rpu"] = Cluster.rpu_for_cluster_name(
                        row["cluster_name"]
                    )

        results_df = pd.DataFrame.from_dict(query_data, orient="index")
        if results_df.empty:
            return results_df

        results_df["actual_latency"] = (
            results_df["completion_time"] - results_df["arrival_time"]
        )
        results_df = results_df.dropna(
            subset=["actual_latency", "predicted_latency", "rpu"]
        )
        results_df = results_df[
            (results_df["actual_latency"] > 0)
            & (results_df["predicted_latency"] > 0)
        ]

        results_df["abs_error"] = (
            results_df["actual_latency"] - results_df["predicted_latency"]
        ).abs()
        results_df["factor_error"] = (
            results_df["predicted_latency"] / results_df["actual_latency"]
        )
        results_df["q_error"] = results_df[["factor_error"]].apply(
            lambda row: max(row["factor_error"], 1.0 / row["factor_error"]),
            axis=1,
        )
        results_df["is_censored_target"] = results_df[
            "is_censored_target"
        ].astype(bool)
        results_df["underprediction_error_s"] = (
            results_df["actual_latency"] - results_df["predicted_latency"]
        ).clip(lower=0.0)
        results_df["rpu"] = results_df["rpu"].astype(int)
        return results_df
