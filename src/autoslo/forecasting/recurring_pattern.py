"""
recurring_pattern.py
====================
Recurring query pattern extraction for workload forecasting (Layer 3a).

A **recurring pattern** is the set of query templates that recur reliably in
specific weekly time slots.  Templates whose reliability exceeds a configurable
threshold are treated as *deterministic* — they will appear in every forecast
scenario for the relevant slot.

The *residual* is everything not in the recurring pattern.  It can be modelled
stochastically (e.g. with the PhaseHMM or per-slot resampling).

Terminology
-----------
**Slot** : A fixed-width, non-overlapping time bin that tiles the week.
  With ``slot_minutes=15`` there are ``7 × 96 = 672`` slots per week;
  with ``slot_minutes=60`` there are ``7 × 24 = 168``.  Slot 0 is
  Monday 00:00–00:14, slot 1 is Monday 00:15–00:29, etc.

Typical usage
-------------
>>> from autoslo.forecasting.recurring_pattern import RecurringPatternExtractor
>>>
>>> extractor = RecurringPatternExtractor(
...     slot_minutes=15, reliability_threshold=0.8,
... )
>>> result = extractor.fit(query_log_df)
>>>
>>> # Deterministic forecasts for recurring templates
>>> recurring_arrivals = extractor.generate_recurring(
...     start_time=datetime(2026, 3, 1, 8, 0),
...     n_slots=16,
... )
>>>
>>> # Everything else
>>> residual_df = result.residual_log
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RecurringTemplate:
    """A single template identified as part of the recurring pattern.

    Attributes
    ----------
    template_id : str
        The template identifier.
    reliable_slots : dict[int, float]
        Mapping from weekly slot index to reliability
        (only slots with reliability ≥ threshold).
    expected_count_per_slot : dict[int, float]
        Mean number of queries of this template per day
        in each reliable slot.
    """

    template_id: str
    reliable_slots: dict[int, float] = field(default_factory=dict)
    expected_count_per_slot: dict[int, float] = field(default_factory=dict)


@dataclass
class RecurringPatternResult:
    """Output of :meth:`RecurringPatternExtractor.fit`.

    Attributes
    ----------
    recurring_templates : dict[str, RecurringTemplate]
        Template id → recurring info for every template that passed the
        reliability threshold in at least one weekly slot.
    residual_log : DataFrame
        The input query log with all recurring events removed.
        Same columns and dtypes as the input.
    reliability_threshold : float
        The threshold that was used.
    slot_minutes : int
        The slot width that was used.
    reliability_matrix : ndarray of shape (n_templates, n_slots)
        Full reliability matrix.  Row order matches ``template_ids``.
    template_ids : ndarray of shape (n_templates,)
        Sorted array of all template ids found in the log.
    """

    recurring_templates: dict[str, RecurringTemplate]
    residual_log: pd.DataFrame
    reliability_threshold: float
    slot_minutes: int
    reliability_matrix: NDArray[np.float64]
    template_ids: NDArray


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class RecurringPatternExtractor:
    """Extract recurring query patterns from a historical query log.

    Terminology
    -----------
    **Slot** — a fixed-width time bin that tiles the week (see module
    docstring).

    Parameters
    ----------
    slot_minutes : int
        Width of each slot in minutes.  Must evenly divide ``24 * 60``.
        Default 15 (gives 672 weekly slots).
    reliability_threshold : float
        A template is included in the recurring pattern for a given slot
        when it appears on at least this fraction of the calendar days
        that cover the slot.  Default 0.8.
    min_days : int
        A template must appear in a slot on at least this many distinct
        calendar days before reliability is considered meaningful.
        Prevents single-occurrence templates from being labelled
        recurring.  Default 2.
    """

    def __init__(
        self,
        slot_minutes: int = 15,
        reliability_threshold: float = 0.8,
        min_days: int = 2,
    ) -> None:
        if (24 * 60) % slot_minutes != 0:
            raise ValueError(
                f"An integer number of slots must fit in each day: "
                f"24*60 % {slot_minutes} != 0."
            )
        if not 0.0 < reliability_threshold <= 1.0:
            raise ValueError(
                f"reliability_threshold={reliability_threshold} must be in (0, 1]."
            )
        if min_days < 1:
            raise ValueError(f"min_days={min_days} must be >= 1.")
        self._slot_minutes = slot_minutes
        self._slots_per_day = 24 * 60 // slot_minutes
        self._reliability_threshold = reliability_threshold
        self._min_days = min_days

        # Populated by fit()
        self._result: Optional[RecurringPatternResult] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def slot_minutes(self) -> int:
        return self._slot_minutes

    @property
    def reliability_threshold(self) -> float:
        return self._reliability_threshold

    @property
    def result(self) -> RecurringPatternResult:
        if self._result is None:
            raise RuntimeError("Call fit() before accessing result.")
        return self._result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _weekly_slot(self, dt: datetime) -> int:
        """Map a datetime to a weekly slot index (fast path)."""
        minute_of_day = dt.hour * 60 + dt.minute
        return (
            dt.weekday() * self._slots_per_day
            + minute_of_day // self._slot_minutes
        )

    def _n_weekly_slots(self, slot_minutes: int) -> int:
        """Total number of weekly slots for a given slot width."""
        return 7 * self._slots_per_day

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        query_log: pd.DataFrame,
        timestamp_col: str = "timestamp",
        template_col: str = "template_id",
    ) -> RecurringPatternResult:
        """Compute reliability and extract recurring patterns.

        Parameters
        ----------
        query_log : DataFrame
            Must contain at least a datetime column and a string
            template-id column.
        timestamp_col : str
            Name of the datetime column.
        template_col : str
            Name of the string template-id column.

        Returns
        -------
        RecurringPatternResult
        """
        df = query_log[[timestamp_col, template_col]].copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df[template_col] = df[template_col].astype(str)
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # Assign weekly slot and calendar date to each row.
        df["_slot"] = df[timestamp_col].apply(self._weekly_slot)
        df["_date"] = df[timestamp_col].dt.date

        # Unique template ids (sorted).
        template_ids = np.sort(df[template_col].unique())
        tid_to_idx = {tid: i for i, tid in enumerate(template_ids)}

        n_templates = len(template_ids)
        n_slots = self._n_weekly_slots(self._slot_minutes)

        # ---- Per-slot day coverage (denominator) --------------------------
        # For each slot, how many distinct calendar dates have any query
        # in that slot?  This is the denominator for reliability.
        date_slot_pairs = df[["_date", "_slot"]].drop_duplicates()
        slot_day_counts = (
            date_slot_pairs.groupby("_slot")["_date"]
            .nunique()
            .reindex(range(n_slots), fill_value=0)
            .values.astype(np.float64)
        )

        # ---- Per-(template, slot) day presence (numerator) ----------------
        presence = np.zeros((n_templates, n_slots), dtype=np.float64)

        # Also track total counts for expected-count computation.
        total_counts = np.zeros((n_templates, n_slots), dtype=np.float64)

        grouped = (
            df.groupby([template_col, "_slot", "_date"])
            .size()
            .reset_index(name="_count")
        )
        for _, row in grouped.iterrows():
            tidx = tid_to_idx[row[template_col]]
            s = int(row["_slot"])
            presence[tidx, s] += 1  # one date's worth
            total_counts[tidx, s] += row["_count"]

        # ---- Reliability matrix -------------------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            reliability = np.where(
                slot_day_counts > 0,
                presence / slot_day_counts[np.newaxis, :],
                0.0,
            )

        # ---- Expected counts per occurrence day --------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_counts = np.where(
                presence > 0,
                total_counts / presence,
                0.0,
            )

        # ---- Build recurring templates ------------------------------------
        recurring_templates: dict[str, RecurringTemplate] = {}
        recurring_mask = np.zeros(len(df), dtype=bool)

        for tidx, tid in enumerate(template_ids):
            reliable_slots: dict[int, float] = {}
            expected: dict[int, float] = {}
            for s in range(n_slots):
                if (
                    reliability[tidx, s] >= self._reliability_threshold
                    and presence[tidx, s] >= self._min_days
                ):
                    reliable_slots[s] = float(reliability[tidx, s])
                    expected[s] = float(expected_counts[tidx, s])
            if reliable_slots:
                recurring_templates[tid] = RecurringTemplate(
                    template_id=tid,
                    reliable_slots=reliable_slots,
                    expected_count_per_slot=expected,
                )
                # Mark rows belonging to this template in reliable slots
                # as recurring (to remove from residual).
                mask = (df[template_col] == tid) & (
                    df["_slot"].isin(reliable_slots.keys())
                )
                recurring_mask |= mask.values

        residual_log = df.loc[
            ~recurring_mask, [timestamp_col, template_col]
        ].copy()
        residual_log.reset_index(drop=True, inplace=True)

        self._result = RecurringPatternResult(
            recurring_templates=recurring_templates,
            residual_log=residual_log,
            reliability_threshold=self._reliability_threshold,
            slot_minutes=self._slot_minutes,
            reliability_matrix=reliability,
            template_ids=template_ids,
        )

        logger.info(
            "Recurring patterns extracted: %d / %d templates, threshold=%.2f, "
            "slot=%d min, %d total slots.",
            len(recurring_templates),
            n_templates,
            self._reliability_threshold,
            self._slot_minutes,
            n_slots,
        )
        return self._result

    # ------------------------------------------------------------------
    # Generate recurring arrivals for a forecast horizon
    # ------------------------------------------------------------------

    def generate_recurring(
        self,
        start_time: datetime,
        n_slots: int,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Produce deterministic recurring arrivals for a forecast period.

        For each slot that contains a recurring template, the expected
        number of queries (rounded to the nearest int) is placed
        uniformly within the slot.

        Parameters
        ----------
        start_time : datetime
            Start of the forecast horizon (should be aligned to a slot
            boundary).
        n_slots : int
            Number of slots to forecast.
        random_state : int or None
            Seed for the uniform scatter within each slot.

        Returns
        -------
        DataFrame
            Columns ``['timestamp', 'template_id']`` where ``timestamp``
            is seconds since *start_time*.
        """
        result = self.result  # raises if not fitted
        rng = np.random.default_rng(random_state)

        rows: list[dict] = []
        slot_s = self._slot_minutes * 60

        for w in range(n_slots):
            slot_start = start_time + timedelta(seconds=w * slot_s)
            slot_idx = self._weekly_slot(slot_start)

            for rt in result.recurring_templates.values():
                if slot_idx in rt.reliable_slots:
                    n_queries = int(round(rt.expected_count_per_slot[slot_idx]))
                    for _ in range(n_queries):
                        offset_s = w * slot_s + rng.uniform(0, slot_s)
                        rows.append(
                            {
                                "timestamp": offset_s,
                                "template_id": rt.template_id,
                            }
                        )

        if not rows:
            return pd.DataFrame(columns=["timestamp", "template_id"])

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df
