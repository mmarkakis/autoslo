"""
workload_sampler.py
===================
Ancestral sampling from a fitted PhaseHMM to produce synthetic workload
trajectories, and within-window Poisson scatter to generate a realistic
query arrival stream.

The forecast always starts at a window boundary (no partial-window handling
needed), and produces M independent trajectory draws over W future windows.

Sampling procedure
------------------
Given a forecast starting at wall-clock time `start_time` (a window boundary):

  1.  Derive (h₀, d₀) = (hour-of-day, day-of-week) of start_time.
  2.  Sample z₁ ~ π^(h₀, d₀)                   [initial phase]
  3.  For each future window t = 1, …, W:
        a.  Emit n_{t,k} ~ Poisson(λ_{z_t, k})  [per-class counts]
        b.  Compute (h_t, d_t) for this window's wall-clock time.
        c.  Transition z_{t+1} ~ Categorical(A^(h_t, d_t)[z_t, :])
  4.  Scatter n_{t,k} arrivals of class k uniformly within [τ_t, τ_t + Δt)
      (homogeneous within-window Poisson process).
  5.  Merge and sort to produce a (timestamp, class) arrival stream.

Repeat M times for a forecast distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from autoslo.forecasting.phase_hmm import PhaseHMM, PhaseHMMResult, _time_slot


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class WorkloadDraw:
    """A single sampled synthetic workload trajectory.

    Attributes
    ----------
    window_counts : array of shape (W, K)
        Per-window, per-class query counts.
    window_states : array of shape (W,)
        Sampled hidden phase index for each window.
    arrivals : DataFrame with columns ['timestamp', 'class_id']
        Full arrival stream produced by within-window Poisson scatter,
        sorted by timestamp.  `timestamp` is a float offset in seconds
        from `start_time`.
    """

    window_counts: NDArray[np.int64]   # (W, K)
    window_states: NDArray[np.int64]   # (W,)
    arrivals: pd.DataFrame             # columns: timestamp (float s), class_id (int)


@dataclass
class ForecastResult:
    """Summary statistics over M workload draws.

    Attributes
    ----------
    draws : list of WorkloadDraw, length M
    mean_counts : array of shape (W, K) — mean per-class count per window
    p10_counts  : array of shape (W, K) — 10th percentile
    p90_counts  : array of shape (W, K) — 90th percentile
    modal_states : array of shape (W,)  — most frequent sampled state per window
    """

    draws: list[WorkloadDraw]
    mean_counts: NDArray[np.float64]  # (W, K)
    p10_counts: NDArray[np.float64]   # (W, K)
    p90_counts: NDArray[np.float64]   # (W, K)
    modal_states: NDArray[np.int64]   # (W,)


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------


class WorkloadSampler:
    """Draw synthetic workload trajectories from a fitted PhaseHMM.

    Parameters
    ----------
    model : PhaseHMM
        A fitted model (model.result_ must not be None).
    window_minutes : int
        Width of each time window in minutes.  Must match the value used
        when binning the training data.
    """

    def __init__(self, model: PhaseHMM, window_minutes: int = 15) -> None:
        if model.result_ is None:
            raise ValueError("PhaseHMM must be fitted before sampling.")
        self._result: PhaseHMMResult = model.result_
        self.window_minutes = window_minutes
        self._window_seconds = window_minutes * 60

    # ------------------------------------------------------------------
    # Single draw
    # ------------------------------------------------------------------

    def _sample_one(
        self,
        start_time: datetime,
        n_windows: int,
        n_classes: int,
        rng: np.random.Generator,
    ) -> WorkloadDraw:
        """Draw one synthetic workload trajectory.

        Parameters
        ----------
        start_time : datetime
            Wall-clock start of the forecast window (must be a window boundary).
        n_windows : W
            Number of windows to forecast.
        n_classes : K
            Number of query classes (should match training data).
        rng : numpy Generator
            Random number generator for reproducibility.

        Returns
        -------
        WorkloadDraw
        """
        r = self._result
        S = r.lam.shape[0]
        dt = timedelta(minutes=self.window_minutes)

        window_counts = np.empty((n_windows, n_classes), dtype=np.int64)
        window_states = np.empty(n_windows, dtype=np.int64)

        # --- Sample initial state ---
        slot_0 = _time_slot(start_time.hour, start_time.weekday())
        pi_0 = r.pi[slot_0]  # (S,)
        z = int(rng.choice(S, p=pi_0))

        # --- Ancestral sampling ---
        all_timestamps: list[float] = []
        all_class_ids: list[int] = []

        for t in range(n_windows):
            window_start: datetime = start_time + t * dt
            wall_offset_s: float = t * self._window_seconds  # seconds from start_time

            # Emit: draw per-class counts from Poisson(λ_z)
            rates = r.lam[z]                                   # (K,)
            counts_t = rng.poisson(rates).astype(np.int64)    # (K,)
            window_counts[t] = counts_t
            window_states[t] = z

            # Within-window scatter: place each arrival uniformly in [0, Δt)
            for k, n_k in enumerate(counts_t):
                if n_k > 0:
                    offsets = rng.uniform(0.0, self._window_seconds, size=n_k)
                    all_timestamps.extend((wall_offset_s + offsets).tolist())
                    all_class_ids.extend([k] * n_k)

            # Transition to next state
            if t < n_windows - 1:
                slot_t = _time_slot(window_start.hour, window_start.weekday())
                A_t = r.A[slot_t, z]  # (S,) — transition probabilities from state z
                z = int(rng.choice(S, p=A_t))

        # Build arrival DataFrame, sorted by timestamp
        arrivals = pd.DataFrame(
            {"timestamp": all_timestamps, "class_id": all_class_ids}
        ).sort_values("timestamp").reset_index(drop=True)

        return WorkloadDraw(
            window_counts=window_counts,
            window_states=window_states,
            arrivals=arrivals,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(
        self,
        start_time: datetime,
        n_windows: int,
        n_draws: int = 100,
        random_state: int | None = None,
    ) -> ForecastResult:
        """Draw M synthetic workload trajectories.

        Parameters
        ----------
        start_time : datetime
            Wall-clock start of the forecast (must align with a window boundary).
        n_windows : int
            Number of windows W to forecast (total duration = W × window_minutes).
        n_draws : int
            Number of independent trajectory samples M.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        ForecastResult
            Contains all M draws plus aggregate statistics.
        """
        rng = np.random.default_rng(random_state)
        n_classes = self._result.lam.shape[1]

        draws: list[WorkloadDraw] = []
        for i in range(n_draws):
            draw = self._sample_one(start_time, n_windows, n_classes, rng)
            draws.append(draw)

        # --- Aggregate statistics ---
        # Stack all draws: (M, W, K)
        all_counts = np.stack([d.window_counts for d in draws], axis=0).astype(float)

        mean_counts = all_counts.mean(axis=0)                          # (W, K)
        p10_counts = np.percentile(all_counts, 10, axis=0)             # (W, K)
        p90_counts = np.percentile(all_counts, 90, axis=0)             # (W, K)

        # Modal state per window across draws
        all_states = np.stack([d.window_states for d in draws], axis=0)  # (M, W)
        modal_states = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=self._result.lam.shape[0]).argmax(),
            axis=0,
            arr=all_states,
        )

        return ForecastResult(
            draws=draws,
            mean_counts=mean_counts,
            p10_counts=p10_counts,
            p90_counts=p90_counts,
            modal_states=modal_states,
        )

    def worst_case_draw(
        self,
        forecast: ForecastResult,
        percentile: float = 95.0,
    ) -> WorkloadDraw:
        """Return the draw whose total query count is at the given percentile.

        Useful for provisioning under a worst-case workload scenario.

        Parameters
        ----------
        forecast : ForecastResult returned by sample().
        percentile : float in (0, 100).

        Returns
        -------
        The WorkloadDraw whose summed total count is closest to the
        given percentile of the distribution of total counts across draws.
        """
        totals = np.array([d.window_counts.sum() for d in forecast.draws])
        threshold = np.percentile(totals, percentile)
        idx = int(np.argmin(np.abs(totals - threshold)))
        return forecast.draws[idx]


# ---------------------------------------------------------------------------
# Utility: attach absolute datetimes to an arrivals DataFrame
# ---------------------------------------------------------------------------


def add_absolute_timestamps(
    arrivals: pd.DataFrame,
    start_time: datetime,
) -> pd.DataFrame:
    """Convert relative offset timestamps (seconds) to absolute datetimes.

    Parameters
    ----------
    arrivals : DataFrame with a 'timestamp' column of float seconds since
               start_time.
    start_time : datetime reference point (the forecast window start).

    Returns
    -------
    Copy of arrivals with an additional 'datetime' column.
    """
    arrivals = arrivals.copy()
    arrivals["datetime"] = arrivals["timestamp"].apply(
        lambda s: start_time + timedelta(seconds=float(s))
    )
    return arrivals
