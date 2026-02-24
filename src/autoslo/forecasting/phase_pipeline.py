"""
phase_pipeline.py
=================
End-to-end pipeline: raw query log → fitted PhaseHMM → sampled workload draws.

This module ties together the three components:
  1. Windowed binning  (raw queries  →  (counts, slots) arrays)
  2. HMM fitting       (counts, slots →  PhaseHMM)
  3. Workload sampling (PhaseHMM      →  ForecastResult + arrival streams)

It also exposes a BIC model-selection helper and a simple matplotlib
visualisation of the inferred phase structure.

Typical usage
-------------
>>> from datetime import datetime
>>> from autoslo.forecasting.phase_pipeline import PhasePipeline, PipelineConfig
>>>
>>> # Prepare a DataFrame with columns: ['timestamp', 'class_id']
>>> # (timestamp: datetime or float seconds since epoch; class_id: int 0..K-1)
>>>
>>> pipeline = PhasePipeline(PipelineConfig(n_states=4, window_minutes=15))
>>> pipeline.fit(query_log_df)
>>>
>>> start = datetime(2026, 3, 1, 8, 0, 0)          # Monday 08:00 — a window boundary
>>> forecast = pipeline.forecast(start, hours=4, n_draws=200)
>>>
>>> # Mean per-class load profile over the 4-hour forecast window
>>> print(forecast.mean_counts)           # shape (16, K)
>>>
>>> # One worst-case draw (95th-percentile total load)
>>> worst = pipeline.sampler.worst_case_draw(forecast, percentile=95)
>>> print(worst.arrivals.head())
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from autoslo.forecasting.phase_hmm import PhaseHMM, PhaseHMMConfig, PhaseHMMResult, _time_slot
from autoslo.forecasting.workload_sampler import (
    ForecastResult,
    WorkloadSampler,
    add_absolute_timestamps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Top-level configuration for the PhasePipeline.

    Parameters
    ----------
    window_minutes : int
        Width of each time window in minutes.  Must evenly divide 60.
        Smaller values give finer temporal resolution but noisier counts.
        Default: 15.
    n_states : int
        Number of hidden phases for the HMM.  Can be selected automatically
        via select_n_states().  Default: 4.
    alpha_0 : float
        Gamma prior shape on Poisson emission rates (see PhaseHMMConfig).
    beta_0 : float
        Gamma prior rate parameter.
    max_iter : int
        EM maximum iterations per restart.
    tol : float
        EM convergence tolerance.
    n_init : int
        Number of random EM restarts.
    sticky_transition : float
        Non-negative strength of a self-transition bias.  Higher values
        encourage longer phase durations.
    postprocess_min_duration_windows : int or None
        If set, post-process Viterbi-decoded phases to enforce a minimum
        duration of this many windows.
    random_state : int or None
        Global seed for reproducibility.
    """

    window_minutes: int = 15
    n_states: int = 4
    alpha_0: float = 2.0
    beta_0: float = 1.0
    max_iter: int = 200
    tol: float = 1e-4
    n_init: int = 5
    sticky_transition: float = 0.0
    postprocess_min_duration_windows: Optional[int] = None
    random_state: Optional[int] = None

    def to_hmm_config(self) -> PhaseHMMConfig:
        return PhaseHMMConfig(
            n_states=self.n_states,
            alpha_0=self.alpha_0,
            beta_0=self.beta_0,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            sticky_transition=self.sticky_transition,
            random_state=self.random_state,
        )


# ---------------------------------------------------------------------------
# Binning utilities
# ---------------------------------------------------------------------------


def bin_queries(
    query_log: pd.DataFrame,
    window_minutes: int = 15,
    timestamp_col: str = "timestamp",
    class_col: str = "class_id",
) -> tuple[NDArray[np.int64], NDArray[np.int64], pd.DatetimeIndex]:
    """Bin a query log into fixed-width windows.

    Parameters
    ----------
    query_log : DataFrame
        Must contain at least:
        - `timestamp_col`: datetime (or timezone-aware datetime) of each query.
        - `class_col`: integer class label 0..K-1.
    window_minutes : int
        Window width in minutes.
    timestamp_col, class_col : str
        Column names.

    Returns
    -------
    counts : array of shape (T, K)
        Per-window, per-class query counts.
    slots : array of shape (T,)
        Time-slot index in [0, 168) for each window:
        slot = day_of_week * 24 + hour_of_day.
    window_starts : DatetimeIndex of length T
        Wall-clock start time of each window (UTC or naive).
    """
    df = query_log.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)

    # Determine window boundaries
    t_min = df[timestamp_col].min().floor(f"{window_minutes}min")
    t_max = df[timestamp_col].max().ceil(f"{window_minutes}min")
    window_freq = f"{window_minutes}min"

    # Assign each query to a window index
    window_bin = pd.cut(
        df[timestamp_col],
        bins=pd.date_range(t_min, t_max, freq=window_freq),
        labels=False,
        right=False,
    )
    df["_window_bin"] = window_bin

    # Number of classes
    n_classes = int(df[class_col].max()) + 1
    n_windows = int(window_bin.max()) + 1

    # Build count matrix
    counts = np.zeros((n_windows, n_classes), dtype=np.int64)
    for (win_idx, cls), grp in df.groupby(["_window_bin", class_col]):
        counts[int(win_idx), int(cls)] = len(grp)

    # Build window start times
    window_starts = pd.date_range(t_min, periods=n_windows, freq=window_freq)

    # Build slot array
    slots = np.array(
        [_time_slot(ts.hour, ts.weekday()) for ts in window_starts],
        dtype=np.int64,
    )

    return counts, slots, window_starts


def enforce_min_duration(states: NDArray[np.int64], min_windows: int) -> NDArray[np.int64]:
    """Post-process a state sequence to enforce a minimum segment duration.

    This is a greedy merge: any segment shorter than min_windows is merged
    into its longer neighbor (or the only neighbor at the boundary).
    """
    if min_windows <= 1:
        return states.copy()

    states = states.copy()

    while True:
        # Build segments: (start_idx, end_idx_exclusive, state)
        segments: list[tuple[int, int, int]] = []
        start = 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start]:
                segments.append((start, i, int(states[start])))
                start = i

        short_indices = [i for i, (s, e, _) in enumerate(segments) if (e - s) < min_windows]
        if not short_indices:
            break

        for idx in short_indices:
            s, e, st = segments[idx]
            prev_seg = segments[idx - 1] if idx > 0 else None
            next_seg = segments[idx + 1] if idx + 1 < len(segments) else None

            if prev_seg is None and next_seg is None:
                continue
            if prev_seg is None:
                target_state = next_seg[2]
            elif next_seg is None:
                target_state = prev_seg[2]
            else:
                prev_len = prev_seg[1] - prev_seg[0]
                next_len = next_seg[1] - next_seg[0]
                target_state = prev_seg[2] if prev_len >= next_len else next_seg[2]

            states[s:e] = target_state

    return states


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class PhasePipeline:
    """End-to-end pipeline from query log to sampled forecast workloads.

    Parameters
    ----------
    config : PipelineConfig
        Combined configuration for binning, HMM, and sampling.

    Attributes (set after fit())
    ----------------------------
    model_ : PhaseHMM
        The fitted time-inhomogeneous HMM.
    sampler_ : WorkloadSampler
        Ready-to-use sampler wrapping the fitted model.
    counts_ : array (T, K)
        Training window counts.
    slots_ : array (T,)
        Training window time-slot indices.
    window_starts_ : DatetimeIndex (T,)
        Wall-clock start times of training windows.
    n_classes_ : int
        Number of query classes K.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.model_: Optional[PhaseHMM] = None
        self.sampler_: Optional[WorkloadSampler] = None
        self.counts_: Optional[NDArray] = None
        self.slots_: Optional[NDArray] = None
        self.window_starts_: Optional[pd.DatetimeIndex] = None
        self.n_classes_: Optional[int] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        query_log: pd.DataFrame,
        timestamp_col: str = "timestamp",
        class_col: str = "class_id",
    ) -> "PhasePipeline":
        """Fit the full pipeline on a historical query log.

        Parameters
        ----------
        query_log : DataFrame
            Must contain `timestamp_col` (datetime) and `class_col` (int 0..K-1).
        timestamp_col, class_col : str
            Column names.

        Returns
        -------
        self
        """
        cfg = self.config

        # Stage 1: bin queries into windows
        logger.info(
            "Binning queries into %d-minute windows …", cfg.window_minutes
        )
        counts, slots, window_starts = bin_queries(
            query_log,
            window_minutes=cfg.window_minutes,
            timestamp_col=timestamp_col,
            class_col=class_col,
        )
        T, K = counts.shape
        logger.info("  %d windows × %d classes", T, K)

        self.counts_ = counts
        self.slots_ = slots
        self.window_starts_ = window_starts
        self.n_classes_ = K

        # Stage 2: fit HMM
        logger.info("Fitting PhaseHMM (n_states=%d, %d restarts) …", cfg.n_states, cfg.n_init)
        hmm_cfg = cfg.to_hmm_config()
        model = PhaseHMM(hmm_cfg).fit(counts, slots)
        assert model.result_ is not None
        logger.info(
            "HMM fitted: ll=%.4f, bic=%.2f, %d iters",
            model.result_.log_likelihood,
            model.result_.bic,
            model.result_.n_iter,
        )

        self.model_ = model
        self.sampler_ = WorkloadSampler(model, window_minutes=cfg.window_minutes)
        return self

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def select_n_states(
        self,
        query_log: pd.DataFrame,
        n_states_range: list[int] | None = None,
        timestamp_col: str = "timestamp",
        class_col: str = "class_id",
    ) -> dict[int, float]:
        """Fit models for each candidate n_states and return BIC scores.

        Useful for choosing n_states before a full fit.  Does NOT modify
        the pipeline's fitted state.

        Parameters
        ----------
        query_log : DataFrame (same format as fit()).
        n_states_range : list of candidate state counts.

        Returns
        -------
        bic_scores : dict {n_states: bic_value}, sorted ascending by BIC.
        """
        cfg = self.config
        counts, slots, _ = bin_queries(
            query_log,
            window_minutes=cfg.window_minutes,
            timestamp_col=timestamp_col,
            class_col=class_col,
        )
        results = PhaseHMM.select_n_states(
            counts,
            slots,
            n_states_range=n_states_range,
            config_kwargs={
                "alpha_0": cfg.alpha_0,
                "beta_0": cfg.beta_0,
                "max_iter": cfg.max_iter,
                "tol": cfg.tol,
                "n_init": cfg.n_init,
                "sticky_transition": cfg.sticky_transition,
                "random_state": cfg.random_state,
            },
        )
        return {n: r.bic for n, r in results.items()}

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def _check_fitted(self) -> tuple[PhaseHMM, WorkloadSampler]:
        if self.model_ is None or self.sampler_ is None:
            raise RuntimeError("Pipeline has not been fitted.  Call fit() first.")
        return self.model_, self.sampler_

    def forecast(
        self,
        start_time: datetime,
        hours: float | None = None,
        n_windows: int | None = None,
        n_draws: int = 100,
        random_state: int | None = None,
    ) -> ForecastResult:
        """Draw M synthetic workload trajectories for a future time window.

        The forecast starts exactly at `start_time`, which must align with a
        window boundary (i.e., minutes divisible by window_minutes).

        Provide exactly one of `hours` or `n_windows` to specify the duration.

        Parameters
        ----------
        start_time : datetime
            Wall-clock start of the forecast window.  Should be a window
            boundary; a warning is issued if it is not.
        hours : float, optional
            Forecast duration in hours.
        n_windows : int, optional
            Forecast duration in windows.
        n_draws : int
            Number of independent stochastic draws M.
        random_state : int or None
            Seed for reproducibility.

        Returns
        -------
        ForecastResult
        """
        _, sampler = self._check_fitted()
        cfg = self.config

        # Validate start_time alignment
        remainder = start_time.minute % cfg.window_minutes
        if remainder != 0:
            warnings.warn(
                f"start_time {start_time} is not aligned to a {cfg.window_minutes}-minute "
                f"window boundary (remainder = {remainder} min).  Results may be misleading.",
                UserWarning,
                stacklevel=2,
            )

        # Compute n_windows
        if hours is not None and n_windows is not None:
            raise ValueError("Provide exactly one of `hours` or `n_windows`.")
        if hours is not None:
            n_windows = max(1, int(round(hours * 60 / cfg.window_minutes)))
        elif n_windows is None:
            raise ValueError("Provide either `hours` or `n_windows`.")

        logger.info(
            "Sampling %d draws over %d windows starting at %s …",
            n_draws, n_windows, start_time,
        )
        return sampler.sample(
            start_time=start_time,
            n_windows=n_windows,
            n_draws=n_draws,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Convenience: infer phases on training data
    # ------------------------------------------------------------------

    def infer_training_phases(
        self,
        min_duration_windows: int | None = None,
        apply_postprocess: bool | None = None,
    ) -> NDArray[np.int64]:
        """Run Viterbi decoding on the training data.

        Parameters
        ----------
        min_duration_windows : int or None
            Minimum number of windows per phase after post-processing.  If None,
            falls back to config.postprocess_min_duration_windows.
        apply_postprocess : bool or None
            Whether to apply the post-decode minimum-duration constraint.
            If None, this is enabled only when min_duration_windows is set.

        Returns
        -------
        states : array of shape (T,)
            Most probable hidden state (phase) for each training window.
        """
        model, _ = self._check_fitted()
        assert self.counts_ is not None and self.slots_ is not None

        states = model.viterbi(self.counts_, self.slots_)

        if min_duration_windows is None:
            min_duration_windows = self.config.postprocess_min_duration_windows
        if apply_postprocess is None:
            apply_postprocess = min_duration_windows is not None

        if apply_postprocess and min_duration_windows is not None:
            states = enforce_min_duration(states, min_duration_windows)

        return states

    # ------------------------------------------------------------------
    # Convenience: describe the fitted emission rates
    # ------------------------------------------------------------------

    def phase_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising the fitted Poisson rates per phase.

        Returns
        -------
        DataFrame with index = phase index, columns = class_0 … class_{K-1},
        plus a 'total_rate' column (sum across classes).
        """
        model, _ = self._check_fitted()
        assert model.result_ is not None
        lam = model.result_.lam  # (S, K)
        S, K = lam.shape
        df = pd.DataFrame(
            lam,
            index=[f"phase_{s}" for s in range(S)],
            columns=[f"class_{k}" for k in range(K)],
        )
        df["total_rate"] = lam.sum(axis=1)
        return df.sort_values("total_rate", ascending=False)


# ---------------------------------------------------------------------------
# Visualisation helpers  (optional; requires matplotlib)
# ---------------------------------------------------------------------------


def plot_phase_assignment(
    window_starts: pd.DatetimeIndex,
    states: NDArray[np.int64],
    counts: NDArray[np.int64] | None = None,
    title: str = "Inferred workload phases",
) -> None:
    """Plot inferred phases as coloured bands over total arrival rate.

    Parameters
    ----------
    window_starts : DatetimeIndex of window start times.
    states : array (T,) of phase indices.
    counts : optional (T, K) count array; if provided, total rate is plotted.
    title : plot title.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot.")
        return

    n_states = int(states.max()) + 1
    cmap = plt.get_cmap("tab10", n_states)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Shade phase regions
    prev_state = states[0]
    prev_start = window_starts[0]
    for i in range(1, len(states)):
        if states[i] != prev_state or i == len(states) - 1:
            ax.axvspan(prev_start, window_starts[i], alpha=0.25, color=cmap(prev_state))
            prev_state = states[i]
            prev_start = window_starts[i]

    # Optionally overlay total arrival rate
    if counts is not None:
        total_rate = counts.sum(axis=1)
        ax.plot(window_starts, total_rate, color="black", linewidth=0.8, label="Total arrivals/window")
        ax.set_ylabel("Queries per window")

    # Legend
    patches = [
        mpatches.Patch(color=cmap(s), alpha=0.6, label=f"Phase {s}")
        for s in range(n_states)
    ]
    ax.legend(handles=patches, loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Time")
    fig.tight_layout()
    plt.show()


def plot_forecast_bands(
    forecast: ForecastResult,
    start_time: datetime,
    window_minutes: int = 15,
    class_names: list[str] | None = None,
    title: str = "Forecast: per-class arrival rate",
) -> None:
    """Plot mean ± P10/P90 bands for each class over the forecast window.

    Parameters
    ----------
    forecast : ForecastResult from PhasePipeline.forecast().
    start_time : datetime start of the forecast.
    window_minutes : int window width (for x-axis labelling).
    class_names : optional list of K class name strings.
    title : plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot.")
        return

    W, K = forecast.mean_counts.shape
    dt = timedelta(minutes=window_minutes)
    times = [start_time + i * dt for i in range(W)]

    class_names = class_names or [f"class_{k}" for k in range(K)]
    cmap = plt.get_cmap("tab10", K)

    fig, ax = plt.subplots(figsize=(14, 5))
    for k in range(K):
        color = cmap(k)
        ax.plot(times, forecast.mean_counts[:, k], color=color, label=class_names[k])
        ax.fill_between(
            times,
            forecast.p10_counts[:, k],
            forecast.p90_counts[:, k],
            color=color,
            alpha=0.2,
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Queries per window (mean ± P10/P90)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def plot_state_occupancy(
    states: NDArray[np.int64],
    title: str = "Phase occupancy",
) -> None:
    """Plot the fraction of windows spent in each phase."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot.")
        return

    counts = np.bincount(states)
    frac = counts / max(1, counts.sum())

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(frac)), frac)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Fraction of windows")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_phase_duration_hist(
    states: NDArray[np.int64],
    window_minutes: int = 15,
    title: str = "Phase duration distribution",
    min_duration_windows: int | None = None,
) -> None:
    """Plot a histogram of contiguous phase durations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot.")
        return

    lengths = []
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            lengths.append(i - start)
            start = i

    lengths_min = np.array(lengths) * window_minutes

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(lengths_min, bins=20, color="#4C72B0", alpha=0.8)
    if min_duration_windows is not None:
        ax.axvline(min_duration_windows * window_minutes, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Segment count")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_transition_heatmap(
    model_or_A: PhaseHMM | NDArray[np.float64],
    title: str = "Mean transition matrix",
) -> None:
    """Plot the mean transition matrix across all time slots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot.")
        return

    if isinstance(model_or_A, PhaseHMM):
        if model_or_A.result_ is None:
            raise ValueError("PhaseHMM must be fitted before plotting transitions.")
        A = model_or_A.result_.A
    else:
        A = model_or_A

    A_mean = A.mean(axis=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A_mean, cmap="viridis", vmin=0.0, vmax=A_mean.max())
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def plot_class_mix_over_time(
    window_starts: pd.DatetimeIndex,
    counts: NDArray[np.int64],
    normalize: bool = False,
    title: str = "Class mix over time",
) -> None:
    """Plot a stacked area chart of per-class counts over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot.")
        return

    data = counts.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = data / np.maximum(row_sums, 1e-12)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.stackplot(window_starts, data.T, labels=[f"class_{k}" for k in range(data.shape[1])], alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction" if normalize else "Queries per window")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    plt.show()
