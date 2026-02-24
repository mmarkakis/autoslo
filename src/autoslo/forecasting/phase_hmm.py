"""
phase_hmm.py
============
Time-Inhomogeneous Hidden Markov Model for workload phase detection.

Each hidden state represents a recurring **phase** of the workload
(e.g. "quiet overnight", "morning ramp-up", "midday peak").  Observations
are per-window class-count vectors modelled as independent Poisson random
variables.  The transition matrix is conditioned on the (hour-of-day,
day-of-week) of the current window, capturing daily/weekly seasonality.

Mathematical summary
--------------------
  Emission:      n_{t,k} | z_t=s  ~  Poisson(λ_{s,k})
  Transition:    P(z_{t+1}=j | z_t=i, h_t, d_t)  =  A^(h,d)[i, j]
  Initial state: P(z_1=s | h_1, d_1)  =  π^(h,d)[s]
  Prior on λ:    λ_{s,k}  ~  Gamma(alpha_0, beta_0)  (MAP regularisation)

The MAP M-step update for the rates is:
    λ̂_{s,k} = (alpha_0 - 1 + Σ_t γ_t(s) n_{t,k}) / (beta_0 + Σ_t γ_t(s))

Fitting uses the standard Baum-Welch (EM) algorithm, modified to handle
the per-timestep transition matrix.  Multiple random restarts are run and
the run with the highest final log-likelihood is kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOURS_PER_DAY = 24
_DAYS_PER_WEEK = 7
_N_TIME_SLOTS = _HOURS_PER_DAY * _DAYS_PER_WEEK  # 168


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PhaseHMMConfig:
    """Hyperparameters for the Time-Inhomogeneous Phase HMM.

    Parameters
    ----------
    n_states:
        Number of hidden phases (S).  Select via BIC over held-out data;
        typical range 2–8.
    alpha_0:
        Shape parameter of the Gamma prior on Poisson rates λ_{s,k}.
        Must be > 1 so the MAP estimate is well-defined; values near 2
        give light regularisation.
    beta_0:
        Rate parameter of the Gamma prior.  The prior mean of λ_{s,k} is
        alpha_0 / beta_0.  Keep small (e.g. 1.0) for a weak prior.
    max_iter:
        Maximum number of EM iterations per restart.
    tol:
        Convergence threshold on the per-iteration change in log-likelihood.
    n_init:
        Number of random restarts.  The best run (highest final log-lik)
        is retained.
    sticky_transition:
        Non-negative strength of a self-transition bias.  When > 0, the
        M-step adds this value to the diagonal of each transition row,
        encouraging longer phase durations.
    random_state:
        Seed for reproducibility.  If None, results differ across runs.
    """

    n_states: int = 4
    alpha_0: float = 2.0
    beta_0: float = 1.0
    max_iter: int = 200
    tol: float = 1e-4
    n_init: int = 5
    sticky_transition: float = 0.0
    random_state: Optional[int] = None


@dataclass
class PhaseHMMResult:
    """All fitted parameters of a trained PhaseHMM."""

    # λ[s, k]: Poisson rates — shape (n_states, n_classes)
    lam: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # A[h*7+d, i, j]: transition matrices — shape (168, n_states, n_states)
    A: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # pi[h*7+d, s]: initial state distributions — shape (168, n_states)
    pi: NDArray[np.float64] = field(default_factory=lambda: np.array([]))

    # Log-likelihood of training data under the fitted model
    log_likelihood: float = -np.inf

    # BIC score (lower is better)
    bic: float = np.inf

    # Number of EM iterations executed
    n_iter: int = 0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _time_slot(hour: int, dow: int) -> int:
    """Map (hour-of-day, day-of-week) to a flat index in [0, 168)."""
    return dow * _HOURS_PER_DAY + hour


def _poisson_log_pmf(n: NDArray, lam: NDArray) -> NDArray:
    """Element-wise log-PMF of Poisson(lam) at counts n.

    Uses the stable formula:  n*log(lam) - lam - log(n!)
    Broadcasting: n can be (..., K), lam can be (S, K); result is (..., S).
    """
    # n: (T, K), lam: (S, K)  →  log_pmf: (T, S)
    # We sum over the K class dimension.
    log_lam = np.log(np.maximum(lam, 1e-300))  # (S, K)
    # n[:, None, :] shape: (T, 1, K); log_lam[None, :, :]: (1, S, K)
    per_class = n[:, None, :] * log_lam[None, :, :] - lam[None, :, :] - _log_factorial(n[:, None, :])
    return per_class.sum(axis=-1)  # (T, S)


# Pre-compute log-factorials up to a safe maximum count.
def _log_factorial(n: NDArray) -> NDArray:
    """Vectorised log(n!) using scipy.special.gammaln for numerical stability.

    gammaln(n+1) == log(n!) for integer n.
    """
    from scipy.special import gammaln  # scipy is available in the autoslo env

    return gammaln(n.astype(float) + 1.0)


def _normalise_rows(M: NDArray, min_val: float = 1e-300) -> NDArray:
    """Row-normalise a 2-D array, clipping denominators away from zero."""
    row_sums = M.sum(axis=-1, keepdims=True)
    return M / np.maximum(row_sums, min_val)


# ---------------------------------------------------------------------------
# Forward-backward (E-step)
# ---------------------------------------------------------------------------


def _forward(
    log_emit: NDArray,  # (T, S)
    A_seq: NDArray,     # (T, S, S)  — A_seq[t] is used to go from t → t+1
    log_pi: NDArray,    # (S,)
) -> tuple[NDArray, NDArray]:
    """Log-domain forward pass.

    Returns
    -------
    log_alpha : (T, S)
        log α_t(s) = log p(n_{1:t}, z_t = s)
    log_scale : (T,)
        Per-step normalisation constants (for numerical stability).
    """
    T, S = log_emit.shape
    log_alpha = np.empty((T, S))

    # t = 0
    log_alpha[0] = log_pi + log_emit[0]
    log_scale = np.empty(T)
    c0 = np.logaddexp.reduce(log_alpha[0])
    log_alpha[0] -= c0
    log_scale[0] = c0

    for t in range(1, T):
        # log p(z_t | z_{t-1}) needs the transition from t-1 to t.
        # log_alpha[t-1]: (S,); A_seq[t-1]: (S, S)
        # log p(z_{t-1} → z_t) = log A_seq[t-1][i, j]
        # log α_t(j) = log Σ_i α_{t-1}(i) A_{i,j} + log_emit[t, j]
        # In log domain: logsumexp over i of (log_alpha[t-1, i] + log A[i, j])
        log_trans = log_alpha[t - 1, :, None] + np.log(np.maximum(A_seq[t - 1], 1e-300))  # (S, S)
        log_alpha[t] = np.logaddexp.reduce(log_trans, axis=0) + log_emit[t]
        c = np.logaddexp.reduce(log_alpha[t])
        log_alpha[t] -= c
        log_scale[t] = c

    return log_alpha, log_scale


def _backward(
    log_emit: NDArray,  # (T, S)
    A_seq: NDArray,     # (T, S, S)
) -> NDArray:
    """Log-domain backward pass.

    Returns
    -------
    log_beta : (T, S)
        log β_t(s) = log p(n_{t+1:T} | z_t = s), normalised for stability.
    """
    T, S = log_emit.shape
    log_beta = np.zeros((T, S))

    for t in range(T - 2, -1, -1):
        # log β_t(i) = log Σ_j A[i,j] * emit(t+1,j) * β_{t+1}(j)
        log_terms = (
            np.log(np.maximum(A_seq[t], 1e-300))  # (S, S)
            + log_emit[t + 1][None, :]             # (1, S)
            + log_beta[t + 1][None, :]             # (1, S)
        )  # (S, S)
        log_beta[t] = np.logaddexp.reduce(log_terms, axis=1)  # (S,)
        # Normalise for stability
        c = np.logaddexp.reduce(log_beta[t])
        log_beta[t] -= c

    return log_beta


def _e_step(
    counts: NDArray,    # (T, K)
    slots: NDArray,     # (T,) int, values in [0, 168)
    lam: NDArray,       # (S, K)
    A: NDArray,         # (168, S, S)
    pi: NDArray,        # (168, S)
) -> tuple[NDArray, NDArray, float]:
    """One E-step: compute γ, ξ, and the log-likelihood.

    Returns
    -------
    gamma : (T, S)   — P(z_t = s | observations)
    xi    : (T-1, S, S) — P(z_t=i, z_{t+1}=j | observations)
    log_likelihood : float
    """
    T, _ = counts.shape
    S = lam.shape[0]

    log_emit = _poisson_log_pmf(counts, lam)  # (T, S)

    # Build per-timestep transition matrices (indexed by slot at time t)
    A_seq = A[slots[:-1]]  # (T-1, S, S); A_seq[t] is transition from t → t+1

    log_pi = np.log(np.maximum(pi[slots[0]], 1e-300))  # (S,)

    log_alpha, log_scale = _forward(log_emit, A_seq, log_pi)
    log_beta = _backward(log_emit, A_seq)

    # γ_t(s) ∝ α_t(s) β_t(s)
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)  # (T, S)

    # ξ_t(i,j) ∝ α_t(i) A_t(i,j) emit(t+1,j) β_{t+1}(j)
    # shape: (T-1, S, S)
    log_xi = (
        log_alpha[:-1, :, None]                                          # (T-1, S, 1)
        + np.log(np.maximum(A_seq, 1e-300))                              # (T-1, S, S)
        + log_emit[1:, None, :]                                          # (T-1, 1, S)
        + log_beta[1:, None, :]                                          # (T-1, 1, S)
    )
    # Normalise each t slice
    log_xi_sum = np.logaddexp.reduce(log_xi.reshape(T - 1, -1), axis=1)  # (T-1,)
    log_xi -= log_xi_sum[:, None, None]
    xi = np.exp(log_xi)  # (T-1, S, S)

    log_likelihood = log_scale.sum()

    return gamma, xi, float(log_likelihood)


# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------


def _m_step(
    counts: NDArray,   # (T, K)
    slots: NDArray,    # (T,) int
    gamma: NDArray,    # (T, S)
    xi: NDArray,       # (T-1, S, S)
    alpha_0: float,
    beta_0: float,
    sticky_transition: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """One M-step: update λ, A, π from sufficient statistics.

    Returns
    -------
    lam_new : (S, K)
    A_new   : (168, S, S)
    pi_new  : (168, S)
    """
    T, K = counts.shape
    S = gamma.shape[1]

    # --- Update λ (MAP with Gamma prior) ---
    # Numerator: (alpha_0 - 1) + Σ_t γ_t(s) n_{t,k}
    # Denominator: beta_0 + Σ_t γ_t(s)
    # gamma: (T, S), counts: (T, K) → weighted_counts: (S, K)
    weighted_counts = gamma.T @ counts  # (S, K)
    gamma_sum = gamma.sum(axis=0)       # (S,)
    lam_new = (alpha_0 - 1.0 + weighted_counts) / (beta_0 + gamma_sum[:, None])
    lam_new = np.maximum(lam_new, 1e-10)

    # --- Update A per time-slot ---
    A_new = np.zeros((_N_TIME_SLOTS, S, S))
    A_count = np.zeros((_N_TIME_SLOTS, S, S))

    for t in range(T - 1):
        slot = int(slots[t])
        A_count[slot] += xi[t]  # (S, S)

    for slot in range(_N_TIME_SLOTS):
        if sticky_transition > 0.0:
            A_count[slot].flat[:: S + 1] += sticky_transition
        row_sums = A_count[slot].sum(axis=1, keepdims=True)
        if row_sums.max() < 1e-12:
            # No observations for this slot: use uniform
            A_new[slot] = np.full((S, S), 1.0 / S)
        else:
            A_new[slot] = A_count[slot] / np.maximum(row_sums, 1e-300)

    # --- Update π per time-slot ---
    pi_new = np.zeros((_N_TIME_SLOTS, S))
    pi_count = np.zeros((_N_TIME_SLOTS, S))

    # Only the first timestep of each sequence contributes to π.
    # Here we treat the entire training data as one long sequence;
    # the first window's slot contributes.
    pi_count[int(slots[0])] += gamma[0]

    for slot in range(_N_TIME_SLOTS):
        if pi_count[slot].sum() < 1e-12:
            pi_new[slot] = np.full(S, 1.0 / S)
        else:
            pi_new[slot] = pi_count[slot] / pi_count[slot].sum()

    return lam_new, A_new, pi_new


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _init_params(
    counts: NDArray,  # (T, K)
    n_states: int,
    alpha_0: float,
    beta_0: float,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray, NDArray]:
    """Random initialisation of HMM parameters.

    λ is initialised by randomly assigning windows to states and computing
    per-state empirical means (with prior regularisation).  A and π are
    initialised near-uniform with small random perturbations.
    """
    T, K = counts.shape
    S = n_states

    # Randomly assign each window to a state
    assignments = rng.integers(0, S, size=T)
    lam = np.zeros((S, K))
    for s in range(S):
        mask = assignments == s
        if mask.sum() > 0:
            lam[s] = counts[mask].mean(axis=0)
        else:
            lam[s] = counts.mean(axis=0)
        # Apply prior: pull towards alpha_0/beta_0
        lam[s] = (alpha_0 - 1.0 + lam[s] * mask.sum()) / (beta_0 + mask.sum())
    lam = np.maximum(lam, 1e-10)

    # Near-uniform transition matrices with Dirichlet noise
    A = rng.dirichlet(alpha=np.ones(S) * 5.0, size=(_N_TIME_SLOTS, S))  # (168, S, S)

    # Near-uniform initial distributions
    pi = rng.dirichlet(alpha=np.ones(S) * 5.0, size=_N_TIME_SLOTS)   # (168, S)

    return lam, A, pi


# ---------------------------------------------------------------------------
# BIC computation
# ---------------------------------------------------------------------------


def _n_free_params(n_states: int, n_classes: int) -> int:
    """Count the number of free parameters in the model.

    - λ: n_states × n_classes  (all free; constrained to be positive)
    - A: 168 × n_states × (n_states - 1)  (row-stochastic)
    - π: 168 × (n_states - 1)  (simplex constraint)
    """
    S, K = n_states, n_classes
    return S * K + _N_TIME_SLOTS * S * (S - 1) + _N_TIME_SLOTS * (S - 1)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class PhaseHMM:
    """Time-Inhomogeneous HMM for workload phase detection.

    Parameters
    ----------
    config : PhaseHMMConfig
        Model hyperparameters.

    Usage
    -----
    >>> model = PhaseHMM(PhaseHMMConfig(n_states=4))
    >>> result = model.fit(counts, slots)
    >>> states = model.viterbi(counts, slots)
    >>> gamma = model.predict_proba(counts, slots)
    """

    def __init__(self, config: PhaseHMMConfig | None = None) -> None:
        self.config = config or PhaseHMMConfig()
        self.result_: Optional[PhaseHMMResult] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        counts: NDArray[np.int64],  # (T, K)
        slots: NDArray[np.int64],   # (T,)
    ) -> "PhaseHMM":
        """Fit the HMM to count observations via EM with multiple restarts.

        Parameters
        ----------
        counts : array of shape (T, K)
            Per-window, per-class query counts.
        slots : array of shape (T,)
            Integer time-slot index for each window, in [0, 168).
            Compute as: slot = day_of_week * 24 + hour_of_day.

        Returns
        -------
        self
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        T, K = counts.shape
        best_result = PhaseHMMResult()

        for restart in range(cfg.n_init):
            seed = int(rng.integers(0, 2**31))
            inner_rng = np.random.default_rng(seed)
            lam, A, pi = _init_params(counts, cfg.n_states, cfg.alpha_0, cfg.beta_0, inner_rng)

            prev_ll = -np.inf
            for iteration in range(cfg.max_iter):
                # E-step
                gamma, xi, ll = _e_step(counts, slots, lam, A, pi)

                # M-step
                lam, A, pi = _m_step(
                    counts,
                    slots,
                    gamma,
                    xi,
                    cfg.alpha_0,
                    cfg.beta_0,
                    cfg.sticky_transition,
                )

                delta = ll - prev_ll
                logger.debug("Restart %d, iter %d: ll=%.4f (Δ=%.6f)", restart + 1, iteration + 1, ll, delta)

                if abs(delta) < cfg.tol and iteration > 0:
                    logger.info("Restart %d converged at iteration %d (ll=%.4f)", restart + 1, iteration + 1, ll)
                    break
                prev_ll = ll

            n_params = _n_free_params(cfg.n_states, K)
            bic = -2.0 * ll + n_params * np.log(T)

            if ll > best_result.log_likelihood:
                best_result = PhaseHMMResult(
                    lam=lam.copy(),
                    A=A.copy(),
                    pi=pi.copy(),
                    log_likelihood=ll,
                    bic=bic,
                    n_iter=iteration + 1,
                )
                logger.info(
                    "New best: restart %d, ll=%.4f, bic=%.4f", restart + 1, ll, bic
                )

        self.result_ = best_result
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _check_fitted(self) -> PhaseHMMResult:
        if self.result_ is None:
            raise RuntimeError("Model has not been fitted yet.  Call fit() first.")
        return self.result_

    def predict_proba(
        self,
        counts: NDArray[np.int64],
        slots: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Compute the posterior state distribution P(z_t | observations).

        Returns
        -------
        gamma : array of shape (T, n_states)
        """
        r = self._check_fitted()
        gamma, _, _ = _e_step(counts, slots, r.lam, r.A, r.pi)
        return gamma

    def viterbi(
        self,
        counts: NDArray[np.int64],
        slots: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Viterbi decoding: most probable hidden state sequence.

        Returns
        -------
        states : array of shape (T,) with values in [0, n_states)
        """
        r = self._check_fitted()
        lam, A, pi = r.lam, r.A, r.pi
        T, _ = counts.shape
        S = lam.shape[0]

        log_emit = _poisson_log_pmf(counts, lam)            # (T, S)
        A_seq = A[slots[:-1]]                               # (T-1, S, S)
        log_A_seq = np.log(np.maximum(A_seq, 1e-300))

        # DP tables
        delta = np.empty((T, S))
        psi = np.zeros((T, S), dtype=int)

        delta[0] = np.log(np.maximum(pi[slots[0]], 1e-300)) + log_emit[0]

        for t in range(1, T):
            scores = delta[t - 1, :, None] + log_A_seq[t - 1]  # (S, S)
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + log_emit[t]

        # Backtrack
        states = np.empty(T, dtype=int)
        states[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def score(
        self,
        counts: NDArray[np.int64],
        slots: NDArray[np.int64],
    ) -> float:
        """Return the log-likelihood of the given observations."""
        r = self._check_fitted()
        _, _, ll = _e_step(counts, slots, r.lam, r.A, r.pi)
        return ll

    # ------------------------------------------------------------------
    # BIC-based model selection (static utility)
    # ------------------------------------------------------------------

    @staticmethod
    def select_n_states(
        counts: NDArray[np.int64],
        slots: NDArray[np.int64],
        n_states_range: list[int] | None = None,
        config_kwargs: dict | None = None,
    ) -> dict[int, PhaseHMMResult]:
        """Fit one model per candidate n_states and return BIC scores.

        Parameters
        ----------
        counts, slots : training data (see fit()).
        n_states_range : list of candidate state counts.  Defaults to [2..8].
        config_kwargs : extra keyword args passed to PhaseHMMConfig
                        (excluding n_states).

        Returns
        -------
        results : dict mapping n_states → PhaseHMMResult (sorted by BIC)
        """
        if n_states_range is None:
            n_states_range = list(range(2, 9))
        config_kwargs = config_kwargs or {}

        results: dict[int, PhaseHMMResult] = {}
        for n in n_states_range:
            logger.info("Fitting PhaseHMM with n_states=%d …", n)
            cfg = PhaseHMMConfig(n_states=n, **config_kwargs)
            model = PhaseHMM(cfg).fit(counts, slots)
            assert model.result_ is not None
            results[n] = model.result_
            logger.info("  BIC=%.2f, ll=%.4f", model.result_.bic, model.result_.log_likelihood)

        return dict(sorted(results.items(), key=lambda kv: kv[1].bic))
