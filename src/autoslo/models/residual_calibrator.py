"""Post-hoc residual calibrator for IconqModel predictions.

Fits per-(cluster_rpu, concurrency_bin) quantiles of (predicted / actual)
on the held-out validation split and applies a corrective division at
prediction time via ModelPrediction.overall_mean_s(percentile=...).
"""

from __future__ import annotations

import logging
import pickle
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concurrency bins — shared with iconq_model_performance.py visualizations.
# ---------------------------------------------------------------------------

CONCURRENCY_BINS: list[float] = [-0.5, 0.5, 25.5, 75.5, 150.5, 250.5, float("inf")]
CONCURRENCY_LABELS: list[str] = ["0", "1-25", "26-75", "76-150", "151-250", "251+"]


def _concurrency_bin(concurrency: int) -> str:
    """Map an integer concurrency count to its CONCURRENCY_LABELS bin label.

    Uses the module-level CONCURRENCY_BINS/CONCURRENCY_LABELS defaults.
    Prefer ResidualCalibrator._concurrency_bin() when a custom config is active.
    """
    idx = bisect_right(CONCURRENCY_BINS[1:-1], concurrency)
    return CONCURRENCY_LABELS[idx]


@dataclass
class ResidualCalibratorConfig:
    shrinkage_k: int = 100
    min_bucket_count: int = 20
    concurrency_bins: list[float] = field(
        default_factory=lambda: list(CONCURRENCY_BINS)
    )
    concurrency_labels: list[str] = field(
        default_factory=lambda: list(CONCURRENCY_LABELS)
    )


class ResidualCalibrator:
    """Post-hoc residual calibrator conditioned on (cluster_rpu, concurrency_bin).

    After fitting on a held-out validation DataFrame, exposes ``correct_scalar``
    which applies a multiplicative correction to a raw scalar latency.

    Shrinkage hierarchy:
        level 1: (rpu, concurrency_bin) fine bucket
        level 2: rpu-marginal
        level 3: global

        ratio_final = w1 * ratio_bucket + (1 - w1) * (w2 * ratio_rpu + (1 - w2) * ratio_global)
        w1 = n_bucket / (n_bucket + k)
        w2 = n_rpu    / (n_rpu    + k)
    """

    def __init__(self, config: Optional[ResidualCalibratorConfig] = None) -> None:
        self._config = config or ResidualCalibratorConfig()
        if len(self._config.concurrency_bins) != len(self._config.concurrency_labels) + 1:
            raise ValueError(
                f"concurrency_bins must have exactly len(concurrency_labels)+1 edges; "
                f"got {len(self._config.concurrency_bins)} bins and "
                f"{len(self._config.concurrency_labels)} labels."
            )
        # rpu (int) -> conc_bin (str) -> np.ndarray of residual_ratio (predicted/actual) values
        self._bucket_residuals: dict[int, dict[str, np.ndarray]] = {}
        # rpu (int) -> np.ndarray (all concurrency bins combined for that rpu)
        self._rpu_residuals: dict[int, np.ndarray] = {}
        # global: all rpus, all concurrency bins
        self._global_residuals: Optional[np.ndarray] = None

    def _concurrency_bin(self, concurrency: int) -> str:
        """Map a concurrency count to its label using the instance's config bins."""
        bins = self._config.concurrency_bins
        labels = self._config.concurrency_labels
        idx = bisect_right(bins[1:-1], concurrency)
        return labels[idx]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, val_df: pd.DataFrame) -> None:
        """Fit residual calibration quantiles from the validation DataFrame.

        Parameters:
            val_df: DataFrame produced by IconqModel.eval_on_split(DataSplit.VAL).
                Required columns: cluster_rpu, num_other_concurrent_queries,
                y, y_pred_mean, model_source, target_is_lower_bound.

        # TODO(future): using val_split_type='random' means these residuals are
        # estimated on an in-distribution (temporally interleaved) val set.
        # A temporally held-out split would give a better estimate of how
        # calibration generalises to future workloads, but requires re-training.
        """
        df = val_df.copy()

        # Restrict to LSTM-sourced rows (skip StageModel predictions).
        if "model_source" in df.columns:
            df = df[df["model_source"] == "lstm"]

        # Exclude aborted / lower-bound targets (censored observations).
        if "target_is_lower_bound" in df.columns:
            df = df[~df["target_is_lower_bound"].astype(bool)]

        # Drop non-positive values (ratio would be undefined or negative).
        df = df[(df["y"] > 0) & (df["y_pred_mean"] > 0)]

        if df.empty:
            logger.warning(
                "ResidualCalibrator.fit: no usable rows after filtering; "
                "calibration will have no effect."
            )
            self._global_residuals = np.array([1.0])
            return

        df = df.copy()
        df["residual_ratio"] = (
            df["y_pred_mean"].astype(float) / df["y"].astype(float)
        )
        df["conc_bin"] = pd.cut(
            df["num_other_concurrent_queries"],
            bins=self._config.concurrency_bins,
            labels=self._config.concurrency_labels,
        ).astype(str)

        self._global_residuals = df["residual_ratio"].to_numpy(dtype=float, copy=True)

        for rpu_val, rpu_group in df.groupby("rpu"):
            rpu = int(rpu_val)
            self._rpu_residuals[rpu] = rpu_group["residual_ratio"].to_numpy(
                dtype=float, copy=True
            )
            self._bucket_residuals[rpu] = {}
            for conc_bin_val, cell_group in rpu_group.groupby("conc_bin", observed=True):
                conc_bin = str(conc_bin_val)
                self._bucket_residuals[rpu][conc_bin] = (
                    cell_group["residual_ratio"].to_numpy(dtype=float, copy=True)
                )

    # ------------------------------------------------------------------
    # Lookup and correction
    # ------------------------------------------------------------------

    def lookup(self, rpu: int, concurrency: int, percentile: float) -> float:
        """Return the correction ratio such that calibrated_s = raw_s / ratio.

        Applies hierarchical shrinkage toward coarser levels when the fine
        bucket is sparse.  ratio < 1 means the model underestimates (dangerous);
        ratio > 1 means the model overestimates (safe).
        """
        if self._global_residuals is None:
            raise RuntimeError("ResidualCalibrator has not been fit yet.")

        # Clip percentile to [0, 1].
        p = max(0.0, min(1.0, percentile))

        conc_bin = self._concurrency_bin(concurrency)
        k = self._config.shrinkage_k
        min_n = self._config.min_bucket_count

        # Global ratio (deepest fallback).
        ratio_global = float(np.quantile(self._global_residuals, p))

        # RPU-marginal ratio and shrinkage weight.
        # Buckets below min_bucket_count are treated as empty (hard fallback).
        rpu_arr = self._rpu_residuals.get(rpu)
        if rpu_arr is not None and len(rpu_arr) >= min_n:
            n_rpu = len(rpu_arr)
            ratio_rpu = float(np.quantile(rpu_arr, p))
            w2 = n_rpu / (n_rpu + k)
        else:
            ratio_rpu = ratio_global
            w2 = 0.0

        fallback = w2 * ratio_rpu + (1.0 - w2) * ratio_global

        # Fine-bucket ratio and shrinkage weight.
        # Buckets below min_bucket_count are treated as empty (hard fallback).
        bucket_arr = self._bucket_residuals.get(rpu, {}).get(conc_bin)
        if bucket_arr is not None and len(bucket_arr) >= min_n:
            n_bucket = len(bucket_arr)
            ratio_bucket = float(np.quantile(bucket_arr, p))
            w1 = n_bucket / (n_bucket + k)
        else:
            ratio_bucket = fallback
            w1 = 0.0

        return w1 * ratio_bucket + (1.0 - w1) * fallback

    def correct_scalar(
        self,
        raw_mean_s: float,
        rpu: int,
        concurrency: int,
        percentile: float,
    ) -> float:
        """Apply the residual correction to a raw scalar latency.

        Called from ModelPrediction.overall_mean_s; all guards
        (point-estimate check, model_source check) are handled there.
        """
        ratio = self.lookup(rpu, concurrency, percentile)
        return raw_mean_s / ratio

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ResidualCalibrator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ResidualCalibrator):
            raise ValueError(
                f"Expected a ResidualCalibrator, got {type(obj).__name__}"
            )
        return obj
