from __future__ import annotations

"""Out-of-distribution diagnostics for IconQ model investigations.

This script compares:
1) The saved training dataset used by an IconQ model.
2) A dataset reconstructed from a target run trace.

It reports:
- Dataset shape differences (sequence lengths, target distribution, censoring).
- Per-feature range checks and out-of-range fractions.
- Arrival-process distribution checks against training runs, with emphasis on
  interarrival statistics, burst/spike intensity, and effective concurrency.

Outputs are written as CSV files under --output_dir for iterative analysis.

Example usage:
python3 experiments/26_bad_model_perf_investigation/out_of_distribution.py \
  --iconq_model_id all_66_and_99_template_runs_with_aborted_v2 \
  --run_id 1780875805584
"""

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from autoslo.filesystem.path_utils import get_data_dir, get_runs_dir
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.models.iconq_model import IconqModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset

EPS = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Out-of-distribution diagnostics for IconQ: compare training "
            "dataset + training run arrival dynamics against a target run."
        )
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        default="all_66_and_99_template_runs_with_aborted_v2",
        help="IconQ model ID under data/iconq_models/<iconq_model_id>/.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="1780875805584",
        help="Run ID under data/runs/<run_id>/structured_log.parquet.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/26_bad_model_perf_investigation/ood_outputs"),
        help="Directory where CSV summaries will be written.",
    )
    parser.add_argument(
        "--windows_s",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 30.0],
        help="Window sizes in seconds used for spike diagnostics.",
    )
    parser.add_argument(
        "--max_train_runs",
        type=int,
        default=None,
        help=(
            "Optional cap on number of training runs used for arrival baseline "
            "(useful for quick iteration)."
        ),
    )
    return parser.parse_args()


def _load_training_run_ids(model_dir: Path) -> list[str]:
    params_path = model_dir / "params.yml"
    if not params_path.exists():
        raise FileNotFoundError(f"Model params file not found: {params_path}")

    params = yaml.safe_load(params_path.read_text())
    if not isinstance(params, dict):
        raise ValueError(f"Invalid params.yml at {params_path}")

    train_config = params.get("train_config") or {}
    if not isinstance(train_config, dict):
        raise ValueError(f"Invalid train_config in {params_path}")

    run_ids = train_config.get("run_ids") or []
    return [str(rid) for rid in run_ids]


def _stack_dataset_rows(dataset: ConcurrentQueryDataset) -> np.ndarray:
    if len(dataset.x) == 0:
        return np.empty((0, 0), dtype=np.float32)
    rows = [seq.detach().cpu().numpy() for seq in dataset.x]
    return np.concatenate(rows, axis=0)


def _percentile_of_score(sample: np.ndarray, value: float) -> float:
    if sample.size == 0:
        return float("nan")
    return float((sample <= value).sum() / sample.size * 100.0)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if values.size == 0:
        return float("nan")
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return float("nan")
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, quantile, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def _first_event_times(
    df: pd.DataFrame,
    event_type: str,
    value_col: str = "rel_time_s",
) -> pd.Series:
    filtered = df[df["event_type"] == event_type]
    if filtered.empty:
        return pd.Series(dtype=float)
    return filtered.groupby("query_id", observed=True)[value_col].min()


def _concurrency_metrics(
    starts: np.ndarray, ends: np.ndarray
) -> dict[str, float]:
    if starts.size == 0 or ends.size == 0:
        return {
            "max": float("nan"),
            "mean_time_weighted": float("nan"),
            "p95_time_weighted": float("nan"),
            "at_start_median": float("nan"),
            "at_start_p95": float("nan"),
        }

    valid = np.isfinite(starts) & np.isfinite(ends) & (ends >= starts)
    starts = starts[valid]
    ends = ends[valid]
    if starts.size == 0:
        return {
            "max": float("nan"),
            "mean_time_weighted": float("nan"),
            "p95_time_weighted": float("nan"),
            "at_start_median": float("nan"),
            "at_start_p95": float("nan"),
        }

    times = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones_like(starts), -np.ones_like(ends)])
    order = np.lexsort((deltas, times))
    times_sorted = times[order]
    deltas_sorted = deltas[order]

    conc_vals: list[float] = []
    durations: list[float] = []
    current = 0.0
    for i in range(times_sorted.size - 1):
        current += deltas_sorted[i]
        dt = times_sorted[i + 1] - times_sorted[i]
        if dt > 0:
            conc_vals.append(current)
            durations.append(dt)

    if len(durations) == 0:
        mean_tw = float(current)
        p95_tw = float(current)
        max_conc = float(max(current, 0.0))
    else:
        conc_arr = np.asarray(conc_vals, dtype=float)
        dur_arr = np.asarray(durations, dtype=float)
        mean_tw = float(np.average(conc_arr, weights=dur_arr))
        p95_tw = _weighted_quantile(conc_arr, dur_arr, 0.95)
        max_conc = float(np.max(conc_arr))

    starts_sorted = np.sort(starts)
    ends_sorted = np.sort(ends)
    at_start = np.searchsorted(
        starts_sorted, starts, side="right"
    ) - np.searchsorted(ends_sorted, starts, side="right")

    return {
        "max": max_conc,
        "mean_time_weighted": mean_tw,
        "p95_time_weighted": p95_tw,
        "at_start_median": (
            float(np.median(at_start)) if at_start.size else float("nan")
        ),
        "at_start_p95": (
            float(np.percentile(at_start, 95))
            if at_start.size
            else float("nan")
        ),
    }


def _arrival_metrics_for_log(
    structured_log: StructuredLog,
    windows_s: Iterable[float],
) -> dict[str, float]:
    df = structured_log.df.copy()
    if df.empty:
        return {}

    arrivals = _first_event_times(df, "arrival")
    completions = _first_event_times(df, "completion")
    exec_starts = _first_event_times(df, "query_execution_start")
    exec_finishes = _first_event_times(df, "query_execution_finish")

    arrival_values = (
        np.sort(arrivals.to_numpy(dtype=float))
        if not arrivals.empty
        else np.array([])
    )
    interarrivals = (
        np.diff(arrival_values) if arrival_values.size >= 2 else np.array([])
    )

    details_map = (
        df[df["event_type"] == "completion"]["details"]
        if "details" in df.columns
        else pd.Series(dtype=object)
    )
    success_rate = float("nan")
    if len(details_map) > 0:
        success_flags = [
            d.get("success")
            for d in details_map
            if isinstance(d, dict) and "success" in d
        ]
        if len(success_flags) > 0:
            success_rate = float(np.mean([bool(x) for x in success_flags]))

    submitted_df = pd.concat(
        [arrivals.rename("start_s"), completions.rename("end_s")], axis=1
    ).dropna()
    executing_df = pd.concat(
        [exec_starts.rename("start_s"), exec_finishes.rename("end_s")], axis=1
    ).dropna()

    submitted_conc = _concurrency_metrics(
        submitted_df["start_s"].to_numpy(dtype=float),
        submitted_df["end_s"].to_numpy(dtype=float),
    )
    executing_conc = _concurrency_metrics(
        executing_df["start_s"].to_numpy(dtype=float),
        executing_df["end_s"].to_numpy(dtype=float),
    )

    # Metric glossary for printed arrival/concurrency diagnostics:
    #   num_arrivals: Number of submitted queries (ARRIVAL events).
    #   arrival_span_s: Wall-clock span between first and last arrival.
    #   arrival_rate_qps: num_arrivals / arrival_span_s.
    #   interarrival_*: Distribution stats of delta between consecutive arrivals.
    #   interarrival_cv: std(interarrival) / mean(interarrival); burstiness proxy.
    #   completion_success_rate: Fraction of COMPLETION events with success=True.
    #   submitted_effective_concurrency_*: Time-weighted/peak in-flight submitted queries.
    #   submitted_concurrency_at_arrival_*: In-flight submitted concurrency sampled at arrival instants.
    #   executing_concurrency_*: Time-weighted/peak actively executing queries.
    #   spike_peak_in_{W}s: Max arrivals in any W-second bucket.
    #   spike_p95_in_{W}s: 95th percentile of arrivals across W-second buckets.
    #   spike_peak_to_p95_ratio_in_{W}s: Peak-to-typical burst amplification at window W.

    out: dict[str, float] = {
        "num_arrivals": float(arrival_values.size),
        "arrival_span_s": (
            float(arrival_values[-1] - arrival_values[0])
            if arrival_values.size > 1
            else 0.0
        ),
        "arrival_rate_qps": (
            float(
                arrival_values.size
                / max(arrival_values[-1] - arrival_values[0], EPS)
            )
            if arrival_values.size > 1
            else 0.0
        ),
        "interarrival_mean_s": (
            float(np.mean(interarrivals))
            if interarrivals.size
            else float("nan")
        ),
        "interarrival_median_s": (
            float(np.median(interarrivals))
            if interarrivals.size
            else float("nan")
        ),
        "interarrival_p95_s": (
            float(np.percentile(interarrivals, 95))
            if interarrivals.size
            else float("nan")
        ),
        "interarrival_cv": (
            float(np.std(interarrivals) / max(np.mean(interarrivals), EPS))
            if interarrivals.size
            else float("nan")
        ),
        "completion_success_rate": success_rate,
        "submitted_effective_concurrency_mean": submitted_conc[
            "mean_time_weighted"
        ],
        "submitted_effective_concurrency_p95": submitted_conc[
            "p95_time_weighted"
        ],
        "submitted_effective_concurrency_max": submitted_conc["max"],
        "submitted_concurrency_at_arrival_median": submitted_conc[
            "at_start_median"
        ],
        "submitted_concurrency_at_arrival_p95": submitted_conc["at_start_p95"],
        "executing_concurrency_mean": executing_conc["mean_time_weighted"],
        "executing_concurrency_p95": executing_conc["p95_time_weighted"],
        "executing_concurrency_max": executing_conc["max"],
    }

    for window_s in windows_s:
        if arrival_values.size == 0 or window_s <= 0:
            out[f"spike_peak_in_{window_s:g}s"] = float("nan")
            out[f"spike_p95_in_{window_s:g}s"] = float("nan")
            out[f"spike_peak_to_p95_ratio_in_{window_s:g}s"] = float("nan")
            continue
        bins = np.floor(arrival_values / window_s).astype(int)
        counts = pd.Series(bins).value_counts().to_numpy(dtype=float)
        peak = float(np.max(counts))
        p95 = float(np.percentile(counts, 95))
        out[f"spike_peak_in_{window_s:g}s"] = peak
        out[f"spike_p95_in_{window_s:g}s"] = p95
        out[f"spike_peak_to_p95_ratio_in_{window_s:g}s"] = peak / max(p95, EPS)

    return out


def _dataset_global_summary(
    dataset: ConcurrentQueryDataset,
    label: str,
) -> pd.DataFrame:
    all_rows = _stack_dataset_rows(dataset)
    if all_rows.size == 0:
        return pd.DataFrame()

    feature_min = np.min(all_rows, axis=0)
    feature_max = np.max(all_rows, axis=0)
    feature_mean = np.mean(all_rows, axis=0)
    feature_std = np.std(all_rows, axis=0)
    feature_p01 = np.percentile(all_rows, 1, axis=0)
    feature_p99 = np.percentile(all_rows, 99, axis=0)

    return pd.DataFrame(
        {
            "dataset": label,
            "feature_idx": np.arange(all_rows.shape[1]),
            "min": feature_min,
            "p01": feature_p01,
            "mean": feature_mean,
            "std": feature_std,
            "p99": feature_p99,
            "max": feature_max,
        }
    )


def _dataset_shape_summary(
    dataset: ConcurrentQueryDataset, label: str
) -> pd.Series:
    # Metric glossary for the printed dataset-shape table:
    #   num_samples: Number of base-query samples in the dataset.
    #   num_rows_total: Total number of interaction rows across all sequences.
    #   seq_len_*: Distribution of interaction sequence lengths per sample.
    #   pinch_point_*: Distribution of self-row indices in each sequence.
    #   y_*: Distribution of the model target values stored in dataset.y.
    #   lower_bound_fraction: Share of censored/lower-bound targets.
    seq_lens = np.asarray([len(x) for x in dataset.x], dtype=float)
    pinch = dataset.pinch_points.detach().cpu().numpy().astype(float)
    y = dataset.y.detach().cpu().numpy().astype(float)
    y_lb = dataset.y_is_lower_bound.detach().cpu().numpy().astype(bool)

    return pd.Series(
        {
            "dataset": label,
            "num_samples": len(dataset),
            "num_rows_total": int(np.sum(seq_lens)) if seq_lens.size else 0,
            "seq_len_mean": (
                float(np.mean(seq_lens)) if seq_lens.size else float("nan")
            ),
            "seq_len_p95": (
                float(np.percentile(seq_lens, 95))
                if seq_lens.size
                else float("nan")
            ),
            "seq_len_max": (
                float(np.max(seq_lens)) if seq_lens.size else float("nan")
            ),
            "pinch_point_mean": (
                float(np.mean(pinch)) if pinch.size else float("nan")
            ),
            "pinch_point_p95": (
                float(np.percentile(pinch, 95)) if pinch.size else float("nan")
            ),
            "y_mean": float(np.mean(y)) if y.size else float("nan"),
            "y_p95": float(np.percentile(y, 95)) if y.size else float("nan"),
            "lower_bound_fraction": (
                float(np.mean(y_lb)) if y_lb.size else float("nan")
            ),
        }
    )


def _feature_range_ood_table(
    train_dataset: ConcurrentQueryDataset,
    run_dataset: ConcurrentQueryDataset,
) -> pd.DataFrame:
    train_rows = _stack_dataset_rows(train_dataset)
    run_rows = _stack_dataset_rows(run_dataset)
    if train_rows.size == 0 or run_rows.size == 0:
        return pd.DataFrame()
    if train_rows.shape[1] != run_rows.shape[1]:
        raise ValueError(
            "Training and run feature dimensions do not match: "
            f"{train_rows.shape[1]} vs {run_rows.shape[1]}"
        )

    train_min = train_rows.min(axis=0)
    train_max = train_rows.max(axis=0)
    run_min = run_rows.min(axis=0)
    run_max = run_rows.max(axis=0)

    frac_below = (run_rows < train_min[None, :]).mean(axis=0)
    frac_above = (run_rows > train_max[None, :]).mean(axis=0)

    return pd.DataFrame(
        {
            "feature_idx": np.arange(train_rows.shape[1]),
            "train_min": train_min,
            "train_max": train_max,
            "run_min": run_min,
            "run_max": run_max,
            "frac_run_rows_below_train_min": frac_below,
            "frac_run_rows_above_train_max": frac_above,
            "frac_run_rows_outside_train_range": frac_below + frac_above,
        }
    )


def _summarize_target_vs_train(
    target: dict[str, float],
    baseline_per_run_df: pd.DataFrame,
) -> pd.DataFrame:
    def _metric_explanation(metric_name: str) -> str:
        if metric_name == "num_arrivals":
            return "Count of ARRIVAL events (submitted queries)."
        if metric_name == "arrival_span_s":
            return "Seconds between first and last arrival."
        if metric_name == "arrival_rate_qps":
            return (
                "Average submitted query rate: num_arrivals / arrival_span_s."
            )
        if metric_name == "interarrival_mean_s":
            return "Mean time between consecutive arrivals (s)."
        if metric_name == "interarrival_median_s":
            return "Median time between consecutive arrivals (s)."
        if metric_name == "interarrival_p95_s":
            return "95th percentile time between consecutive arrivals (s)."
        if metric_name == "interarrival_cv":
            return "Burstiness proxy: std(interarrival) / mean(interarrival)."
        if metric_name == "completion_success_rate":
            return "Fraction of completion events with success=True."
        if metric_name == "submitted_effective_concurrency_mean":
            return "Time-weighted mean in-flight submitted query count."
        if metric_name == "submitted_effective_concurrency_p95":
            return "Time-weighted 95th percentile of in-flight submitted query count."
        if metric_name == "submitted_effective_concurrency_max":
            return "Maximum in-flight submitted query count."
        if metric_name == "submitted_concurrency_at_arrival_median":
            return "Median in-flight submitted concurrency sampled at arrival instants."
        if metric_name == "submitted_concurrency_at_arrival_p95":
            return "95th percentile in-flight submitted concurrency sampled at arrival instants."
        if metric_name == "executing_concurrency_mean":
            return "Time-weighted mean actively executing query count."
        if metric_name == "executing_concurrency_p95":
            return (
                "Time-weighted 95th percentile actively executing query count."
            )
        if metric_name == "executing_concurrency_max":
            return "Maximum actively executing query count."

        spike_peak_match = re.fullmatch(r"spike_peak_in_(.+)s", metric_name)
        if spike_peak_match:
            window_s = spike_peak_match.group(1)
            return f"Peak arrivals seen in any {window_s}s bucket."

        spike_p95_match = re.fullmatch(r"spike_p95_in_(.+)s", metric_name)
        if spike_p95_match:
            window_s = spike_p95_match.group(1)
            return f"95th percentile arrivals across {window_s}s buckets."

        spike_ratio_match = re.fullmatch(
            r"spike_peak_to_p95_ratio_in_(.+)s", metric_name
        )
        if spike_ratio_match:
            window_s = spike_ratio_match.group(1)
            return (
                "Peak-to-typical burst amplification at "
                f"{window_s}s window (peak / p95)."
            )

        return "No description available."

    records: list[dict[str, float | str]] = []
    if baseline_per_run_df.empty:
        return pd.DataFrame(records)

    for metric, target_value in target.items():
        if metric not in baseline_per_run_df.columns:
            continue
        if not np.isfinite(target_value):
            continue

        baseline = (
            pd.to_numeric(baseline_per_run_df[metric], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if baseline.size == 0:
            continue
        mu = float(np.mean(baseline))
        sigma = float(np.std(baseline))
        z = (target_value - mu) / sigma if sigma > 0 else float("nan")
        pctl = _percentile_of_score(baseline, target_value)
        records.append(
            {
                "metric": metric,
                "metric_explanation": _metric_explanation(metric),
                "target_value": target_value,
                "train_runs_mean": mu,
                "train_runs_std": sigma,
                "target_zscore": z,
                "target_percentile_among_train_runs": pctl,
            }
        )

    if len(records) == 0:
        return pd.DataFrame(records)

    return pd.DataFrame(records).sort_values(
        by="target_percentile_among_train_runs", ascending=False
    )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_id = args.iconq_model_id
    run_id = str(args.run_id)

    model_parent_dir = get_data_dir() / "iconq_models"
    model_dir = (model_parent_dir / model_id).resolve()
    dataset_path = (model_dir / "dataset.pkl").resolve()
    run_structured_log_path = (
        get_runs_dir() / run_id / "structured_log.parquet"
    ).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"IconQ model directory not found: {model_dir}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not run_structured_log_path.exists():
        raise FileNotFoundError(
            f"Run structured log not found: {run_structured_log_path}"
        )

    print("=" * 100)
    print("Loading training artifacts")
    print("=" * 100)
    print(f"Model ID: {model_id}")
    print(f"Model parent dir: {model_parent_dir}")
    print(f"Dataset path: {dataset_path}")

    model = IconqModel.load(model_id=model_id, parent_load_dir=model_parent_dir)
    train_dataset = ConcurrentQueryDataset.load_from(dataset_path)
    print(f"Target run_id: {run_id}")

    print("\nBuilding IconQ-formatted dataset from target run trace...")
    run_dataset = model.build_dataset_from_run_id(run_id=run_id)

    train_shape = _dataset_shape_summary(train_dataset, "train")
    run_shape = _dataset_shape_summary(run_dataset, f"run_{run_id}")
    dataset_shape_df = pd.DataFrame([train_shape, run_shape])

    train_feature_summary_df = _dataset_global_summary(train_dataset, "train")
    run_feature_summary_df = _dataset_global_summary(
        run_dataset, f"run_{run_id}"
    )
    feature_summary_df = pd.concat(
        [train_feature_summary_df, run_feature_summary_df], ignore_index=True
    )
    feature_ood_df = _feature_range_ood_table(train_dataset, run_dataset)

    print("\n" + "=" * 100)
    print("Arrival-process diagnostics (target run vs training runs)")
    print("=" * 100)

    training_run_ids = _load_training_run_ids(model_dir)
    if args.max_train_runs is not None:
        training_run_ids = training_run_ids[: max(args.max_train_runs, 0)]
    print(f"Training run_ids in params.yml: {len(training_run_ids)}")

    train_run_metrics: list[dict[str, Any]] = []
    skipped_run_ids: list[str] = []
    for rid in training_run_ids:
        try:
            metrics: dict[str, Any] = _arrival_metrics_for_log(
                StructuredLog.load(rid), windows_s=args.windows_s
            )
            metrics["run_id"] = rid
            train_run_metrics.append(metrics)
        except Exception:
            skipped_run_ids.append(rid)

    if skipped_run_ids:
        print(
            "Skipped training runs with missing/invalid structured logs: "
            f"{len(skipped_run_ids)}"
        )

    train_run_metrics_df = pd.DataFrame(train_run_metrics)
    target_metrics = _arrival_metrics_for_log(
        StructuredLog.load(run_structured_log_path), windows_s=args.windows_s
    )
    target_metrics["run_id"] = run_id
    target_metrics_df = pd.DataFrame([target_metrics])

    target_vs_train_df = _summarize_target_vs_train(
        target={
            k: float(v)
            for k, v in target_metrics.items()
            if k != "run_id" and isinstance(v, (int, float, np.floating))
        },
        baseline_per_run_df=train_run_metrics_df,
    )

    dataset_shape_df.to_csv(
        args.output_dir / "dataset_shape_summary.csv", index=False
    )
    feature_summary_df.to_csv(
        args.output_dir / "feature_summary_train_vs_target.csv", index=False
    )
    feature_ood_df.to_csv(
        args.output_dir / "feature_range_ood.csv", index=False
    )
    train_run_metrics_df.to_csv(
        args.output_dir / "train_runs_arrival_metrics.csv", index=False
    )
    target_metrics_df.to_csv(
        args.output_dir / f"target_run_{run_id}_arrival_metrics.csv",
        index=False,
    )
    target_vs_train_df.to_csv(
        args.output_dir
        / f"target_run_{run_id}_vs_train_arrival_distribution.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("Quick textual summary")
    print("=" * 100)
    print(dataset_shape_df.to_string(index=False))

    if not feature_ood_df.empty:
        top_ood = feature_ood_df.sort_values(
            by="frac_run_rows_outside_train_range", ascending=False
        ).head(12)
        print("\nTop feature dimensions by out-of-range fraction:")
        print(
            top_ood[
                [
                    "feature_idx",
                    "frac_run_rows_outside_train_range",
                    "run_min",
                    "run_max",
                    "train_min",
                    "train_max",
                ]
            ].to_string(index=False)
        )

    if not target_vs_train_df.empty:
        high_tail = target_vs_train_df[
            target_vs_train_df["target_percentile_among_train_runs"] >= 95.0
        ]
        low_tail = target_vs_train_df[
            target_vs_train_df["target_percentile_among_train_runs"] <= 5.0
        ]
        print(
            "\nArrival/concurrency metrics in extreme tails vs training runs "
            f"(>=95th or <=5th percentile): {len(high_tail) + len(low_tail)}"
        )
        if len(high_tail) > 0:
            print("\nHigh-tail metrics (target unusually high):")
            print(
                high_tail[
                    [
                        "metric",
                        "metric_explanation",
                        "target_value",
                        "train_runs_mean",
                        "target_percentile_among_train_runs",
                        "target_zscore",
                    ]
                ].to_string(index=False)
            )
        if len(low_tail) > 0:
            print("\nLow-tail metrics (target unusually low):")
            print(
                low_tail[
                    [
                        "metric",
                        "metric_explanation",
                        "target_value",
                        "train_runs_mean",
                        "target_percentile_among_train_runs",
                        "target_zscore",
                    ]
                ].to_string(index=False)
            )

    print("\nWrote outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
