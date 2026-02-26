"""
SLO-table generation helpers.

Computes per-``(template, query_index)`` SLOs from isolated (non-overlapping)
latency measurements, and persists them as YAML dictionaries compatible with
:class:`autoslo.blueprint_selection.slo_resolver.SloResolver`.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import autoslo.utils.paths as pu
from autoslo.workload_execution.trace import Trace

_DEFAULT_DATA_SUBDIR = "slos"


# ------------------------------------------------------------------
# Isolated-latency extraction
# ------------------------------------------------------------------


def compute_isolated_latencies(
    run_ids: list[str],
) -> pd.DataFrame:
    """
    Extract isolated (non-overlapping) latencies from a list of benchmark
    runs.

    For each run the method keeps only queries that:
      - were not aborted,
      - were not cached,
      - had no temporal overlap with any other query in the same run.

    Parameters:
        run_ids: Benchmark run identifiers.

    Returns:
        A DataFrame with columns:
        ``run_id``, ``rpu``, ``tpcds_temp_and_q_idx``, ``latency_s``.
    """
    rows: list[dict] = []
    for run_id in run_ids:
        trace = Trace(run_id)
        rpu_dict = trace.rpu_per_cluster()
        # Single-cluster blueprints — always one entry.
        rpu = list(rpu_dict.values())[0]

        latencies = trace.latencies_s
        temp_q_idxs = trace.tpcds_temp_and_q_idxs
        non_overlapping = trace.query_is_non_overlapping()
        aborted = trace.was_aborted()
        cached = trace.was_cached()

        mask = non_overlapping & ~aborted & ~cached
        for qid in mask.index[mask]:
            rows.append(
                {
                    "run_id": run_id,
                    "rpu": rpu,
                    "tpcds_temp_and_q_idx": temp_q_idxs[qid],
                    "latency_s": latencies[qid],
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# SLO computation & dump
# ------------------------------------------------------------------


def compute_slo_dict(
    latencies_df: pd.DataFrame,
    rpu: int,
    baseline_percentile: float = 50.0,
    multiplier: float = 2.5,
) -> dict[str, float]:
    """
    Compute a per-``(template, query_index)`` SLO dictionary.

    For each ``tpcds_temp_and_q_idx`` the SLO is:

    .. math::
        \\text{SLO} = k \\times \\text{percentile}(\\text{latency}, p)

    where *k* is ``multiplier`` and *p* is ``baseline_percentile``.

    Parameters:
        latencies_df: Output of :func:`compute_isolated_latencies`.
        rpu: RPU level at which to compute the baseline.
        baseline_percentile: Percentile of the isolated-latency distribution
            (0–100) used as the baseline (e.g. ``50.0`` for the median).
        multiplier: Multiplicative headroom factor *k*.

    Returns:
        A dict mapping ``tpcds_temp_and_q_idx`` → SLO in seconds (rounded to
        3 decimal places).
    """
    subset = latencies_df.loc[latencies_df["rpu"] == rpu]
    if subset.empty:
        raise ValueError(
            f"No isolated latencies found for rpu={rpu}. "
            f"Available RPUs: {sorted(latencies_df['rpu'].unique())}."
        )

    baseline = (
        subset.groupby("tpcds_temp_and_q_idx")["latency_s"]
        .quantile(baseline_percentile / 100.0)
    )
    slo = baseline * multiplier
    return {str(k): round(float(v), 3) for k, v in slo.items()}


def dump_slo_table(
    latencies_df: pd.DataFrame,
    rpu: int,
    baseline_percentile: float,
    multiplier: float,
    output_path: Optional[str] = None,
    workload_tag: str = "default",
) -> dict[str, float]:
    """
    Compute and persist an SLO table as a YAML file.

    If ``output_path`` is not given, the file is written to
    ``data/slos/<workload_tag>_rpu<rpu>_p<percentile>_k<multiplier>.yml``.

    Parameters:
        latencies_df: Output of :func:`compute_isolated_latencies`.
        rpu: RPU level for the baseline.
        baseline_percentile: Percentile of the isolated-latency distribution.
        multiplier: Multiplicative headroom factor *k*.
        output_path: Optional explicit output path.
        workload_tag: Short label used in the default filename.

    Returns:
        The computed SLO dictionary (same as :func:`compute_slo_dict`).
    """
    slo_dict = compute_slo_dict(
        latencies_df, rpu, baseline_percentile, multiplier
    )

    if output_path is None:
        slo_dir = os.path.join(pu.get_data_path(), _DEFAULT_DATA_SUBDIR)
        os.makedirs(slo_dir, exist_ok=True)
        fname = (
            f"{workload_tag}_rpu{rpu}"
            f"_p{int(baseline_percentile)}"
            f"_k{multiplier}.yml"
        )
        output_path = os.path.join(slo_dir, fname)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    meta = {
        "rpu": rpu,
        "baseline_percentile": baseline_percentile,
        "multiplier": multiplier,
        "num_templates": len(slo_dict),
        "slo_dict": slo_dict,
    }
    with open(output_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    return slo_dict
