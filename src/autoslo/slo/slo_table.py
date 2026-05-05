"""
SLO-table generation helpers.

Computes per-``(template, query_index)`` SLOs from isolated (non-overlapping)
latency measurements, and persists them as YAML dictionaries compatible with
:class:`autoslo.blueprint_selection.slo_resolver.SloResolver`.

This module can also be invoked directly as a script to regenerate all SLO
tables from scratch::

    python -m autoslo.slo.slo_table [options]

Run with ``--help`` for the full list of options.
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import autoslo.filesystem.path_utils as pu
from autoslo.filesystem.yaml_helpers import dump_yaml
from autoslo.workload_execution.trace import Trace

from autoslo.workload_definition.query import QueryTextId

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
        ``run_id``, ``rpu``, ``query_text_id``, ``latency_s``.
    """
    rows: list[dict] = []
    for run_id in run_ids:
        trace = Trace(run_id)
        rpu_dict = trace.rpu_per_cluster()
        # Single-cluster blueprints — always one entry.
        rpu = list(rpu_dict.values())[0]

        latencies = trace.latencies_s
        query_text_ids = trace.query_text_ids
        non_overlapping = trace.query_is_non_overlapping()
        aborted = trace.was_aborted()
        cached = trace.was_cached()

        mask = non_overlapping & ~aborted & ~cached
        for qid in mask.index[mask]:
            rows.append(
                {
                    "run_id": run_id,
                    "rpu": rpu,
                    "query_text_id": query_text_ids[qid],
                    "latency_s": latencies[qid],
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# SLO computation & dump
# ------------------------------------------------------------------


def _extract_template_id(query_text_id: QueryTextId) -> str:
    """Extract template ID from a ``'042_001'`` style key."""
    return query_text_id.split("_")[0]


def compute_slo_dict(
    latencies_df: pd.DataFrame,
    rpu: int,
    baseline_percentile: float = 50.0,
    multiplier: float = 2.5,
) -> dict[str, float]:
    """
    Compute a per-template SLO dictionary.

    All variants (query indices) of the same TPC-DS template are pooled
    into a single sample.  The SLO is then:

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
        A dict mapping ``template_id`` (zero-padded str, e.g. ``"042"``) →
        SLO in seconds (rounded to 3 decimal places).
    """
    subset = latencies_df.loc[latencies_df["rpu"] == rpu].copy()
    if subset.empty:
        raise ValueError(
            f"No isolated latencies found for rpu={rpu}. "
            f"Available RPUs: {sorted(latencies_df['rpu'].unique())}."
        )

    subset["template_id"] = subset["query_text_id"].apply(
        lambda qid: QueryTextId(qid).template_id
    )

    baseline = subset.groupby("template_id")["latency_s"].quantile(
        baseline_percentile / 100.0
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

    dump_yaml(meta, output_path)

    return slo_dict


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

_DEFAULT_RPUS = [4, 8, 16, 32]
_DEFAULT_WORKLOAD_NAME = "benchmarking_workload_99_3_3_shuffled_42"
_DEFAULT_BLUEPRINT_PATTERN = "single_{rpu}"
_DEFAULT_REFERENCE_RPU = 16
_DEFAULT_PERCENTILES = [50.0, 90.0, 95.0]
_DEFAULT_MULTIPLIERS = [1.5, 2.5, 4.0]
_DEFAULT_WORKLOAD_TAG = "ext_tpcds1000"
_DEFAULT_RUNS_PER_RPU = 1


def generate_slo_tables(
    rpus: list[int] = _DEFAULT_RPUS,
    workload_name: str = _DEFAULT_WORKLOAD_NAME,
    blueprint_pattern: str = _DEFAULT_BLUEPRINT_PATTERN,
    reference_rpu: int = _DEFAULT_REFERENCE_RPU,
    baseline_percentiles: list[float] = _DEFAULT_PERCENTILES,
    multipliers: list[float] = _DEFAULT_MULTIPLIERS,
    workload_tag: str = _DEFAULT_WORKLOAD_TAG,
    runs_per_rpu: int = _DEFAULT_RUNS_PER_RPU,
    latencies_path: Optional[str] = None,
) -> None:
    """
    End-to-end generation of all SLO tables.

    1. Collects isolated-run IDs for each RPU level via
       :class:`~autoslo.filesystem.path_utils.RunLocator`.
    2. Extracts isolated (non-overlapping) latencies and saves them to
       ``data/slos/isolated_latencies.parquet``.
    3. Sweeps every ``(baseline_percentile, multiplier)`` combination and
       writes one YAML SLO table per combination via
       :func:`dump_slo_table`.

    Parameters:
        rpus: RPU levels to collect isolated runs for.
        workload_name: ``workload_name`` filter passed to
            :meth:`~autoslo.filesystem.path_utils.RunLocator.get_run_ids`.
        blueprint_pattern: Python format string for the ``blueprint_name``
            filter.  ``{rpu}`` is replaced with the current RPU value, e.g.
            ``"single_{rpu}"`` → ``"single_16"``.
        reference_rpu: The RPU level used as the SLO baseline.  Must be
            present in *rpus*.
        baseline_percentiles: Percentiles (0–100) to sweep.
        multipliers: Headroom multipliers (*k*) to sweep.
        workload_tag: Short label embedded in output filenames.
        runs_per_rpu: How many of the most-recent runs to use per RPU level.
        latencies_path: Optional explicit path for the
            ``isolated_latencies.parquet`` file.  Defaults to
            ``data/slos/isolated_latencies.parquet``.
    """
    # Import here to avoid a circular dependency at module load time; the
    # path_utils module is heavy and not needed when slo_table is imported
    # purely for its computation helpers.
    from autoslo.filesystem.path_utils import RunLocator

    # ── 1. Collect run IDs ─────────────────────────────────────────
    all_run_ids: list[str] = []
    for rpu in rpus:
        run_ids = RunLocator.get_run_ids(
            workload_name=workload_name,
            blueprint_name=blueprint_pattern.format(rpu=rpu),
        )[-runs_per_rpu:]
        if not run_ids:
            raise ValueError(
                f"No runs found for rpu={rpu} "
                f"(workload_name={workload_name!r}, "
                f"blueprint_name={blueprint_pattern.format(rpu=rpu)!r})."
            )
        all_run_ids.extend(run_ids)
    print(
        f"Collected {len(all_run_ids)} run(s) across {len(rpus)} RPU level(s)."
    )

    # ── 2. Extract & persist isolated latencies ────────────────────
    latencies_df = compute_isolated_latencies(all_run_ids)
    print(
        f"Extracted {len(latencies_df)} isolated measurements "
        f"across {latencies_df['rpu'].nunique()} RPU level(s)."
    )

    if latencies_path is None:
        slo_dir = os.path.join(pu.get_data_path(), _DEFAULT_DATA_SUBDIR)
        os.makedirs(slo_dir, exist_ok=True)
        latencies_path = os.path.join(slo_dir, "isolated_latencies.parquet")

    latencies_df.to_parquet(latencies_path, index=False)
    print(f"Saved isolated latencies → {latencies_path}")

    # ── 3. Sweep parameter combinations ───────────────────────────
    for p in baseline_percentiles:
        for k in multipliers:
            output_path = dump_slo_table(
                latencies_df,
                rpu=reference_rpu,
                baseline_percentile=p,
                multiplier=k,
                workload_tag=workload_tag,
            )
            print(f"  Wrote SLO table  p={p}  k={k}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SLO tables from isolated benchmark runs. "
            "Writes isolated_latencies.parquet and one YAML SLO table per "
            "(percentile, multiplier) combination to data/slos/."
        )
    )
    parser.add_argument(
        "--rpus",
        nargs="+",
        type=int,
        default=_DEFAULT_RPUS,
        metavar="RPU",
        help=f"RPU levels to include (default: {_DEFAULT_RPUS}).",
    )
    parser.add_argument(
        "--workload-name",
        default=_DEFAULT_WORKLOAD_NAME,
        help="workload_name filter for RunLocator (default: %(default)s).",
    )
    parser.add_argument(
        "--blueprint-pattern",
        default=_DEFAULT_BLUEPRINT_PATTERN,
        help=(
            "blueprint_name filter pattern; {rpu} is substituted "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--reference-rpu",
        type=int,
        default=_DEFAULT_REFERENCE_RPU,
        help=f"RPU level used as SLO baseline (default: {_DEFAULT_REFERENCE_RPU}).",
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=_DEFAULT_PERCENTILES,
        metavar="P",
        help=f"Baseline percentiles to sweep (default: {_DEFAULT_PERCENTILES}).",
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=_DEFAULT_MULTIPLIERS,
        metavar="K",
        help=f"Headroom multipliers to sweep (default: {_DEFAULT_MULTIPLIERS}).",
    )
    parser.add_argument(
        "--workload-tag",
        default=_DEFAULT_WORKLOAD_TAG,
        help="Short label embedded in output filenames (default: %(default)s).",
    )
    parser.add_argument(
        "--runs-per-rpu",
        type=int,
        default=_DEFAULT_RUNS_PER_RPU,
        help=(
            "Number of most-recent runs to use per RPU level "
            f"(default: {_DEFAULT_RUNS_PER_RPU})."
        ),
    )
    parser.add_argument(
        "--latencies-path",
        default=None,
        help=(
            "Explicit output path for isolated_latencies.parquet. "
            "Defaults to data/slos/isolated_latencies.parquet."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_slo_tables(
        rpus=args.rpus,
        workload_name=args.workload_name,
        blueprint_pattern=args.blueprint_pattern,
        reference_rpu=args.reference_rpu,
        baseline_percentiles=args.percentiles,
        multipliers=args.multipliers,
        workload_tag=args.workload_tag,
        runs_per_rpu=args.runs_per_rpu,
        latencies_path=args.latencies_path,
    )
