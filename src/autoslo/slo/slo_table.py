"""
SLO-table generation from a single run's structured log.

Writes one YAML SLO table per (percentile, multiplier) combination to
``data/slos/``, compatible with
:class:`autoslo.blueprint_selection.slo_resolver.SloResolver`.

Run with ``--help`` for the full list of options.
"""

import argparse
import os
from pathlib import Path

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import WorkloadRunnerConfig
from autoslo.filesystem.logging import query_latencies_from_log
from autoslo.filesystem.yaml_helpers import dump_yaml, load_yaml
from autoslo.workload_definition.query import QueryTextId

_DEFAULT_PERCENTILES = [50, 100]
_DEFAULT_MULTIPLIERS = [1, 2, 3, 4, 5, 8, 10]


def generate_slo_tables(
    run_id: str,
    baseline_percentiles: list[float] = _DEFAULT_PERCENTILES,
    multipliers: list[float] = _DEFAULT_MULTIPLIERS,
) -> None:
    """
    Generate SLO tables for all ``(percentile, multiplier)`` combinations.

    Reads latencies from ``data/runs/<run_id>/structured_log.parquet`` and
    writes one YAML file per combination to ``data/slos/``.

    Parameters:
        run_id: Run identifier (e.g. ``"1778469201294"``).
        baseline_percentiles: Percentiles (0–100) to sweep.
        multipliers: Headroom multipliers (*k*) to sweep.
    """
    log_path = Path(pu.get_runs_path()) / run_id / "structured_log.parquet"
    if not log_path.exists():
        raise FileNotFoundError(
            f"No structured_log.parquet found for run {run_id!r} "
            f"(expected at {log_path})."
        )

    df = query_latencies_from_log(log_path)
    df["template_id"] = df["query_text_id"].map(
        lambda qid: QueryTextId(qid).template_id
    )
    print(f"Loaded {len(df)} latency records from run {run_id!r}.")

    exec_cfg_path = Path(pu.get_runs_path()) / run_id / "execution_config.yml"
    if exec_cfg_path.exists():
        runner_cfg = WorkloadRunnerConfig.from_config(load_yaml(exec_cfg_path))
        if runner_cfg.closed_loop:
            print(
                "Run is closed-loop — latencies are suitable for SLO baselines."
            )
        else:
            print(
                "WARNING: run is not closed-loop; latencies may reflect queuing effects."
            )
    else:
        print(
            "WARNING: execution_config.yml not found; could not verify closed-loop mode."
        )

    slo_dir = os.path.join(pu.get_data_path(), "slos")
    os.makedirs(slo_dir, exist_ok=True)

    for p in baseline_percentiles:
        baseline = df.groupby("template_id")["latency_s"].quantile(p / 100.0)
        for k in multipliers:
            slo_dict = {
                str(t): round(float(v * k), 3) for t, v in baseline.items()
            }
            out_path = os.path.join(slo_dir, f"{run_id}_p{int(p)}_k{int(k)}.yml")
            dump_yaml(
                {
                    "run_id": run_id,
                    "baseline_percentile": p,
                    "multiplier": k,
                    "num_templates": len(slo_dict),
                    "slo_dict": slo_dict,
                },
                out_path,
            )
            print(f"  Wrote SLO table  p={p}  k={k}  → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate SLO tables from a run's structured log. Writes one YAML "
            "file per (percentile, multiplier) combination to data/slos/."
        )
    )
    parser.add_argument(
        "--run_id", required=True, help="Run identifier (e.g. 1778469201294)."
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=_DEFAULT_PERCENTILES,
        metavar="P",
        help=f"Baseline percentiles to sweep (default: {_DEFAULT_PERCENTILES}).",
    )
    parser.add_argument(
        "--multipliers",
        nargs="+",
        type=int,
        default=_DEFAULT_MULTIPLIERS,
        metavar="K",
        help=f"Headroom multipliers to sweep (default: {_DEFAULT_MULTIPLIERS}).",
    )
    args = parser.parse_args()
    generate_slo_tables(
        run_id=args.run_id,
        baseline_percentiles=args.percentiles,
        multipliers=args.multipliers,
    )
