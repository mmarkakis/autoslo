"""
SLO-table generation from a single run's structured log.

Writes one YAML SLO table per (percentile, multiplier) combination to
``data/slos/``, compatible with
:class:`autoslo.blueprint_selection.slo_resolver.SloResolver`.

Run with ``--help`` for the full list of options.
"""

import argparse

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import WorkloadRunnerConfig
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.filesystem.yaml_helpers import dump_yaml, load_yaml
from autoslo.workload_definition.query import QueryTextId

_DEFAULT_PERCENTILES = [-1, 50, 100]
_DEFAULT_MULTIPLIERS = [1, 2, 3, 4, 5, 8, 10]


def generate_slo_tables(
    run_id: str,
    baseline_percentiles: list[int] = _DEFAULT_PERCENTILES,
    multipliers: list[int] = _DEFAULT_MULTIPLIERS,
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
    log_path = pu.get_runs_path() / run_id / "structured_log.parquet"
    if not log_path.exists():
        raise FileNotFoundError(
            f"No structured_log.parquet found for run {run_id!r} "
            f"(expected at {log_path})."
        )

    df = StructuredLog.load(log_path).query_latencies(drop_incomplete=True)
    df["template_id"] = df["query_text_id"].map(
        lambda qid: QueryTextId(qid).template_id
    )
    print(f"Loaded {len(df)} latency records from run {run_id!r}.")

    exec_cfg_path = pu.get_runs_path() / run_id / "execution_config.yml"
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

    slo_dir = pu.get_data_path() / "slos"
    slo_dir.mkdir(parents=True, exist_ok=True)

    for p in baseline_percentiles:
        mid_part = f"p{int(p)}" if p != -1 else "mean"
        if p == -1:
            baseline = df.groupby("template_id")["latency_s"].mean()
        else:
            baseline = df.groupby("template_id")["latency_s"].quantile(
                p / 100.0
            )
        for k in multipliers:
            slo_dict = {
                str(t): round(float(v * k), 3) for t, v in baseline.items()
            }
            out_path = slo_dir / f"{run_id}_{mid_part}_k{int(k)}.yml"
            dump_yaml(
                {
                    "run_id": run_id,
                    "agg_method": mid_part,
                    "multiplier": k,
                    "num_templates": len(slo_dict),
                    "slo_dict": slo_dict,
                },
                out_path,
            )
            print(
                f"  Wrote SLO table  agg_method={mid_part}  k={k}  → {out_path}"
            )


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
