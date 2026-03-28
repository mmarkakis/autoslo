"""Launch the history-window experiment (3 scenarios sequentially).

Usage
-----
::

    python experiments/23_history_window/run_experiment.py \\
        --config data/__run_configs/test.yml \\
        --trace data/__workloads/ext_tpcds1000/redbench_provisioned_157_0.parquet \\
        [--out-dir data/tuner_runs/history_exp]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from autoslo.tuner.config import load_tuner_config
from autoslo.tuner.policy_tuner import PolicyTuner

EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = EXPERIMENT_DIR / "configs"

SCENARIOS = [
    ("prev_day", CONFIG_DIR / "tuner_prev_day.yml"),
    ("prev_week", CONFIG_DIR / "tuner_prev_week.yml"),
    ("prev_month", CONFIG_DIR / "tuner_prev_month.yml"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the history-window depth experiment.",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "base_config.yml"),
        help="Path to the base simulator YAML config (default: configs/base_config.yml).",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to the historical workload Parquet file.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/tuner_runs/history_exp",
        help="Root output directory for the three runs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    trace_path = Path(args.trace)
    if not trace_path.exists():
        parser.error(f"Trace file not found: {trace_path}")

    with open(args.config) as f:
        base_config: dict = yaml.safe_load(f)

    out_root = Path(args.out_dir)

    for scenario_name, tuner_config_path in SCENARIOS:
        run_dir = out_root / scenario_name
        print(f"\n{'=' * 60}")
        print(f"  Scenario: {scenario_name}")
        print(f"  Tuner config: {tuner_config_path}")
        print(f"  Run dir: {run_dir}")
        print(f"{'=' * 60}\n")

        tuner_config = load_tuner_config(tuner_config_path)
        tuner = PolicyTuner(base_config, tuner_config, run_dir=run_dir)
        final_path = tuner.tune([trace_path])

        print(f"\n  [{scenario_name}] Done. Final config: {final_path}\n")

    print(f"\nAll scenarios complete. Results in: {out_root}")
    print(
        "Run the aggregation script to compare:\n"
        f"  python experiments/23_history_window/aggregate_results.py "
        f"--run-dir {out_root}"
    )


if __name__ == "__main__":
    main()
