"""CLI entry point: ``python -m autoslo.tuner``

Usage
-----
::

    python -m autoslo.tuner \\
        --config data/__run_configs/my_config.yml \\
        --tuner-config config/tuner_config.yml \\
        --traces data/workloads/trace_a.parquet data/workloads/trace_b.parquet \\
        --set slo_config.slo_s=5.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from autoslo.tuner.config import load_tuner_config
from autoslo.tuner.policy_tuner import PolicyTuner
from autoslo.utils.config import apply_overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the automated policy tuner.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the initial YAML config file (e.g. data/__run_configs/test.yml).",
    )
    parser.add_argument(
        "--tuner-config",
        required=True,
        help="Path to the tuner settings YAML file.",
    )
    parser.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="One or more historical workload Parquet files.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override initial config values using dot-delimited keys, e.g. "
            "--set slo_config.slo_s=5.0 basic_config.schema_name=my_schema"
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional explicit run directory (otherwise auto-generated).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load initial simulator config.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply --set overrides.
    overrides: dict[str, object] = {}
    for item in getattr(args, "set"):
        key, sep, val = item.partition("=")
        if not key or not sep:
            parser.error(
                f"Invalid --set format: {item!r}  (expected KEY=VALUE)"
            )
        overrides[key] = yaml.safe_load(val)
    apply_overrides(cfg, overrides)

    # Load tuner config.
    tuner_config = load_tuner_config(args.tuner_config)

    # Resolve trace paths.
    traces = [Path(t) for t in args.traces]
    for t in traces:
        if not t.exists():
            parser.error(f"Trace file not found: {t}")

    run_dir = Path(args.run_dir) if args.run_dir else None

    tuner = PolicyTuner(cfg, tuner_config, run_dir=run_dir)
    final_config_path = tuner.tune(traces)

    print(f"\nDone. Final config: {final_config_path}")


if __name__ == "__main__":
    main()
