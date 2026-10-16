"""
tune_phase7_summary.py
----------------------
For each entry in a tuning manifest, print only the Phase 7 output (the
"Initial vs Final" comparison table) — without re-running any tuning.

Reads the already-persisted summaries from each tuning run's output
directory (data/tuner_runs/<run_id>/{03_baseline,07_final}/{train,val}_summary.yml)
and the run's aggregation method / SLO metric from its persisted
tuner_config.yml, then reconstructs and prints the same comparison table
that PolicyTuner.tune()'s Phase 7 prints.

Entries whose tuning run hasn't reached Phase 7 yet (missing summary files)
are reported as not-yet-available and skipped.

Note: scenario-level min/max ranges are not reconstructed (would require
rebuilding full ExecutionResult objects) — only the aggregated metrics are
shown.

Usage:
    python tools/tune_phase7_summary.py --tuning_manifest_path main_eval_v8_extra_clusters
    python tools/tune_phase7_summary.py --tuning_manifest_path /abs/path/to/manifest.yml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.workload_execution.aggregated_execution_results import (
    AggregatedExecutionResults,
)

console = Console()


def _load_agg(path: Path) -> AggregatedExecutionResults | None:
    if not path.exists():
        return None
    d = load_yaml(path)
    return AggregatedExecutionResults(
        violation_rate=d["violation_rate"],
        violation_amount_s=d["violation_amount_s"],
        violation_relative_mean=d["violation_relative_mean"],
        cost=d["cost"],
        scenario_results=[],
    )


def print_phase7_for_run(run_id: str) -> None:
    run_dir = pu.get_data_dir() / "tuner_runs" / run_id
    tuner_config_path = run_dir / "tuner_config.yml"
    baseline_train_path = run_dir / "03_baseline" / "train_summary.yml"
    baseline_val_path = run_dir / "03_baseline" / "val_summary.yml"
    final_train_path = run_dir / "07_final" / "train_summary.yml"
    final_val_path = run_dir / "07_final" / "val_summary.yml"

    missing = [
        p
        for p in [
            tuner_config_path,
            baseline_train_path,
            baseline_val_path,
            final_train_path,
            final_val_path,
        ]
        if not p.exists()
    ]
    if missing:
        console.print(
            f"[yellow]'{run_id}': Phase 7 not available yet "
            f"(missing {missing[0].relative_to(run_dir)}).[/]"
        )
        return

    tuner_config = load_yaml(tuner_config_path)
    agg_method = tuner_config["sampling_config"]["aggregation_method"]
    slo_metric = tuner_config["slo_objective_config"]["slo_metric"]

    baseline_train = _load_agg(baseline_train_path)
    baseline_val = _load_agg(baseline_val_path)
    tuned_train = _load_agg(final_train_path)
    tuned_val = _load_agg(final_val_path)

    console.rule(f"[bold cyan]{run_id}[/]")
    AggregatedExecutionResults.print_comparison(
        ("Initial (train)", baseline_train),
        ("Final (train)", tuned_train),
        console=console,
        agg_method=agg_method,
        slo_metric=slo_metric,
    )
    AggregatedExecutionResults.print_comparison(
        ("Initial (val)", baseline_val),
        ("Final (val)", tuned_val),
        console=console,
        agg_method=agg_method,
        slo_metric=slo_metric,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print only the Phase 7 (initial vs final) comparison table for "
            "each entry in a tuning manifest, without re-running tuning."
        )
    )
    parser.add_argument(
        "--tuning_manifest_path",
        required=True,
        help=(
            "Name of the tuning manifest (resolved under "
            "data/manifests/tuning) or an explicit path to a .yml file."
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.tuning_manifest_path)
    if not manifest_path.is_absolute():
        name = (
            args.tuning_manifest_path
            if args.tuning_manifest_path.endswith(".yml")
            else args.tuning_manifest_path + ".yml"
        )
        manifest_path = pu.get_data_dir() / "manifests" / "tuning" / name
    if not manifest_path.exists():
        parser.error(f"Tuning manifest not found: {manifest_path}")

    manifest = load_yaml(manifest_path)
    for run_id in manifest.get("main_content", {}).keys():
        print_phase7_for_run(run_id)


if __name__ == "__main__":
    main()
