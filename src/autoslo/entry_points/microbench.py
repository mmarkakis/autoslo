from __future__ import annotations

import argparse
import shutil

from rich.console import Console

from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.microbenchmarks.autoscaling_efficiency import (
    AutoscalingEfficiencyBenchmark,
)
from autoslo.microbenchmarks.microbenchmark_runner import MicrobenchmarkRunner
from autoslo.microbenchmarks.routing_efficiency import (
    RoutingEfficiencyBenchmark,
)
from autoslo.microbenchmarks.scenario_evaluator_efficiency import (
    ScenarioEvaluatorEfficiencyBenchmark,
)
from autoslo.microbenchmarks.spinup_optimizer_efficiency import (
    SpinupOptimizerEfficiencyBenchmark,
)

console = Console()

KNOWN_RUNNERS: list[type[MicrobenchmarkRunner]] = [
    RoutingEfficiencyBenchmark,
    AutoscalingEfficiencyBenchmark,
    ScenarioEvaluatorEfficiencyBenchmark,
    SpinupOptimizerEfficiencyBenchmark,
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run manifest-driven microbenchmarks from "
            "data/manifests/microbench."
        )
    )
    parser.add_argument(
        "benchmark_name",
        help=(
            "Name of the microbenchmark to run. Must match the name of a "
            "manifest in data/manifests/microbench, with the .yml suffix."
        ),
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help=("Regenerate plots from existing results."),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate results, if they exist.",
    )
    args = parser.parse_args()

    # Resolve key paths.
    bm_cls = next(
        (
            runner
            for runner in KNOWN_RUNNERS
            if runner.name() == args.benchmark_name
        ),
        None,
    )
    if not bm_cls:
        parser.error(
            f"Unsupported benchmark '{args.benchmark_name}'. "
            f"Supported: {', '.join(r.name() for r in KNOWN_RUNNERS)}."
        )
    if not bm_cls.manifest_path().exists():
        parser.error(f"Manifest not found: {bm_cls.manifest_path()}")

    # Load and check manifest.
    console.print("[cyan]Reading manifest...[/]")
    manifest = load_yaml(bm_cls.manifest_path())
    missing = [k for k in bm_cls.required_keys() if k not in manifest]
    if missing:
        missing_s = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing required key(s) for {bm_cls.name()}: {missing_s}"
        )

    # Run benchmark, if needed.
    if not args.plot_only:
        if bm_cls.csv_path().exists() and not args.force:
            console.print(
                f"[yellow]Results already exist: {bm_cls.csv_path()}.[/]"
                " Use --force to overwrite or --plot-only to regenerate plots."
            )
            return

        if bm_cls.scratch_dir().exists():
            shutil.rmtree(bm_cls.scratch_dir())
        bm_cls.scratch_dir().mkdir(parents=True, exist_ok=True)
        console.print("[cyan]Running benchmark...[/]")
        bm_cls.run_from_manifest(manifest)
        console.print("[bold green]Benchmark run completed.[/]")

    # Plot results and return.
    if not bm_cls.csv_path().exists():
        console.print(
            f"[red]Error: Results file not found: {bm_cls.csv_path()}.[/]"
            " Cannot generate plot."
        )
        return
    bm_cls.plot()
    console.print("[bold green]Microbenchmark completed.[/]")
    console.print(f"  csv: {bm_cls.csv_path()}")
    console.print(f"  plot: {bm_cls.plot_path()}")


if __name__ == "__main__":
    main()
