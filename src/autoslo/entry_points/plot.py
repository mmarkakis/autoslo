from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import (
    SloObjectiveConfig,
    SloResolverConfig,
    WorkloadConfig,
)
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.path_utils import is_up_to_date
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.visualizations.scatter_plots import (
    ScatterPoint,
    cost_vs_compliance_scatter,
    plot_legend_to,
)
from autoslo.workload_execution.execution_result import ExecutionResult

console = Console()


def _x_value(result: ExecutionResult, metric: SloMetric) -> float:
    if metric is SloMetric.BINARY:
        return result.violation_rate
    if metric is SloMetric.ABSOLUTE_S:
        return result.violation_amount_s
    if metric in (SloMetric.RELATIVE, SloMetric.RELATIVE_UNCONSTRAINED):
        return result.violation_relative_mean
    raise ValueError(f"Unsupported SloMetric for plotting: {metric}")


def _plot_is_up_to_date(
    manifest_path: Path,
    plot_path: Path,
    points: list[dict],
    sim_runs_dir: Path,
) -> bool:
    inputs = [manifest_path]
    for point in points:
        workload_config = WorkloadConfig.from_config(point)
        exec_cfg_path = resolve_config(point["execution_config"])
        run_dir = sim_runs_dir / workload_config.id() / exec_cfg_path.stem
        inputs.append(run_dir / "execution_config.yml")
    return is_up_to_date(plot_path, *inputs)


def _generate_plot(manifest_path: Path, force: bool) -> None:
    manifest = load_yaml(manifest_path)
    content = manifest["main_content"]
    plot_name = manifest_path.stem

    data_path = Path(pu.get_data_path())
    sim_runs_dir = data_path / "simulator_runs"
    plots_dir = data_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{plot_name}.png"

    points_spec: list[dict] = content["points"]

    if not force and _plot_is_up_to_date(
        manifest_path, plot_path, points_spec, sim_runs_dir
    ):
        console.print(f"[dim]Skipping '{plot_name}' (up to date)[/]")
        return

    slo_obj = SloObjective(SloObjectiveConfig.from_config(content))
    slo_resolver = SloResolver(SloResolverConfig.from_config(content))

    scatter_points: list[ScatterPoint] = []
    for point in points_spec:
        workload_config = WorkloadConfig.from_config(point)
        exec_cfg_path = resolve_config(point["execution_config"])
        run_dir = sim_runs_dir / workload_config.id() / exec_cfg_path.stem

        if not (run_dir / "execution_config.yml").exists():
            console.print(
                f"[yellow]Warning: simulation run not found at {run_dir} "
                f"— skipping point '{point['label']}'.[/]"
            )
            continue

        result = ExecutionResult.load(run_dir, slo_resolver=slo_resolver)
        scatter_points.append(
            ScatterPoint(
                formatting_id=point["formatting_id"],
                label=point["label"],
                x=_x_value(result, slo_obj.slo_metric),
                y=result.total_cost,
            )
        )

    if not scatter_points:
        console.print(
            f"[yellow]No data points found for '{plot_name}' — nothing to plot.[/]"
        )
        return

    title: str | None = content.get("title") or None
    fig, _, _, _ = cost_vs_compliance_scatter(
        scatter_points,
        x_metric=slo_obj.slo_metric,
        x_threshold_objective=slo_obj,
        title=title,
    )
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {plot_path}")

    if content.get("show_legend", False):
        legend_path = plots_dir / f"{plot_name}_legend.png"
        plot_legend_to(legend_path)
        console.print(f"[green]Saved:[/] {legend_path}")


def main() -> None:
    # Argument parsing.
    description = "Generate a plot from a plot manifest."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--plotting_manifest_path",
        help=(
            "Name of the plot manifest (resolved under "
            "data/manifests/plotting) or an explicit path to a .yml file."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all plots defined in the manifest directory.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run sequentially using the workload runner, not the simulator.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render the plot even if it is already up to date.",
    )
    args = parser.parse_args()

    manifests_dir = Path(pu.get_data_path()) / "manifests" / "plotting"

    if args.all:
        manifest_paths = sorted(manifests_dir.glob("*.yml"))
        if not manifest_paths:
            console.print(
                "[yellow]No plot manifests found in the manifest directory.[/]"
            )
            return
        for manifest_path in manifest_paths:
            _generate_plot(manifest_path, force=args.force)
        console.print("\n[bold green]Done.[/]")
        return

    if not args.plotting_manifest_path:
        parser.error("one of --plotting_manifest_path or --all is required")

    manifest_path = Path(args.plotting_manifest_path)
    if not manifest_path.is_absolute():
        name = (
            args.plotting_manifest_path
            if args.plotting_manifest_path.endswith(".yml")
            else args.plotting_manifest_path + ".yml"
        )
        manifest_path = manifests_dir / name
    if not manifest_path.exists():
        parser.error(f"Plot manifest not found: {manifest_path}")

    _generate_plot(manifest_path, force=args.force)
    console.print("\n[bold green]Done.[/]")


if __name__ == "__main__":
    main()
