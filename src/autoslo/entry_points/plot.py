from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import (
    SloObjectiveConfig,
    SloResolverConfig,
    WorkloadConfig,
)
from autoslo.config.utils import make_run_id
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.path_utils import (
    find_most_recent_live_run_id,
    is_up_to_date,
)
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.visualizations.scatter_plots import (
    ImprovementArrow,
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
    all_points_specs: list[list[dict]],
    sim_runs_dir: Path,
    live: bool = False,
) -> bool:
    inputs = [manifest_path]
    runs_dir = Path(pu.get_runs_path())
    for points_spec in all_points_specs:
        for point in points_spec:
            workload_config = WorkloadConfig.from_config(point)
            exec_cfg_path = resolve_config(point["execution_config"])
            params = point.get("params", {})
            config_label = make_run_id([exec_cfg_path.stem], params)
            if live:
                run_id = find_most_recent_live_run_id(
                    config_label, workload_config.id()
                )
                if run_id is None:
                    continue
                run_dir = runs_dir / run_id
            else:
                run_dir = sim_runs_dir / workload_config.id() / config_label
            inputs.append(run_dir / "execution_config.yml")
    return is_up_to_date(plot_path, *inputs)


def _load_scatter_points(
    points_spec: list[dict],
    slo_obj: SloObjective,
    slo_resolver: SloResolver,
    sim_runs_dir: Path,
    live: bool = False,
    tail_fraction: float = 1.0,
) -> list[ScatterPoint]:
    scatter_points: list[ScatterPoint] = []
    runs_dir = Path(pu.get_runs_path())
    for point in points_spec:
        workload_config = WorkloadConfig.from_config(point)
        exec_cfg_path = resolve_config(point["execution_config"])
        params = point.get("params", {})
        config_label = make_run_id([exec_cfg_path.stem], params)
        if live:
            run_id = find_most_recent_live_run_id(
                config_label, workload_config.id()
            )
            if run_id is None:
                console.print(
                    f"[yellow]Warning: no live run found for workload "
                    f"'{workload_config.id()}' / config '{config_label}' "
                    f"— skipping point '{point['label']}'.[/]"
                )
                continue
            run_dir = runs_dir / run_id
            if not (run_dir / "structured_log.parquet").exists():
                console.print(
                    f"[yellow]Warning: execution directory has no structured log "
                    f"for live run '{run_id}' — skipping point '{point['label']}'.[/]"
                )
                continue
        else:
            run_dir = sim_runs_dir / workload_config.id() / config_label
            if not (run_dir / "execution_config.yml").exists():
                console.print(
                    f"[yellow]Warning: simulation run not found at {run_dir} "
                    f"— skipping point '{point['label']}'.[/]"
                )
                continue
        result = ExecutionResult.load(
            run_dir, slo_resolver=slo_resolver, tail_fraction=tail_fraction
        )
        scatter_points.append(
            ScatterPoint(
                formatting_id=point["formatting_id"],
                label=point["label"],
                x=_x_value(result, slo_obj.slo_metric),
                y=result.total_cost,
            )
        )
    return scatter_points


def _generate_single_panel_plot(
    content: dict,
    manifest_path: Path,
    plot_path: Path,
    plot_name: str,
    sim_runs_dir: Path,
    plots_dir: Path,
    force: bool,
    live: bool = False,
) -> None:
    points_spec: list[dict] = content["points"]

    if not force and _plot_is_up_to_date(
        manifest_path, plot_path, [points_spec], sim_runs_dir, live=live
    ):
        console.print(f"[dim]Skipping '{plot_name}' (up to date)[/]")
        return

    slo_obj = SloObjective(SloObjectiveConfig.from_config(content))
    slo_resolver = SloResolver(SloResolverConfig.from_config(content))

    show_target_region: bool = content.get("show_target_region", False)
    tail_fraction: float = content.get("tail_fraction", 1.0)

    improvement_arrow_spec = content.get("improvement_arrow")
    improvement_arrow = (
        ImprovementArrow(
            base_label=improvement_arrow_spec["base_label"],
            target_label=improvement_arrow_spec["target_label"],
        )
        if improvement_arrow_spec
        else None
    )

    scatter_points = _load_scatter_points(
        points_spec,
        slo_obj,
        slo_resolver,
        sim_runs_dir,
        live=live,
        tail_fraction=tail_fraction,
    )

    if not scatter_points:
        console.print(
            f"[yellow]No data points found for '{plot_name}' — nothing to plot.[/]"
        )
        return

    title: str | None = content.get("title") or None
    fig, ax, _, _ = cost_vs_compliance_scatter(
        scatter_points,
        x_metric=slo_obj.slo_metric,
        x_threshold_objective=slo_obj if show_target_region else None,
        title=title,
        show_legend=content.get("show_legend", False),
        improvement_arrow=improvement_arrow,
    )
    _ann_lines = []
    if not live:
        _ann_lines.append("[Simulated]")
    if tail_fraction < 1.0:
        _ann_lines.append(f"[Viol. over last {tail_fraction * 100:.0f}%]")
    if _ann_lines:
        ax.text(
            0.98,
            0.02,
            "\n".join(_ann_lines),
            transform=ax.transAxes,
            color="red",
            ha="right",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {plot_path}")


def _generate_multi_panel_plot(
    manifest: dict,
    manifest_path: Path,
    plot_path: Path,
    plot_name: str,
    sim_runs_dir: Path,
    plots_dir: Path,
    force: bool,
    live: bool = False,
) -> None:
    layout: dict = manifest.get("layout", {})
    panels_spec: list[dict] = manifest["panels"]

    # Validate: no duplicate (row, col).
    seen_positions: set[tuple[int, int]] = set()
    for panel in panels_spec:
        pos = (panel["row"], panel["col"])
        if pos in seen_positions:
            raise ValueError(
                f"Duplicate panel position {pos} in manifest '{plot_name}'."
            )
        seen_positions.add(pos)

    # Infer grid dimensions from panel positions, allow explicit overrides.
    max_row = max(p["row"] for p in panels_spec)
    max_col = max(p["col"] for p in panels_spec)
    rows: int = layout.get("rows", max_row + 1)
    cols: int = layout.get("cols", max_col + 1)

    # Validate all panels are within bounds.
    for panel in panels_spec:
        if panel["row"] >= rows or panel["col"] >= cols:
            raise ValueError(
                f"Panel at ({panel['row']}, {panel['col']}) is out of bounds "
                f"for a {rows}x{cols} grid in manifest '{plot_name}'."
            )

    # Up-to-date check across all panels.
    all_points_specs = [p["points"] for p in panels_spec]
    if not force and _plot_is_up_to_date(
        manifest_path, plot_path, all_points_specs, sim_runs_dir, live=live
    ):
        console.print(f"[dim]Skipping '{plot_name}' (up to date)[/]")
        return

    figsize: tuple[float, float] = tuple(
        layout.get("figsize", [6 * cols, 5 * rows])
    )
    shared_xlim: bool = layout.get("shared_xlim", False)
    shared_ylim: bool = layout.get("shared_ylim", False)
    show_legend: bool = layout.get("show_legend", False)
    show_target_region: bool = layout.get("show_target_region", False)

    fig, axes_2d = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    rendered_xlims: list[tuple[float, float]] = []
    rendered_ylims: list[tuple[float, float]] = []
    rendered_axes: list[Axes] = []

    for panel in panels_spec:
        row, col = panel["row"], panel["col"]
        ax: Axes = axes_2d[row][col]

        slo_obj = SloObjective(SloObjectiveConfig.from_config(panel))
        slo_resolver = SloResolver(SloResolverConfig.from_config(panel))
        points_spec: list[dict] = panel["points"]

        improvement_arrow_spec = panel.get("improvement_arrow")
        improvement_arrow = (
            ImprovementArrow(
                base_label=improvement_arrow_spec["base_label"],
                target_label=improvement_arrow_spec["target_label"],
            )
            if improvement_arrow_spec
            else None
        )

        tail_fraction: float = panel.get("tail_fraction", 1.0)
        scatter_points = _load_scatter_points(
            points_spec,
            slo_obj,
            slo_resolver,
            sim_runs_dir,
            live=live,
            tail_fraction=tail_fraction,
        )

        # if not scatter_points:
        #     console.print(
        #         f"[yellow]No data points found for panel ({row}, {col}) "
        #         f"in '{plot_name}' — leaving panel blank.[/]"
        #     )
        #     ax.set_visible(False)
        #     continue

        title: str | None = panel.get("title") or None
        _, _, xlims, ylims = cost_vs_compliance_scatter(
            scatter_points,
            x_metric=slo_obj.slo_metric,
            x_threshold_objective=(
                slo_obj
                if (
                    show_target_region or panel.get("show_target_region", False)
                )
                else None
            ),
            title=title,
            ax=ax,
            show_legend=show_legend or panel.get("show_legend", False),
            improvement_arrow=improvement_arrow,
        )
        _ann_lines = []
        if not live:
            _ann_lines.append("[Simulated]")
        if tail_fraction < 1.0:
            _ann_lines.append(f"[Viol. over last {tail_fraction * 100:.0f}%]")
        if _ann_lines:
            ax.text(
                0.98,
                0.02,
                "\n".join(_ann_lines),
                transform=ax.transAxes,
                color="red",
                ha="right",
                va="bottom",
                fontsize=9,
            )
        rendered_xlims.append(xlims)
        rendered_ylims.append(ylims)
        rendered_axes.append(ax)

    # Apply shared axis limits if requested.
    if shared_xlim and rendered_xlims:
        unified_left = min(lims[0] for lims in rendered_xlims)
        unified_right = max(lims[1] for lims in rendered_xlims)
        padding = (unified_right - unified_left) * 0.05
        for ax in rendered_axes:
            ax.set_xlim(0, unified_right + padding)

    if shared_ylim and rendered_ylims:
        unified_bottom = min(lims[0] for lims in rendered_ylims)
        unified_top = max(lims[1] for lims in rendered_ylims)
        padding = (unified_top - unified_bottom) * 0.05
        for ax in rendered_axes:
            ax.set_ylim(0, unified_top + padding)

    # Hide unused axes.
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in seen_positions:
                axes_2d[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {plot_path}")

    if show_legend:
        legend_path = plots_dir / f"{plot_name}_legend.png"
        plot_legend_to(legend_path)
        console.print(f"[green]Saved:[/] {legend_path}")


def _generate_plot(
    manifest_path: Path, force: bool, live: bool = False
) -> None:
    manifest = load_yaml(manifest_path)
    plot_name = manifest_path.stem

    data_path = Path(pu.get_data_path())
    sim_runs_dir = data_path / "simulator_runs"
    plots_dir = data_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{plot_name}_live" if live else plot_name
    plot_path = plots_dir / f"{stem}.png"

    if "panels" in manifest["main_content"]:
        _generate_multi_panel_plot(
            manifest["main_content"],
            manifest_path,
            plot_path,
            plot_name,
            sim_runs_dir,
            plots_dir,
            force,
            live=live,
        )
    else:
        _generate_single_panel_plot(
            manifest["main_content"],
            manifest_path,
            plot_path,
            plot_name,
            sim_runs_dir,
            plots_dir,
            force,
            live=live,
        )


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
            _generate_plot(manifest_path, force=args.force, live=args.live)
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

    _generate_plot(manifest_path, force=args.force, live=args.live)
    console.print("\n[bold green]Done.[/]")


if __name__ == "__main__":
    main()
