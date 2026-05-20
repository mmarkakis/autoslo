from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import Cluster
from autoslo.config.component_configs import (
    SloObjectiveConfig,
    SloResolverConfig,
    WorkloadConfig,
)
from autoslo.config.utils import make_run_id
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.path_utils import find_most_recent_live_run_id
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.filesystem.yaml_helpers import load_yaml
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


def _cluster_annotation(run_dir: Path) -> str | None:
    """
    Return the size of the index-1 cluster spun up.
    """
    log_path = run_dir / "structured_log.parquet"
    if not log_path.exists():
        return None
    df = StructuredLog.load(log_path).df
    if df.empty:
        return None

    unique_nonempty_cluster_names = df["cluster_name"].dropna().unique()
    target_name = [
        name for name in unique_nonempty_cluster_names if 
        (name.strip() and (Cluster.counter_for_cluster_name(name) == 1))
    ]
    if len(target_name) != 1:
        return None

    return f"{Cluster.rpu_for_cluster_name(target_name[0])}"


@dataclass
class _PanelData:
    """All resolved, typed inputs needed to render one scatter panel."""

    scatter_points: list[ScatterPoint]
    slo_obj: SloObjective
    x_threshold_objective: SloObjective | None
    title: str | None
    show_legend: bool
    improvement_arrow: ImprovementArrow | None
    tail_fraction: float


def _load_panel_data(
    panel: dict,
    sim_runs_dir: Path,
    live: bool,
    *,
    layout_show_target_region: bool = False,
    layout_show_legend: bool = False,
    layout_annotate_cluster_sizes: bool = False,
) -> _PanelData:
    """Parse a panel config dict into a fully-resolved _PanelData.

    This is the single place that knows the panel dict schema.  All
    layout-level defaults are merged here so downstream rendering
    functions operate purely on typed objects.
    """
    slo_obj = SloObjective(SloObjectiveConfig.from_config(panel))
    slo_resolver = SloResolver(SloResolverConfig.from_config(panel))
    tail_fraction: float = panel.get("tail_fraction", 1.0)
    show_target_region = layout_show_target_region or panel.get(
        "show_target_region", False
    )
    show_legend = layout_show_legend or panel.get("show_legend", False)
    annotate_cluster_sizes: bool = layout_annotate_cluster_sizes or panel.get(
        "annotate_with_cluster_sizes", False
    )

    arrow_spec = panel.get("improvement_arrow")
    improvement_arrow = (
        ImprovementArrow(
            base_label=arrow_spec["base_label"],
            target_label=arrow_spec["target_label"],
        )
        if arrow_spec
        else None
    )

    scatter_points: list[ScatterPoint] = []
    runs_dir = Path(pu.get_runs_path())
    for point in panel["points"]:
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
                    f"[yellow]Warning: execution directory has no structured "
                    f"log for live run '{run_id}' — skipping point "
                    f"'{point['label']}'.[/]"
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
        annotation = (
            _cluster_annotation(run_dir) if annotate_cluster_sizes else None
        )
        scatter_points.append(
            ScatterPoint(
                formatting_id=point["formatting_id"],
                label=point["label"],
                x=result.violation_for_metric(slo_obj.slo_metric),
                y=result.total_cost,
                annotation=annotation,
            )
        )

    return _PanelData(
        scatter_points=scatter_points,
        slo_obj=slo_obj,
        x_threshold_objective=slo_obj if show_target_region else None,
        title=panel.get("title") or None,
        show_legend=show_legend,
        improvement_arrow=improvement_arrow,
        tail_fraction=tail_fraction,
    )


def _save_points_csv(
    csv_path: Path,
    panels: list[tuple[int | None, int | None, _PanelData]],
) -> None:
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["row", "col", "panel_title", "x_metric", "label", "x", "y"]
        )
        for row, col, panel_data in panels:
            for point in panel_data.scatter_points:
                writer.writerow(
                    [
                        "" if row is None else row,
                        "" if col is None else col,
                        panel_data.title or "",
                        panel_data.slo_obj.slo_metric.value,
                        point.label,
                        point.x,
                        point.y,
                    ]
                )
    console.print(f"[green]Saved:[/] {csv_path}")


def _annotate_ax(ax: Axes, panel_data: _PanelData, live: bool) -> None:
    lines = []
    if not live:
        lines.append("[Simulated]")
    if panel_data.tail_fraction < 1.0:
        lines.append(f"[Viol. over last {panel_data.tail_fraction * 100:.0f}%]")
    if lines:
        ax.text(
            0.98,
            0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            color="red",
            ha="right",
            va="bottom",
            fontsize=10,
        )


def _render_and_save_figure(
    panel_data: _PanelData,
    plot_path: Path,
    live: bool,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Render panel_data as a standalone figure and save it."""
    fig, ax, _, _ = cost_vs_compliance_scatter(
        panel_data.scatter_points,
        x_metric=panel_data.slo_obj.slo_metric,
        x_threshold_objective=panel_data.x_threshold_objective,
        title=panel_data.title,
        show_legend=panel_data.show_legend,
        improvement_arrow=panel_data.improvement_arrow,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _annotate_ax(ax, panel_data, live)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {plot_path}")


def _generate_single_panel_plot(
    content: dict,
    plot_path: Path,
    sim_runs_dir: Path,
    force: bool,
    live: bool,
) -> None:
    if not force and plot_path.exists():
        console.print(f"[dim]Skipping '{plot_path.stem}' (exists)[/]")
        return

    panel_data = _load_panel_data(content, sim_runs_dir, live)
    if not panel_data.scatter_points:
        console.print(
            f"[yellow]No data points found for '{plot_path.stem}' — "
            "nothing to plot.[/]"
        )
        return

    _render_and_save_figure(panel_data, plot_path, live)
    _save_points_csv(plot_path.with_suffix(".csv"), [(None, None, panel_data)])


def _generate_multi_panel_plot(
    manifest: dict,
    plot_path: Path,
    sim_runs_dir: Path,
    plots_dir: Path,
    force: bool,
    live: bool,
) -> None:
    layout: dict = manifest.get("layout", {})
    panels_spec: list[dict] = manifest["panels"]
    plot_name = plots_dir.name

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
    rows: int = layout.get("rows", max(p["row"] for p in panels_spec) + 1)
    cols: int = layout.get("cols", max(p["col"] for p in panels_spec) + 1)

    # Validate all panels are within bounds.
    for panel in panels_spec:
        if panel["row"] >= rows or panel["col"] >= cols:
            raise ValueError(
                f"Panel at ({panel['row']}, {panel['col']}) is out of bounds "
                f"for a {rows}x{cols} grid in manifest '{plot_name}'."
            )

    if not force and plot_path.exists():
        console.print(f"[dim]Skipping '{plot_name}' (exists)[/]")
        return

    figsize: tuple[float, float] = tuple(
        layout.get("figsize", [6 * cols, 5 * rows])
    )
    shared_xlim: bool = layout.get("shared_xlim", False)
    shared_ylim: bool = layout.get("shared_ylim", False)
    show_legend: bool = layout.get("show_legend", False)
    show_target_region: bool = layout.get("show_target_region", False)
    suppress_subplot_titles: bool = layout.get("suppress_subplot_titles", False)
    annotate_cluster_sizes: bool = layout.get(
        "annotate_with_cluster_sizes", False
    )

    fig, axes_2d = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    rendered_axes: list[Axes] = []
    rendered_xlims: list[tuple[float, float]] = []
    rendered_ylims: list[tuple[float, float]] = []
    panel_data_list: list[_PanelData] = []

    for panel in panels_spec:
        ax: Axes = axes_2d[panel["row"]][panel["col"]]
        panel_data = _load_panel_data(
            panel,
            sim_runs_dir,
            live,
            layout_show_target_region=show_target_region,
            layout_show_legend=show_legend,
            layout_annotate_cluster_sizes=annotate_cluster_sizes,
        )
        _, _, xlims, ylims = cost_vs_compliance_scatter(
            panel_data.scatter_points,
            x_metric=panel_data.slo_obj.slo_metric,
            x_threshold_objective=panel_data.x_threshold_objective,
            title=panel_data.title,
            ax=ax,
            show_legend=panel_data.show_legend,
            improvement_arrow=panel_data.improvement_arrow,
        )
        _annotate_ax(ax, panel_data, live)
        rendered_axes.append(ax)
        rendered_xlims.append(xlims)
        rendered_ylims.append(ylims)
        panel_data_list.append(panel_data)

    # Apply shared axis limits if requested.
    if shared_xlim and rendered_xlims:
        unified_right = max(lims[1] for lims in rendered_xlims)
        padding = (
            unified_right - min(lims[0] for lims in rendered_xlims)
        ) * 0.05
        for ax in rendered_axes:
            ax.set_xlim(0, unified_right + padding)

    if shared_ylim and rendered_ylims:
        unified_top = max(lims[1] for lims in rendered_ylims)
        padding = (unified_top - min(lims[0] for lims in rendered_ylims)) * 0.05
        for ax in rendered_axes:
            ax.set_ylim(0, unified_top + padding)

    # Capture final axis limits after shared-limit adjustments.
    final_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in rendered_axes]

    # Hide unused axes.
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in seen_positions:
                axes_2d[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {plot_path}")
    _save_points_csv(
        plot_path.with_suffix(".csv"),
        [
            (panel["row"], panel["col"], pd)
            for panel, pd in zip(panels_spec, panel_data_list)
        ],
    )

    if show_legend:
        legend_path = plots_dir / f"{plot_name}_legend.png"
        plot_legend_to(legend_path)
        console.print(f"[green]Saved:[/] {legend_path}")

    # Generate individual panel plots with the same axis limits as the multi-panel.
    multi_stem = plot_path.stem
    for panel_data, (final_xlim, final_ylim) in zip(
        panel_data_list, final_limits
    ):
        export_data = (
            replace(panel_data, title=None)
            if suppress_subplot_titles
            else panel_data
        )
        suffix = re.sub(r"\s+", "_", panel_data.title.lower()).strip("_")
        panel_path = plots_dir / f"{multi_stem}#{suffix}.png"
        _render_and_save_figure(
            export_data, panel_path, live, xlim=final_xlim, ylim=final_ylim
        )


def _generate_plot(
    manifest_path: Path, force: bool, live: bool = False
) -> None:
    manifest = load_yaml(manifest_path)
    plot_name = manifest_path.stem

    data_path = Path(pu.get_data_path())
    sim_runs_dir = data_path / "simulator_runs"
    plots_dir = data_path / "plots" / plot_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{plot_name}#live" if live else plot_name
    plot_path = plots_dir / f"{stem}.png"

    if "panels" in manifest["main_content"]:
        _generate_multi_panel_plot(
            manifest["main_content"],
            plot_path,
            sim_runs_dir,
            plots_dir,
            force,
            live,
        )
    else:
        _generate_single_panel_plot(
            manifest["main_content"],
            plot_path,
            sim_runs_dir,
            force,
            live,
        )


def main() -> None:
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
