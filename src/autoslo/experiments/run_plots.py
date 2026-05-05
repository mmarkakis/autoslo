"""Plot simulator evaluation results from per-experiment plot specs.

Each experiment subdirectory contains a ``plot_spec.yml`` that describes a
single plot: its title and a list of panels, each with a list of data series.
Each series identifies a completed simulation run and a visual formatting style.

Within each panel all series must share the same SLO objective and SLO resolver
configuration — a mismatch raises an error rather than silently mixing
incomparable axes.

Output is written to ``<experiment_dir>/plots/``.

Usage (from repo root)::

    # Discover all sub-specs under a directory (one level deep):
    python src/autoslo/experiments/run_plots.py experiments/9991_main_experiments/

    # Run a single spec directly:
    python src/autoslo/experiments/run_plots.py experiments/9991_main_experiments/observation_period/plot_spec.yml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import (
    SloObjectiveConfig,
    SloResolverConfig,
)
from autoslo.filesystem.config_resolver import (
    expand_series_entries,
    resolve_series_exec_config_id,
)
from autoslo.filesystem.path_utils import is_up_to_date
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.simulator.simulation_result import SimulationResult
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.visualizations.scatter_plots import (
    ScatterPoint,
    cost_vs_compliance_scatter,
    plot_legend_to,
)

console = Console()


def _plot_is_up_to_date(
    spec: dict, spec_path: Path, plot_path: Path, sim_runs_dir: Path
) -> bool:
    """Return True iff *plot_path* is up to date with respect to its inputs.

    Inputs considered:
    - The plot_spec.yml itself.
    - Every ``execution_config.yml`` in the simulator_runs directories
      referenced by the spec's series entries (resolved via
      :func:`resolve_series_exec_config_id`).
    """
    root = Path(pu.AUTOSLO_ROOT)
    extra_inputs = [spec_path]
    for panel in spec.get("panels", []):
        for entry in expand_series_entries(panel.get("series", [])):
            wid = entry.get("workload_id", "")
            eid = resolve_series_exec_config_id(entry, root)
            if wid and eid:
                extra_inputs.append(
                    sim_runs_dir / wid / eid / "execution_config.yml"
                )
    return is_up_to_date(plot_path, *extra_inputs)


def _x_value(result: SimulationResult, metric: SloMetric) -> float:
    """Extract the x-axis value from a SimulationResult for the given metric."""
    if metric is SloMetric.BINARY:
        return result.violation_rate
    if metric is SloMetric.ABSOLUTE_S:
        return result.violation_amount_s
    if metric in (SloMetric.RELATIVE, SloMetric.RELATIVE_UNCONSTRAINED):
        return result.violation_relative_mean
    raise ValueError(f"Unsupported SloMetric for plotting: {metric}")


def _slo_fingerprint(
    exec_config: dict, slo_obj: SloObjective
) -> tuple[SloObjective, SloResolverConfig]:
    """Return a hashable fingerprint of the SLO objective and resolver config.

    Uses the parsed, typed dataclass objects rather than raw dict repr so that
    equivalent configs with different key ordering or null values compare equal.
    For example, round-robin and single-cluster configs share the same SLO
    settings and will trivially produce the same fingerprint.
    """
    return (slo_obj, SloResolverConfig.from_config(exec_config))


def _render_spec(
    spec_path: Path, sim_runs_dir: Path, force: bool = False
) -> None:
    """Render one plot_spec.yml to PNGs in its sibling plots/ directory."""
    spec = load_yaml(spec_path)
    spec_dir = spec_path.parent
    plot_name: str = spec["plot_name"]
    plot_title: str = spec.get("title", plot_name)
    panels: list[dict] = spec["panels"]
    num_panels = len(panels)

    plot_path = spec_dir / f"{plot_name}.png"
    if not force and _plot_is_up_to_date(
        spec, spec_path, plot_path, sim_runs_dir
    ):
        console.print(f"[dim]Skipping '{plot_name}' (up to date)[/]")
        return

    if num_panels == 0:
        console.print(
            f"[yellow]Warning: '{plot_name}' has no panels — skipping.[/]"
        )
        return

    spec_dir = spec_dir

    # Accumulate points and SLO objects per panel.
    panel_points: dict[int, list[ScatterPoint]] = {
        i: [] for i in range(num_panels)
    }
    panel_slo_fingerprint: dict[int, tuple] = {}
    panel_slo_obj: dict[int, SloObjective] = {}

    for panel_id, panel_def in enumerate(panels):
        for series_entry in expand_series_entries(panel_def["series"]):
            workload_id: str = series_entry["workload_id"]
            exec_config_id: str | None = resolve_series_exec_config_id(
                series_entry, Path(pu.AUTOSLO_ROOT)
            )
            if exec_config_id is None:
                console.print(
                    f"[yellow]Warning: could not resolve exec_config_id for "
                    f"series entry in panel {panel_id} of '{plot_name}' "
                    f"— skipping this point.[/]"
                )
                continue
            label: str = series_entry["label"]
            formatting_id: str = series_entry["formatting_id"]

            run_dir = sim_runs_dir / workload_id / exec_config_id
            config_path = run_dir / "execution_config.yml"

            if not config_path.exists():
                console.print(
                    f"[yellow]Warning: simulation run not found for "
                    f"workload='{workload_id}', exec_config='{exec_config_id}' "
                    f"— skipping this point.[/]"
                )
                continue

            result = SimulationResult.load(run_dir)
            exec_config = load_yaml(config_path)
            slo_obj = SloObjective(SloObjectiveConfig.from_config(exec_config))
            x_val = _x_value(result, slo_obj.slo_metric)
            point = ScatterPoint(
                formatting_id=formatting_id,
                x=x_val,
                y=result.total_cost,
            )

            # Round-robin configs are fixed-cluster baselines whose cost and
            # violation are meaningful on any SLO scale, so they are exempt
            # from the panel SLO consistency check. Only non-round-robin
            # configs must all share the same SLO objective and resolver.
            routing = exec_config.get("query_router_config", {}).get(
                "routing_policy_name", ""
            )
            if routing != "round_robin":
                fingerprint = _slo_fingerprint(exec_config, slo_obj)
                if panel_id in panel_slo_fingerprint:
                    if panel_slo_fingerprint[panel_id] != fingerprint:
                        raise ValueError(
                            f"Panel {panel_id} of plot '{plot_name}' has "
                            f"inconsistent SLO objective or resolver config "
                            f"across non-round-robin series entries. All such "
                            f"points within a panel must share the same SLO "
                            f"configuration."
                        )
                else:
                    panel_slo_fingerprint[panel_id] = fingerprint
                    panel_slo_obj[panel_id] = slo_obj

            panel_points[panel_id].append(point)

    # Build figure with num_panels subplots.
    num_cols = min(num_panels, 2)
    num_rows = (num_panels + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6 * num_cols, 5 * num_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None

    for panel_id in range(num_panels):
        row = panel_id // num_cols
        col = panel_id % num_cols
        ax = axes[row][col]
        points = panel_points[panel_id]

        if not points:
            fig.delaxes(ax)
            continue

        panel_def = panels[panel_id]
        if num_panels == 1:
            panel_title = plot_title
        elif "title" in panel_def:
            panel_title = panel_def["title"]
        else:
            panel_title = f"{plot_title} — Panel {panel_id}"

        slo_obj = panel_slo_obj.get(panel_id)
        _, _, xlims, ylims = cost_vs_compliance_scatter(
            points,
            x_metric=slo_obj.slo_metric if slo_obj else SloMetric.BINARY,
            x_threshold_objective=slo_obj,
            existing_xlims=xlims,
            existing_ylims=ylims,
            title=panel_title,
            ax=ax,
        )

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/] {plot_path}")

    legend_path = spec_dir / "legend.png"
    plot_legend_to(legend_path)
    console.print(f"[green]Saved:[/] {legend_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot simulator evaluation results from per-experiment plot specs."
    )
    parser.add_argument(
        "path",
        help=(
            "Path to a plot_spec.yml file, or a directory whose immediate "
            "subdirectories are searched for plot_spec.yml files."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render all plots unconditionally, ignoring staleness.",
    )
    args = parser.parse_args()

    p = Path(args.path)
    target = p if p.is_absolute() else Path(pu.AUTOSLO_ROOT) / p

    if target.is_dir():
        spec_paths = sorted(target.glob("*/plot_spec.yml"))
        if not spec_paths:
            parser.error(f"No plot_spec.yml files found under: {target}")
    elif target.is_file():
        spec_paths = [target]
    else:
        parser.error(f"Path not found: {target}")

    sim_runs_dir = Path(pu.get_data_path()) / "simulator_runs"

    for spec_path in spec_paths:
        console.print(
            f"\n[cyan]Processing:[/] {spec_path.relative_to(pu.AUTOSLO_ROOT)}"
        )
        _render_spec(spec_path, sim_runs_dir, force=args.force)

    console.print(f"\n[bold green]Done.[/]")


if __name__ == "__main__":
    main()
