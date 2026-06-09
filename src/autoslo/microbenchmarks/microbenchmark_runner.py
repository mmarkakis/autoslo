from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import ClusterState, ClusterView
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.config.component_configs import (
    ManagedClusterPoolConfig,
    ProvisionerConfig,
    WorkloadConfig,
)
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter
from autoslo.routing.wrapper import (
    NoOpAutoscaler,
    _AutoscalerLike,
    route_and_update_bookkeeping,
)
from autoslo.visualizations.colors import Palette
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.workload import Workload


class MicrobenchmarkRunner:

    #############
    # Path-related.
    #############

    @classmethod
    def manifest_path(cls) -> Path:
        return (
            Path(pu.get_data_path())
            / "manifests"
            / "microbench"
            / f"{cls.name()}.yml"
        )

    @classmethod
    def scratch_dir(cls) -> Path:
        return Path(pu.get_data_path()) / "microbenchmark_runs" / cls.name()

    @classmethod
    def csv_path(cls) -> Path:
        return Path(pu.get_data_path()) / "plot_data" / f"{cls.name()}.csv"

    @classmethod
    def plot_path(cls) -> Path:
        return Path(pu.get_data_path()) / "plots" / f"{cls.name()}.png"

    #############
    # To be implemented by subclasses.
    #############

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError("Must be implemented by subclasses.")

    @classmethod
    def required_keys(cls) -> list[str]:
        raise NotImplementedError("Must be implemented by subclasses.")

    @classmethod
    def run_from_manifest(cls, manifest: dict) -> None:
        raise NotImplementedError("Must be implemented by subclasses.")

    @classmethod
    def plot(cls) -> None:
        raise NotImplementedError("Must be implemented by subclasses.")

    ############
    # Other.
    ############

    TPCDS_99_TEMPLATE_SOURCE_WORKLOAD = "poisson_99_1_3_0.2_42"
    TPCDS_TEMPLATE_QUERY_INDEX = "001"
    TPCDS_TEMPLATE_COUNT = 99
    SCATTER_FIGSIZE = (6.5, 5)
    SCATTER_DPI = 180
    SCATTER_MARKER_SIZE = 200
    SCATTER_POINT_ALPHA = 0.9
    SCATTER_EDGE_LINEWIDTH = 0.4
    SCATTER_LEGEND_MARKER_SIZE = 12
    SCATTER_LEGEND_MARKER_EDGE_WIDTH = 1.0
    SCATTER_SHOW_MINOR_GRID = False
    SCATTER_MAJOR_GRID_ALPHA = 0.6
    SCATTER_MINOR_GRID_ALPHA = 0.35
    SCATTER_COLORBAR_MARKER_LINEWIDTH = 0.8
    SCATTER_COLORBAR_MARKER_ALPHA = 0.35
    BASE_FONT_SIZE = 22
    SCATTER_AXIS_LABEL_FONT_SIZE = BASE_FONT_SIZE
    SCATTER_TICK_FONT_SIZE = BASE_FONT_SIZE
    SCATTER_LEGEND_TITLE_FONT_SIZE = BASE_FONT_SIZE
    SCATTER_LEGEND_TEXT_FONT_SIZE = BASE_FONT_SIZE
    SCATTER_COLORBAR_LABEL_FONT_SIZE = BASE_FONT_SIZE
    SCATTER_COLORBAR_TICK_FONT_SIZE = BASE_FONT_SIZE
    LEGEND_LABEL_SPACING = 0.1

    DEFAULT_PROVISIONER_CONFIG_ARGS = {
        "aws_config_path": "",
        "cluster_cache_state_dim": 20,
        "run_id": "microbenchmark_run",
        "spin_up_delay_s": 300.0,
    }

    @staticmethod
    def round_up_to_next_multiple_of_base(value: int, base: int) -> int:
        """Round up value to the next multiple of the given number."""
        if base <= 0:
            raise ValueError("base must be positive")
        return int(((value + base - 1) // base) * base)

    @staticmethod
    def make_progress() -> Progress:
        """Return a pre-configured Rich Progress context manager."""
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )

    @classmethod
    def load_uniform_tpcds_template_001_pool(
        cls,
        *,
        model: IconqModel,
        allowed_rpu_sizes: list[int],
        n_queries: int,
        template_selection_seed: int,
    ) -> list[Query]:
        """
        Create a pool of queries by subsetting the 001 queries per template of
        the provided workload, drawing uniformly, and copying them with new IDs
        and start times.
        """
        if n_queries <= 0:
            raise ValueError("n_queries must be positive")

        workload = Workload(
            WorkloadConfig(
                workload_name=cls.TPCDS_99_TEMPLATE_SOURCE_WORKLOAD,
            )
        )
        workload.populate_featurizations_and_isolated_predictions(
            iconq_model=model,
            allowed_rpu_sizes=allowed_rpu_sizes,
        )

        qs_by_template: dict[str, Query] = {}
        for query in workload.queries():
            template_id = query.query_text_id.template_id
            query_index = query.query_text_id.query_index
            if query_index != cls.TPCDS_TEMPLATE_QUERY_INDEX:
                continue
            if template_id not in qs_by_template:
                qs_by_template[template_id] = query

        if len(qs_by_template) < cls.TPCDS_TEMPLATE_COUNT:
            raise ValueError(
                "TPC-DS template sampling requires 99 template#001 queries, "
                f"but found {len(qs_by_template)} in "
                f"{cls.TPCDS_99_TEMPLATE_SOURCE_WORKLOAD}."
            )

        template_ids = sorted(qs_by_template.keys())
        rng = np.random.default_rng(template_selection_seed)

        pool: list[Query] = []
        temp_idxs = rng.integers(0, len(template_ids), size=n_queries)
        for idx in range(n_queries):
            q = qs_by_template[template_ids[temp_idxs[idx]]]
            q_ = q.copy_with_new_info(
                new_query_id_prefix=f"pool_{idx:05d}_",
                new_rel_start_time_s=float(idx),
            )
            pool.append(q_)
        return pool

    @classmethod
    def ingest_initial(
        cls,
        *,
        workload: Workload,
        n_to_ingest: int,
        initial_cluster_sizes: list[int],
        query_router: QueryRouter,
        autoscaler: Optional[_AutoscalerLike] = None,
    ) -> dict[str, ClusterView]:
        """
        Ingest queries into a number of clusters, using the given router.
        """

        if len(initial_cluster_sizes) == 0:
            raise ValueError("initial_cluster_sizes must be non-empty")

        if autoscaler is None:
            autoscaler = NoOpAutoscaler()

        # Initialize cluster pool without any queries.
        cache_state_dim = (
            query_router._iconq_model.iconq_query_featurizer.num_tables
        )
        mcp = ManagedClusterPool(
            provisioner=SimulatedProvisioner(
                ProvisionerConfig(
                    **cls.DEFAULT_PROVISIONER_CONFIG_ARGS
                    | {"cluster_cache_state_dim": cache_state_dim},
                )
            ),
            config=ManagedClusterPoolConfig(
                initial_rpus=initial_cluster_sizes,
                max_clusters=len(initial_cluster_sizes),
            ),
        )
        mcp.add_details_and_spin_up_initial_clusters()
        for name in mcp.clusters_in_state(ClusterState.PENDING):
            mcp.on_cluster_ready(name, 0.0)

        # Ingest queries.
        for query in workload.queries()[:n_to_ingest]:
            route_and_update_bookkeeping(
                source="ingest_initial",
                rel_time_s_getter=lambda: query.rel_start_time_s,
                pool=mcp,
                router=query_router,
                query=query,
                autoscaler=autoscaler,
            )

        return mcp.snapshot(only_ready=True)

    @classmethod
    def microbenchmark_scatter_plot(
        cls,
        *,
        x_col: str,
        y_col: str,
        shape_col: str,
        color_col: str,
        shape_legend_title: str,
        colorbar_label: str,
        cmap_colors: list[str],
        log_x: bool = False,
        log_y: bool = False,
        log_color_base: Optional[int] = None,
    ) -> None:
        df = pd.read_csv(cls.csv_path())
        fig, ax = plt.subplots(figsize=cls.SCATTER_FIGSIZE)
        fig.patch.set_facecolor(Palette.white)
        ax.set_facecolor(Palette.white)

        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        shape_values = sorted(df[shape_col].unique().tolist())
        color_values = df[color_col].astype(float)
        cmap_name = f"{cls.name()}_cmap"
        cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colors)
        if log_color_base is not None:
            vmin = max(float(color_values.min()), 1e-12)
            vmax = float(color_values.max())
            norm: Normalize = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = float(color_values.min())
            vmax = float(color_values.max())
            norm = Normalize(vmin=vmin, vmax=vmax)

        for idx, shape_value in enumerate(shape_values):
            marker = markers[idx % len(markers)]
            subset = df[df[shape_col] == shape_value]
            ax.scatter(
                subset[x_col],
                subset[y_col],
                c=subset[color_col],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=cls.SCATTER_MARKER_SIZE,
                edgecolors=Palette.gray,
                linewidths=cls.SCATTER_EDGE_LINEWIDTH,
                alpha=cls.SCATTER_POINT_ALPHA,
            )

        shape_handles = [
            Line2D(
                [0],
                [0],
                marker=markers[idx % len(markers)],
                color="none",
                markerfacecolor=Palette.white,
                markeredgecolor=Palette.black,
                markeredgewidth=cls.SCATTER_LEGEND_MARKER_EDGE_WIDTH,
                markersize=cls.SCATTER_LEGEND_MARKER_SIZE,
                label=shape_value,
            )
            for idx, shape_value in enumerate(shape_values)
        ]
        shape_legend = ax.legend(
            handles=shape_handles,
            title=shape_legend_title,
            loc="upper left",
            facecolor=Palette.white,
            edgecolor=Palette.gray,
            title_fontsize=cls.SCATTER_LEGEND_TITLE_FONT_SIZE,
            fontsize=cls.SCATTER_LEGEND_TEXT_FONT_SIZE,
            labelspacing=cls.LEGEND_LABEL_SPACING,
        )
        shape_legend.get_title().set_color(Palette.black)
        shape_legend.get_title().set_ha("center")
        shape_legend._legend_box.align = "center"
        for text in shape_legend.get_texts():
            text.set_color(Palette.black)
        ax.add_artist(shape_legend)

        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        colorbar.set_label(
            colorbar_label,
            color=Palette.black,
            fontsize=cls.SCATTER_COLORBAR_LABEL_FONT_SIZE,
        )
        plotted_color_values = sorted(color_values.unique().tolist())
        if len(plotted_color_values) <= 15:
            colorbar.set_ticks(plotted_color_values)
            tick_labels = [f"{v:.2f}" for v in plotted_color_values]
            if log_color_base is not None:
                formatter = LogFormatterSciNotation(base=log_color_base)
                tick_labels = [formatter(v) for v in plotted_color_values]
            colorbar.set_ticklabels(tick_labels)

        for value in plotted_color_values:
            colorbar.ax.hlines(
                y=value,
                xmin=0.0,
                xmax=1.0,
                colors=Palette.black,
                linewidth=cls.SCATTER_COLORBAR_MARKER_LINEWIDTH,
                alpha=cls.SCATTER_COLORBAR_MARKER_ALPHA,
            )
        colorbar.ax.tick_params(
            colors=Palette.black,
            labelsize=cls.SCATTER_COLORBAR_TICK_FONT_SIZE,
        )

        ax.set_xlabel(
            "Total Queries",
            color=Palette.black,
            fontsize=cls.SCATTER_AXIS_LABEL_FONT_SIZE,
        )
        ax.set_ylabel(
            "Time (s)",
            color=Palette.black,
            fontsize=cls.SCATTER_AXIS_LABEL_FONT_SIZE,
        )
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        else:
            prev_y_lims = ax.get_ylim()
            ax.set_ylim(bottom=min(prev_y_lims[0], 0))
        ax.grid(
            True,
            which="major",
            color=Palette.light_gray,
            alpha=cls.SCATTER_MAJOR_GRID_ALPHA,
        )
        if cls.SCATTER_SHOW_MINOR_GRID:
            ax.minorticks_on()
            ax.grid(
                True,
                which="minor",
                color=Palette.light_gray,
                alpha=cls.SCATTER_MINOR_GRID_ALPHA,
            )
        ax.tick_params(
            colors=Palette.black,
            which="major",
            labelsize=cls.SCATTER_TICK_FONT_SIZE,
        )
        ax.tick_params(
            which="minor",
            labelbottom=False,
            labelleft=False,
        )

        fig.tight_layout()
        fig.savefig(cls.plot_path(), dpi=cls.SCATTER_DPI)
        plt.close(fig)
