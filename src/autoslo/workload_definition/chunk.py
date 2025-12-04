import os
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import autoslo.utils.colors as cu
import autoslo.utils.paths as pu
from autoslo.workload_execution.trace import Trace


class Chunk:
    """
    A chunk represents a specific workload characterized by H and T values.
    """

    H_SHAPE_MAP = {
        0: "o",
        10: "H",
        25: "s",
        50: "^",
        # 75: "*",
    }
    T_COLOR_MAP = {
        120: cu.Palette.light_green,
        60: cu.Palette.light_blue,
        30: cu.Palette.light_yellow,
        10: cu.Palette.light_orange,
        # 5: cu.Palette.light_red,
        # 1: cu.Palette.gray,
    }

    SUPPORTED_SCHEMAS = ["tpcds"]

    DEFAULT_SCHEMA = "tpcds"
    DEFAULT_CHUNK_DURATION_S = 3600  # 1 hour
    DEFAULT_RANDOM_SEED = 42
    DEFAULT_NUM_TEMPLATES = 99
    DEFAULT_NUM_QUERIES_PER_TEMPLATE = 3
    DEFAULT_STDDEV_INTERARRIVAL_S = None

    def __init__(
        self,
        H: int,
        T: int,
        schema: str = DEFAULT_SCHEMA,
        chunk_duration_s: int = DEFAULT_CHUNK_DURATION_S,
        random_seed: int = DEFAULT_RANDOM_SEED,
        num_templates: int = DEFAULT_NUM_TEMPLATES,
        num_queries_per_template: int = DEFAULT_NUM_QUERIES_PER_TEMPLATE,
        stddev_interarrival_s: Optional[int] = DEFAULT_STDDEV_INTERARRIVAL_S,
    ):
        """
        Initialize a new chunk.

        Parameters:
            H: The percentage of heavy queries (0-100).
            T: The mean query interarrival time in seconds.
            schema: The database schema to use.
            chunk_duration_s: The duration of the chunk in seconds.
            random_seed: The random seed for workload generation.
            num_templates: The maximum number of query templates to use.
            num_queries_per_template: The number of queries per template.
            stddev_interarrival_s: The standard deviation of interarrival times
                in seconds. If None, defaults to T / 2.

        Raises:
            ValueError: If the arguments are out of valid ranges.
        """
        if not (0 <= H <= 100):
            raise ValueError("H must be between 0 and 100.")
        if T <= 0:
            raise ValueError("T must be a positive number.")
        if schema.lower() not in self.SUPPORTED_SCHEMAS:
            raise ValueError(
                f"Schema '{schema}' is not supported. Supported schemas: "
                f"{self.SUPPORTED_SCHEMAS}."
            )
        if chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be a positive integer.")
        if num_templates <= 0:
            raise ValueError("num_templates must be a positive integer.")
        if num_queries_per_template <= 0:
            raise ValueError(
                "num_queries_per_template must be a positive integer."
            )
        if stddev_interarrival_s is not None and stddev_interarrival_s <= 0:
            raise ValueError(
                "stddev_interarrival_s must be a positive integer if provided."
            )

        self.H = int(H)
        self.T = int(T)
        self.schema = schema.lower()
        self.chunk_duration_s = chunk_duration_s
        self.random_seed = random_seed
        self.num_templates = num_templates
        self.num_queries_per_template = num_queries_per_template
        self.stddev_interarrival_s = (
            stddev_interarrival_s
            if stddev_interarrival_s is not None
            else T / 2
        )
        self._chunk_id = self.form_chunk_id(
            schema_name=self.schema,
            num_templates=self.num_templates,
            H=self.H,
            T=self.T,
        )

    def chunk_id(self) -> str:
        """Get the chunk ID string."""
        return self._chunk_id

    @staticmethod
    def form_chunk_id(
        schema_name: str, num_templates: int, H: int, T: int
    ) -> str:
        """
        Get the chunk ID string for the given parameters.

        Parameters:
            schema_name: The name of the database schema.
            num_templates: The number of query templates.
            H: The percentage of heavy queries (0-100).
            T: The mean query interarrival time in seconds.

        Returns:
            The chunk ID string.
        """
        return "_".join(
            [
                schema_name,
                f"{num_templates}templates",
                f"{H:02d}pctheavy",
                f"{T:02d}meaninterarrivals",
            ]
        )

    def save_dir(self) -> str:
        """Get the directory path where the chunk workload is saved."""
        return os.path.join(
            pu.get_data_path(),
            "chunk_workloads",
            self.chunk_id(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Get the chunk parameters as a dictionary."""
        return {
            "H": self.H,
            "T": self.T,
            "schema": self.schema,
            "chunk_duration_s": self.chunk_duration_s,
            "random_seed": self.random_seed,
            "num_templates": self.num_templates,
            "num_queries_per_template": self.num_queries_per_template,
            "stddev_interarrival_s": self.stddev_interarrival_s,
        }

    @staticmethod
    def from_dict(params: dict[str, Any]) -> "Chunk":
        """Create a Chunk instance from a parameters dictionary."""
        return Chunk(
            H=params["H"],
            T=params["T"],
            schema=params.get("schema", Chunk.DEFAULT_SCHEMA),
            chunk_duration_s=params.get(
                "chunk_duration_s", Chunk.DEFAULT_CHUNK_DURATION_S
            ),
            random_seed=params.get("random_seed", Chunk.DEFAULT_RANDOM_SEED),
            num_templates=params.get(
                "num_templates", Chunk.DEFAULT_NUM_TEMPLATES
            ),
            num_queries_per_template=params.get(
                "num_queries_per_template",
                Chunk.DEFAULT_NUM_QUERIES_PER_TEMPLATE,
            ),
            stddev_interarrival_s=params.get(
                "stddev_interarrival_s", Chunk.DEFAULT_STDDEV_INTERARRIVAL_S
            ),
        )

    def save(self):
        """Save the chunk definition as a YAML file."""
        out_dir = self.save_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "chunk_definition.yml")
        with open(out_path, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    def color(self) -> str:
        """Get the color associated with the chunk's T value."""
        for t_threshold, color in sorted(
            self.T_COLOR_MAP.items(), reverse=True
        ):
            if self.T >= t_threshold:
                return color
        return "black"  # Default color if no thresholds match

    def shape(self) -> str:
        """Get the shape associated with the chunk's H value."""
        for h_threshold, shape in sorted(
            self.H_SHAPE_MAP.items(), reverse=True
        ):
            if self.H >= h_threshold:
                return shape
        return "o"  # Default shape if no thresholds match

    def synthesize_chunk_workload(
        self,
    ) -> None:
        # Retrieve the query texts for the specified number of templates.
        query_texts: dict[int, list[str]] = {}
        for template_id in range(1, self.num_templates + 1):
            template_str = f"query{template_id:03d}"
            template_dir = os.path.join(pu.QUERIES_PATH, template_str)
            if not os.path.exists(template_dir):
                print(f"Template directory {template_dir} missing. Skipping.")
                continue

            query_texts[template_id] = []

            for query_num in range(1, self.num_queries_per_template + 1):
                with open(
                    os.path.join(
                        template_dir, f"{template_str}_{query_num:03d}.sql"
                    ),
                    "r",
                ) as f:
                    query_text = f.read()
                query_texts[template_id].append(query_text)

        # Determine which templates are heavy and which are light.
        heavy_templates = set()
        with open(pu.get_heavy_templates_files()[self.schema], "r") as f:
            for line in f:
                template_id = int(line.strip())
                if template_id <= self.num_templates:
                    heavy_templates.add(template_id)
        light_templates = (
            set(range(1, self.num_templates + 1)) - heavy_templates
        )

        # Generate the chunk trace.
        chunk_trace = []
        current_time_s = 0.0
        np.random.seed(self.random_seed)
        query_id = 0
        while current_time_s < self.chunk_duration_s:
            # Create record for current query.
            pick_heavy = (np.random.rand() * 100) < self.H
            if pick_heavy:
                template_id = np.random.choice(list(heavy_templates))
            else:
                template_id = np.random.choice(list(light_templates))
            query_num_within_template = np.random.randint(
                0, self.num_queries_per_template
            )
            query_text = query_texts[template_id][query_num_within_template]

            chunk_trace.append(
                {
                    "chunk_id": self.chunk_id(),
                    "query_id": query_id,
                    "rel_start_time_s": current_time_s,
                    "query_template": template_id,
                    "query_num_within_template": query_num_within_template,
                    "query_text": query_text,
                }
            )
            query_id += 1

            # Update current time.
            interarrival_time_s = max(
                0.0,
                np.random.normal(
                    loc=self.T,
                    scale=self.stddev_interarrival_s,
                ),
            )
            current_time_s += interarrival_time_s

        # Write out the chunk_workload as a Parquet file.
        df = pd.DataFrame(chunk_trace)
        out_path = os.path.join(
            self.save_dir(),
            f"chunk_workload.parquet",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path, index=False)

        # Save sanity check statistics from the file we just wrote out and return.
        stats = {
            "chunk_id": self.chunk_id(),
            "chunk_duration_s": self.chunk_duration_s,
            "expected__num_queries": int(self.chunk_duration_s / self.T),
            "actual__num_queries": df.shape[0],
            "expected__num_templates": self.num_templates,
            "actual__num_templates": df["query_template"].nunique(),
            "expected__num_queries_per_template": self.num_queries_per_template,
            "actual__num_queries_per_template": int(
                df.groupby("query_template")["query_num_within_template"]
                .nunique()
                .max()
            ),
            "expected__pct_heavy": self.H,
            "actual__pct_heavy": (
                100
                * df[df["query_template"].isin(heavy_templates)].shape[0]
                / df.shape[0]
            ),
            "expected__mean_interarrival_time_s": self.T,
            "actual__mean_interarrival_time_s": float(
                df["rel_start_time_s"].diff().mean()
            ),
            "expected__stddev_interarrival_time_s": self.stddev_interarrival_s,
            "actual__stddev_interarrival_time_s": float(
                df["rel_start_time_s"].diff().std()
            ),
        }
        with open(
            os.path.join(self.save_dir(), "chunk_workload_stats.yml"),
            "w",
        ) as f:
            yaml.dump(stats, f, sort_keys=False)

    def get_most_recent_run_id_on(
        self, blueprint_name: str, query_router_name: str
    ) -> str:
        """
        Get the most recent run ID for this chunk on the specified blueprint
        and query router.

        Parameters:
            blueprint_name: The name of the blueprint.
            query_router_name: The name of the query router.

        Returns:
            A string representing the run ID.

        Raises:
            ValueError: If no matching run ID is found.
        """
        run_ids = pu.RunLocator.get_run_ids(
            workload_name=self.chunk_id(),
            blueprint_name=blueprint_name,
            query_router_name=query_router_name,
        )
        if len(run_ids) == 0:
            raise ValueError(
                f"No run ID found for chunk with H={self.H}, T={self.T} on "
                f"blueprint '{blueprint_name}' and "
                f"query router '{query_router_name}'."
            )
        return sorted(run_ids)[-1]

    def get_most_recent_trace_on(
        self,
        blueprint_name: str,
        query_router_name: str,
        normalize_start_to: Optional[datetime] = None
    ) -> Trace:
        """
        Get the (most recent) trace for this chunk on the specified blueprint
        and query router, optionally shifting all timestamps so that the
        earliest timestamp is equal to `normalize_start_to`.

        Parameters:
            blueprint_name: The name of the blueprint.
            query_router_name: The name of the query router.
            normalize_start_to: A datetime object to which the earliest
                timestamp will be normalized.

        Returns:
            A Trace instance.
        """
        most_recent_run_id = self.get_most_recent_run_id_on(
            blueprint_name=blueprint_name,
            query_router_name=query_router_name,
        )
        trace = Trace(run_id=most_recent_run_id)

        if normalize_start_to is not None:
            trace.normalize_start_to(normalize_start_to)

        return trace

    @staticmethod
    def plot_legend(show: bool = False, save_path: Optional[str] = None):
        """
        Plot a legend for chunk shapes and colors.

        Parameters:
            show: If True, display the plot after saving (for interactive use).
            save_path: Optional path to save the legend image. If None, do not
                save the image.
        """
        _, ax = plt.subplots(figsize=(3.5, 2))
        ax.axis("off")

        shape_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker=shape,
                color="w",
                label=f"H >= {h_threshold}%",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
            )
            for h_threshold, shape in Chunk.H_SHAPE_MAP.items()
        ]
        color_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"T >= {t_threshold}s",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="black",
            )
            for t_threshold, color in Chunk.T_COLOR_MAP.items()
        ]

        first_legend = ax.legend(
            handles=shape_legend_elements,
            title="Shapes (H values)",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.5),
            columnspacing=1,
            handletextpad=0.05,
            ncols=len(Chunk.H_SHAPE_MAP),
        )
        ax.add_artist(first_legend)
        ax.legend(
            handles=color_legend_elements,
            title="Colors (T values)",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.5),
            columnspacing=1,
            handletextpad=0.05,
            ncols=len(Chunk.T_COLOR_MAP),
        )
        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
