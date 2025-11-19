import os
from datetime import datetime, timedelta
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.routing.r_fixed import RFixed
from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_definition.day import Day
from autoslo.workload_execution.trace import Trace


class Composite:

    @staticmethod
    def outputs_parent_dir() -> str:
        """
        Get the directory path where composite workloads are stored.
        """
        return os.path.join(
            pu.get_data_path(),
            "composite_workloads",
        )

    @staticmethod
    def all_composite_workload_names() -> list[str]:
        """
        Get a list of all available composite workload names.

        Returns:
            A list of composite workload names.
        """
        composite_dir = Composite.outputs_parent_dir()
        if not os.path.exists(composite_dir):
            return []
        return [
            name
            for name in os.listdir(composite_dir)
            if os.path.isdir(os.path.join(composite_dir, name))
        ]

    @staticmethod
    def dir_for_composite_workload(workload_name: str) -> str:
        """
        Get the directory path for a specific composite workload.

        Parameters:
            workload_name: The name of the composite workload.

        Returns:
            The directory path for the specified composite workload.
        """
        return os.path.join(Composite.outputs_parent_dir(), workload_name)

    def __init__(
        self,
        name: str,
        days: list[Day],
        monday_index: int = 0,
    ):
        """
        Initialize a new Composite workload.

        Parameters:
            name: A string representing the name of the composite workload.
            days: A list of Days.
            monday_index: An integer representing the index of the first Monday
                within the `days` list.

        Raises:
            ValueError: If monday_index is not between 0 and
                min(7, len(`days`)).
        """
        self.name = name
        self._days = days
        self.monday_index = monday_index

        if not (0 <= self.monday_index < min(len(self.days), 7)):
            raise ValueError(
                "monday_index must be between 0 and min(7, number of days)."
            )

    def save_dir(self) -> str:
        """Get the directory path where the composite workload is saved."""
        return os.path.join(Composite.outputs_parent_dir(), self.name)

    def to_dict(self) -> dict[str, Any]:
        """Get the composite workload representation as a dictionary."""
        return {
            "name": self.name,
            "monday_index": self.monday_index,
            "days": [day.to_dict() for day in self.days],
        }

    @property
    def days(self) -> list[Day]:
        """Get the list of days in the composite workload."""
        return self._days

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Composite":
        """Create a Composite instance from a dictionary representation."""
        days = [Day.from_dict(day_data) for day_data in data["days"]]
        return Composite(
            name=data["name"],
            days=days,
            monday_index=data["monday_index"],
        )

    @staticmethod
    def load(workload_name: str) -> "Composite":
        """
        Load a composite workload from its saved directory.

        Parameters:
            workload_name: The name of the composite workload to load.

        Returns:
            A Composite instance corresponding to the loaded workload.

        Raises:
            FileNotFoundError: If the composite workload definition file does
            not exist.
        """
        definition_path = os.path.join(
            Composite.dir_for_composite_workload(workload_name),
            "definition.yml",
        )
        if not os.path.exists(definition_path):
            raise FileNotFoundError(
                f"Composite definition file '{definition_path}' does not exist."
            )
        with open(definition_path, "r") as f:
            data = yaml.safe_load(f)
        return Composite.from_dict(data)

    def num_days(self) -> int:
        """
        Get the number of days in the composite workload.

        Returns:
            The number of days in the composite workload.
        """
        return len(self.days)

    def save(self):
        """
        Save the composite workload definition, traces and associated plots.
        """
        print("Saving composite workload:", self.name)

        out_dir = self.save_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Save composite workload definition as YAML.
        definition_out_path = os.path.join(out_dir, "definition.yml")
        with open(definition_out_path, "w") as f:
            yaml.dump(self.to_dict(), f)

        # Save plots.
        self.plot_definition(
            save_path=os.path.join(out_dir, f"{self.name}_definition.png")
        )
        self.plot_legend(
            save_path=os.path.join(out_dir, f"{self.name}_legend.png")
        )

    def day_initials(self) -> list[str]:
        """
        Get the initials of the days of the week starting from monday_index.
        """
        day_names = ["M", "T", "W", "T", "F", "S", "S"]
        return [
            f"{day_names[(self.monday_index + i) % 7]}"
            for i in range(len(self.days))
        ]

    DEFAULT_TRACE_START_DATE = datetime(
        year=2025, month=9, day=1, hour=0, minute=0, second=0
    )

    def get_most_recent_trace_on(
        self,
        blueprint_name: str,
        query_router_name: str,
        normalize_start_to: datetime = DEFAULT_TRACE_START_DATE,
        inter_chunk_gap: timedelta = timedelta(0),
    ) -> Trace:
        """
        Get the synthesized trace for the entire composite workload on the
        specified blueprint and query router. The synthesized trace is formed by
        concatenating the traces of all days in the composite workload on the
        specified blueprint and query router.

        All the timestamps in the trace are shifted, so that the first day's
        earliest timestamp is equal to `normalize_start_to`, and the earliest
        timestamp of each subsequent day is exactly 24 hours later than the
        earliest timestamp of the previous day.

        Additionally, an optional gap can be inserted between consecutive chunks
        within each day by specifying `inter_chunk_gap`.

        Parameters:
            endpoint_name: The name of the endpoint.
            normalize_start_to: A datetime object to which the earliest
                timestamp will be normalized.
            inter_chunk_gap: A timedelta object representing the gap to insert
                between consecutive chunks within each day.

        Returns:
            A pandas DataFrame representing the synthesized trace for the
                composite workload.
        """

        l = [
            self.days[0].get_most_recent_trace_on(
                blueprint_name=blueprint_name,
                query_router_name=query_router_name,
                normalize_start_to=normalize_start_to,
                inter_chunk_gap=inter_chunk_gap,
            )
        ]

        for day in self.days[1:]:
            prev_day_start = l[-1].trace_df["start_time"].min()
            this_day_start = prev_day_start + timedelta(days=1)
            day_trace = day.get_most_recent_trace_on(
                blueprint_name=blueprint_name,
                query_router_name=query_router_name,
                normalize_start_to=this_day_start,
                inter_chunk_gap=inter_chunk_gap,
            )
            l.append(day_trace)

        # Concatenate the synthesized trace.
        synthesized_trace = Trace.concat(l)

        return synthesized_trace

    def plot_definition(
        self,
        show: bool = False,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        start_day_idx: int = 0,
    ):
        """
        Plot the composite workload definition.

        Parameters:
            show: If True, display the plot after saving (for interactive use).
            save_path: Optional path to save the plot image. If None, does not
                save the plot image.
            ax: Optional Matplotlib Axes to plot on. If None, creates a new
                figure and axes.
            start_day_idx: The starting index for the days to plot (useful for
                plotting a subset of days).
        """
        eff_num_days = max(7, len(self.days) - start_day_idx)

        if ax is not None:
            fig = ax.figure
        else:
            fig_width = eff_num_days * 0.25 + 0.5
            most_chunks_in_a_day = max(
                len(day.chunks) for day in self.days[start_day_idx:]
            )
            fig_height = (
                max(
                    5,
                    most_chunks_in_a_day,
                )
                * 0.2
                + 1
            )

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot each chunk in its respective day position
        for day_idx, day in enumerate(
            self.days[start_day_idx:], start=start_day_idx
        ):
            for chunk_idx, chunk in enumerate(day.chunks):
                ax.scatter(
                    [day_idx],
                    [chunk_idx],
                    color=chunk.color(),
                    marker=chunk.shape(),
                    s=100,
                    edgecolors="black",
                )

        # Shade the background of the weekends.
        weekend_day_idxs = [
            i
            for i in range(start_day_idx, len(self.days))
            if (self.monday_index + i) % 7 in [5, 6]
        ]
        weekend_starts_ends = []
        for i in range(len(weekend_day_idxs)):
            if (
                i == start_day_idx
                or weekend_day_idxs[i] != weekend_day_idxs[i - 1] + 1
            ):
                weekend_starts_ends.append(
                    [weekend_day_idxs[i], weekend_day_idxs[i]]
                )
            else:
                weekend_starts_ends[-1][1] = weekend_day_idxs[i]

        for start, end in weekend_starts_ends:
            ax.axvspan(start - 0.5, end + 0.5, color="lightgray", alpha=0.5)

        ax.set_xlabel("Day")
        ax.set_xticks(range(start_day_idx, start_day_idx + eff_num_days))
        ax.set_xticklabels(
            self.day_initials()[start_day_idx : start_day_idx + eff_num_days]
            + [""] * (eff_num_days - len(self.days[start_day_idx:]))
        )
        ax.set_yticks([])
        ax.set_title(self.name)
        ax.set_ylim(
            -1, max(len(day.chunks) for day in self.days[start_day_idx:])
        )
        ax.set_xlim(start_day_idx - 0.5, start_day_idx + eff_num_days - 0.5)

        if save_path is not None:
            plt.tight_layout()
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.tight_layout()
            plt.show()

    def plot_legend(self, show: bool = False, save_path: Optional[str] = None):
        """
        Plot a legend for chunk shapes and colors.

        Parameters:
            show: If True, display the plot after saving (for interactive use).
            save_path: Optional path to save the legend image. If None, does
                not save the legend image.
        """
        Chunk.plot_legend(show=show, save_path=save_path)

    def calculate_day_tail_s(
        self,
        day_idx: int,
        blueprint_name: str,
        query_router_name: str,
        tail_percentile: float = 95.0,
    ) -> float:
        """
        Calculate the specified tail percentile of the query durations across
        the specified day on the given blueprint and query router.

        Parameters:
            day_idx: The index of the day within the composite workload.
            blueprint_name: The name of the blueprint.
            query_router_name: The name of the query router.
            tail_percentile: The percentile to calculate (e.g., 95.0 for 95th
                percentile).

        Returns:
            The calculated tail percentile of response times in seconds.
        """
        return (
            self.days[day_idx]
            .get_most_recent_trace_on(
                blueprint_name=blueprint_name,
                query_router_name=query_router_name,
            )
            .latency_s_at(quantile=tail_percentile / 100.0)
        )

    @staticmethod
    def ground_truth_smallest_adherent_single_cluster_blueprint(
        composite_name: str,
        tail_slo_s: float,
        tail_percentile: float = 95.0,
        day_idx: Optional[int] = None,
    ) -> list[Optional[int]]:
        """
        Determine the smallest single-cluster blueprint (i.e. among the blueprints
        with only one cluster, the one where the cluster has the fewest RPUs)
        that meets the specified tail SLO for the specified day of the specified
        composite workload. If day_idx is None, evaluates all days in the
        composite workload.

        Parameters:
            composite_name: The name of the composite workload.
            tail_slo_s: The tail SLO in seconds.
            tail_percentile: The percentile to consider for the SLO (e.g., 95.0
                for 95th percentile).
            day_idx: The index of the day within the composite workload.
                If None, evaluates all days in the composite workload.

        Returns:
            The smallest single-cluster blueprint that meets the tail SLO, or
            None if no suitable blueprint is found, for each day (if day_idx is
            None) or for the specified day (if day_idx is provided).
        """

        composite = Composite.load(composite_name)
        days_to_evaluate = (
            [day_idx] if day_idx is not None else range(len(composite.days))
        )

        sizes: list[Optional[int]] = []
        for idx in days_to_evaluate:
            selected = None
            for rpu in sorted(Cluster.ALL_ALLOWED_RPU_SIZES):
                # Retrieve the trace on a single-cluster blueprint with the
                # current RPU size.
                blueprint = Blueprint.one_cluster_with(cluster_rpu=rpu)
                trace = composite.days[idx].get_most_recent_trace_on(
                    blueprint_name=blueprint.name,
                    query_router_name=RFixed(
                        blueprint, fixed_cluster_name=blueprint.cluster_names[0]
                    ).name,
                )

                # Check if the tail latency meets the SLO.
                if (
                    trace.latency_s_at(
                        quantile=tail_percentile / 100.0,
                    )
                    <= tail_slo_s
                ):
                    selected = rpu
                    break

            sizes.append(selected)

        return sizes
