import os

import matplotlib.pyplot as plt

import chunkbench.path_utils as pu
from chunkbench.building_blocks.chunk import Chunk
from chunkbench.building_blocks.day import Day

from typing import Optional, Any
from datetime import datetime, timedelta

import pandas as pd

import yaml

from tqdm.auto import tqdm


class Composite:

    def __init__(
        self,
        name: str,
        days: list[Day],
        monday_index: int = 0,
    ):
        """
        Initialize a new MultiDay.

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
        self.days = days
        self.monday_index = monday_index

        if not (0 <= self.monday_index < min(len(self.days), 7)):
            raise ValueError(
                "monday_index must be between 0 and min(7, number of days)."
            )

    def save_dir(self) -> str:
        """Get the directory path where the composite workload is saved."""
        return os.path.join(pu.DATA_PATH, "composite_workloads", self.name)

    def to_dict(self) -> dict[str, Any]:
        """Get the composite workload representation as a dictionary."""
        return {
            "name": self.name,
            "monday_index": self.monday_index,
            "days": [day.to_dict() for day in self.days],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Composite":
        """Create a Composite instance from a dictionary representation."""
        days = [Day.from_dict(day_data) for day_data in data["days"]]
        return Composite(
            name=data["name"],
            days=days,
            monday_index=data["monday_index"],
        )

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

        # For each day, create and save its trace on each endpoint.
        for day_idx, day in tqdm(
            enumerate(self.days),
            desc="Saving day traces...",
            total=len(self.days),
        ):
            day_dir = os.path.join(out_dir, "day_traces", f"day_{day_idx}")
            os.makedirs(day_dir, exist_ok=True)
            for endpoint_name in ["4", "8", "16", "32"]:
                trace_out_path = os.path.join(
                    day_dir, f"{self.name}_day{day_idx}_{endpoint_name}.parquet"
                )
                day.get_trace_on(
                    endpoint_name=endpoint_name,
                    save_path=trace_out_path,
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

    def get_trace_on(
        self,
        endpoint_name: str,
        normalize_start_to: datetime = DEFAULT_TRACE_START_DATE,
        inter_chunk_gap: timedelta = timedelta(0),
        save_path: Optional[str] = None,
        force_recompose: bool = False,
    ) -> pd.DataFrame:
        """
        Get the synthesized trace for the entire composite workload on the
        specified endpoint. The synthesized trace is formed by concatenating the
        traces of all days in the composite workload on the specified endpoint.

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
            save_path: Optional path to save the synthesized trace as a Parquet
                file. If None, does not save the trace.
            force_recompose: If True, forces recomposition of the trace from its
                constituent chunks even if a saved trace already exists.

        Returns:
            A pandas DataFrame representing the synthesized trace for the
                composite workload.
        """
        
        l = [
            self.days[0].get_trace_on(
                endpoint_name=endpoint_name,
                normalize_start_to=normalize_start_to,
                inter_chunk_gap=inter_chunk_gap,
            )
        ]
        for day in self.days[1:]:
            prev_day_start = l[-1]["start_time"].min()
            this_day_start = prev_day_start + timedelta(days=1)
            day_trace = day.get_trace_on(
                endpoint_name=endpoint_name,
                normalize_start_to=this_day_start,
                inter_chunk_gap=inter_chunk_gap,
                save_path=None,
            )
            l.append(day_trace)

        # Concatenate and optionally save the synthesized trace.
        synthesized_trace = pd.concat(l).reset_index(drop=True)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            synthesized_trace.to_parquet(save_path, index=False)

        return synthesized_trace

    def plot_definition(
        self, show: bool = False, save_path: Optional[str] = None
    ):
        """
        Plot the composite workload definition.

        Parameters:
            show: If True, display the plot after saving (for interactive use).
            save_path: Optional path to save the plot image. If None, does not
                save the plot image.
        """
        eff_num_days = max(7, len(self.days))
        fig_width = eff_num_days * 0.25 + 0.5
        fig_height = max(5, max(len(day.chunks) for day in self.days)) * 0.2 + 1

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot each chunk in its respective day position
        for day_idx, day in enumerate(self.days):
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
            for i in range(len(self.days))
            if (self.monday_index + i) % 7 in [5, 6]
        ]
        weekend_starts_ends = []
        for i in range(len(weekend_day_idxs)):
            if i == 0 or weekend_day_idxs[i] != weekend_day_idxs[i - 1] + 1:
                weekend_starts_ends.append(
                    [weekend_day_idxs[i], weekend_day_idxs[i]]
                )
            else:
                weekend_starts_ends[-1][1] = weekend_day_idxs[i]

        for start, end in weekend_starts_ends:
            ax.axvspan(start - 0.5, end + 0.5, color="lightgray", alpha=0.5)

        ax.set_xlabel("Day")
        ax.set_xticks(range(eff_num_days))
        ax.set_xticklabels(
            self.day_initials() + [""] * (eff_num_days - len(self.days))
        )
        ax.set_yticks([])
        ax.set_title(self.name)
        ax.set_ylim(-1, max(len(day.chunks) for day in self.days))
        ax.set_xlim(-0.5, eff_num_days - 0.5)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )
        if show:
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


    def calculate_day_tail_on_endpoint(
        self,
        day_idx: int,
        endpoint_name: str,
        tail_percentile: float = 95.0,
    ) -> float:
        """
        Calculate the specified tail percentile of the query durations across
        the specified day on the given endpoint.

        Parameters:
            day_idx: The index of the day within the composite workload.
            endpoint_name: The name of the endpoint.
            tail_percentile: The percentile to calculate (e.g., 95.0 for 95th
                percentile).

        Returns:
            The calculated tail percentile of response times in milliseconds.
        """
        synthesized_trace = self.get_trace_on(
            endpoint_name=endpoint_name,
            normalize_start_to=self.DEFAULT_TRACE_START_DATE,
            inter_chunk_gap=timedelta(0),
            save_path=None,
        )
        tail_value = synthesized_trace["response_time_ms"].quantile(
            tail_percentile / 100.0
        )
        return tail_value
