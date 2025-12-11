import os
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import SubFigure

import autoslo.user.strategies_metadata as smd


class StrategyPlotter:

    @staticmethod
    def _maybe_create_ax(
        ax: Optional[plt.Axes],
        figsize: tuple[int, int] = (12, 6),
    ) -> tuple[Union[plt.Figure, SubFigure], plt.Axes, bool]:
        """
        Helper to create a new figure and axes if ax is None.

        Parameters:
            ax: Optional existing Axes to use.
            figsize: Figure size if creating a new figure.

        Returns:
            A tuple containing the figure, axes, and a boolean indicating
            whether a new axes was created.
        """
        if ax is not None:
            return ax.figure, ax, False
        fig, ax = plt.figure(figsize=figsize), plt.axes()
        return fig, ax, True

    @staticmethod
    def _lineplot_kwargs(scatter: bool = False) -> dict:
        """
        Common plot keyword arguments.
        """
        strategy_colors = {
            name: info["color"] for name, info in smd.STRATEGIES.items()
        }
        strategy_markers = {
            name: info["marker"] for name, info in smd.STRATEGIES.items()
        }
        d: dict[str, Any] = {
            "style": "strategy_name",
            "markers": strategy_markers,
            "hue": "strategy_name",
            "palette": strategy_colors,
        }
        if scatter:
            d["s"] = 100
            d["edgecolor"] = "black"
            d["linewidth"] = 1.0
        else:
            d["markersize"] = 8
            d["markeredgewidth"] = 1.0
            d["markeredgecolor"] = "black"
            d["dashes"] = False
        return d

    @staticmethod
    def _finalize_axes(
        ax: plt.Axes,
        max_x_val: float,
        max_y_val: float,
        min_y_val: Optional[float] = None,
    ) -> None:
        """
        Finalize axes with limits and legend.
        """
        ax.set_ylim(
            bottom=min_y_val if min_y_val is not None else max_y_val * -0.1,
            top=max_y_val * 1.1,
        )
        ax.set_xlim(-0.5, max_x_val + 0.5)

        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_labels = [
            name for name in smd.STRATEGIES.keys() if name in label_to_handle
        ]
        ordered_handles = [label_to_handle[label] for label in ordered_labels]
        ax.legend(ordered_handles, ordered_labels, title="Strategy")

        ax.grid(True)

    @staticmethod
    def _save_and_close(
        fig: plt.Figure,
        output_dir: str,
        filename: str,
    ) -> None:
        plt_path = os.path.join(output_dir, filename)
        fig.savefig(plt_path)
        plt.close(fig)

    @staticmethod
    def plot_daily_slo_violation_rates(
        workload_name: str,
        output_dir: str,
        results_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Plot daily SLO violation rates for all workloads and strategies.
        """
        fig, ax, new_ax = StrategyPlotter._maybe_create_ax(ax)

        sns.lineplot(
            data=results_df,
            ax=ax,
            x="day_idx",
            y="slo_violation_rate",
            **StrategyPlotter._lineplot_kwargs(),
        )
        ax.set_title(
            f"SLO Violation Rates over Days for Workload '{workload_name}' "
            f"with SLO {results_df['latency_slo_s'].iloc[0]:.1f}s"
        )
        ax.set_xlabel("Day Index")
        ax.set_ylabel("SLO Violation Rate")

        # Shade the area from y=0 to y=slo_violation_rate_threshold for each day
        assert results_df["slo_violation_rate_threshold"].nunique() == 1
        ax.fill_between(
            x=[-0.5, results_df["day_idx"].max() + 0.5],
            y1=0,
            y2=results_df["slo_violation_rate_threshold"].iloc[0],
            color="green",
            alpha=0.1,
            label="Acceptable SLO Violation Rate Area",
        )

        StrategyPlotter._finalize_axes(
            ax,
            max_x_val=results_df["day_idx"].max(),
            max_y_val=results_df["slo_violation_rate"].max(),
        )

        if new_ax:
            assert type(fig) is plt.Figure  # for mypy
            StrategyPlotter._save_and_close(
                fig,
                output_dir,
                "slo_violation_rates.png",
            )

    @staticmethod
    def plot_daily_costs(
        workload_name: str,
        output_dir: str,
        results_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Plot daily total costs for all workloads and strategies.
        """
        fig, ax, new_ax = StrategyPlotter._maybe_create_ax(ax)

        sns.lineplot(
            data=results_df,
            ax=ax,
            x="day_idx",
            y="total_cost",
            **StrategyPlotter._lineplot_kwargs(),
        )
        ax.set_title(
            f"Total Cost over Days for Workload '{workload_name}' with SLO "
            f"{results_df['latency_slo_s'].iloc[0]:.1f}s"
        )
        ax.set_xlabel("Day Index")
        ax.set_ylabel("Total Cost ($)")
        StrategyPlotter._finalize_axes(
            ax,
            max_x_val=results_df["day_idx"].max(),
            max_y_val=results_df["total_cost"].max(),
        )
        if new_ax:
            assert type(fig) is plt.Figure  # for mypy
            StrategyPlotter._save_and_close(fig, output_dir, "total_costs.png")

    @staticmethod
    def plot_daily_chosen_rpu(
        workload_name: str,
        output_dir: str,
        results_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Plot daily chosen RPU for all workloads and strategies.
        """
        fig, ax, new_ax = StrategyPlotter._maybe_create_ax(ax)

        results_df["suggested_blueprint_0_rpu"] = results_df[
            "blueprint_name"
        ].apply(lambda name: int(name.split("_")[-1]))

        sns.lineplot(
            data=results_df,
            ax=ax,
            x="day_idx",
            y="suggested_blueprint_0_rpu",
            **StrategyPlotter._lineplot_kwargs(),
        )
        ax.set_title(
            f"Chosen RPU over Days for Workload '{workload_name}' with SLO "
            f"{results_df['latency_slo_s'].iloc[0]:.1f}s"
        )
        ax.set_xlabel("Day Index")
        ax.set_ylabel("Chosen RPU")
        StrategyPlotter._finalize_axes(
            ax,
            max_x_val=results_df["day_idx"].max(),
            max_y_val=32,
            min_y_val=4,
        )
        ax.set_yscale("log", base=2)

        if new_ax:
            assert type(fig) is plt.Figure  # for mypy
            StrategyPlotter._save_and_close(fig, output_dir, "chosen_rpus.png")

    @staticmethod
    def plot_scatter_slo_violations_vs_cost(
        workload_name: str,
        output_dir: str,
        results_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Scatter plot of # days with good SLO violation rate vs. total cost.
        """
        fig, ax, new_ax = StrategyPlotter._maybe_create_ax(ax, figsize=(8, 6))

        strategy_colors = {
            name: info["color"] for name, info in smd.STRATEGIES.items()
        }

        summary_df = (
            results_df.groupby("strategy_name")
            .apply(
                lambda df: pd.Series(
                    {
                        "frac_days_met_slo": (
                            df["slo_violation_rate"]
                            <= df["slo_violation_rate_threshold"]
                        ).mean(),
                        "total_cost": df["total_cost"].sum(),
                    }
                )
            )
            .reset_index()
        )

        # Don't plot the "training period" strategy if present.
        summary_df = summary_df[
            summary_df["strategy_name"] != "training_period"
        ]

        sns.scatterplot(
            data=summary_df,
            x="total_cost",
            y="frac_days_met_slo",
            ax=ax,
            **StrategyPlotter._lineplot_kwargs(scatter=True),
        )
        ax.set_title(
            "Fraction of Days Meeting SLO vs. Total Cost\n "
            f"Workload: {workload_name}\n"
            f"SLO {results_df['latency_slo_s'].iloc[0]:.1f}s "
            "(Acceptable violation Rate: "
            f"{results_df['slo_violation_rate_threshold'].iloc[0]:.2})"
        )
        ax.set_xlabel("Total Cost ($)")
        ax.set_ylabel("Fraction of Days Meeting SLO")
        StrategyPlotter._finalize_axes(
            ax,
            max_x_val=summary_df["total_cost"].max(),
            max_y_val=summary_df["frac_days_met_slo"].max(),
        )

        if new_ax:
            assert type(fig) is plt.Figure  # for mypy
            StrategyPlotter._save_and_close(fig, output_dir, "slo_vs_cost.png")
        # Also return the summary DataFrame for further analysis if needed.
        return summary_df
