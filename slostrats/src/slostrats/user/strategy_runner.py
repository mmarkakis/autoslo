import multiprocessing as mp
import os
from typing import Callable, Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

import chunkload.utils.paths as pu
import slostrats.user.strategies_metadata as smd
from chunkload.building_blocks.composite import Composite
from slostrats.user.strategy_plotter import StrategyPlotter


class StrategyRunner:

    @staticmethod
    def outputs_parent_dir() -> str:
        """
        Return the parent directory where strategy outputs are stored.
        """
        return os.path.join(pu.get_data_path(), "strategy_runs")

    def __init__(
        self,
        include_strategy_names: Optional[list[str]],
        exclude_strategy_names: Optional[list[str]],
        latency_slo_s: float,
        slo_violation_rate_threshold: float,
        workload_name: str,
        num_training_days: int,
    ):
        """
        Instantiate a new StrategyRunner.

        Parameters:
            include_strategy_names: List of strategy names to include.
            exclude_strategy_names: List of strategy names to exclude.
            latency_slo_s: The latency SLO in seconds.
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold.
            workload_name: The workload to run the strategies against.
            num_training_days: The number of training days to use. The strategy
                is evaluated only on days after the training period.
        """

        self._validate_args(
            include_strategy_names,
            exclude_strategy_names,
            latency_slo_s,
            slo_violation_rate_threshold,
            workload_name,
            num_training_days,
        )
        self.include_strategy_names = include_strategy_names
        self.exclude_strategy_names = exclude_strategy_names
        self.latency_slo_s = latency_slo_s
        self.slo_violation_rate_threshold = slo_violation_rate_threshold
        self.workload_name = workload_name
        self.num_training_days = num_training_days

        # Determine which strategies to run based on include/exclude lists.
        self.strategy_names_to_run = self._strategy_names_to_run()

    def _validate_args(
        self,
        include_strategy_names: Optional[list[str]],
        exclude_strategy_names: Optional[list[str]],
        latency_slo_s: float,
        slo_violation_rate_threshold: float,
        workload_name: str,
        num_training_days: int,
    ) -> None:
        """
        Validate the provided arguments to the constructor.

        Parameters:
            include_strategy_names: List of strategy names to include.
            exclude_strategy_names: List of strategy names to exclude.
            latency_slo_s: The latency SLO in seconds.
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold.
            workload_name: The workload to run the strategies against.
            num_training_days: The number of training days to use. The strategy
                is evaluated only on days after the training period.

        Raises:
            ValueError: If any argument is invalid.
        """
        # Validate include/exclude strategy names.
        if include_strategy_names and exclude_strategy_names:
            raise ValueError(
                "Cannot specify both include_strategy_names and "
                "exclude_strategy_names."
            )
        if include_strategy_names:
            for name in include_strategy_names:
                if name not in smd.STRATEGIES:
                    raise ValueError(
                        f"Included strategy name '{name}' is not recognized."
                    )
        if exclude_strategy_names:
            for name in exclude_strategy_names:
                if name not in smd.STRATEGIES:
                    raise ValueError(
                        f"Excluded strategy name '{name}' is not recognized."
                    )

        # Validate latency_slo_s and slo_violation_rate_threshold.
        if latency_slo_s <= 0:
            raise ValueError("latency_slo_s must be positive.")
        if not (0 <= slo_violation_rate_threshold <= 1):
            raise ValueError(
                "slo_violation_rate_threshold must be between 0 and 1."
            )

        # Validate workload_name.
        if not workload_name:
            raise ValueError("workload_name must be a non-empty string.")
        if workload_name not in Composite.all_composite_workload_names():
            raise ValueError(
                f"workload_name '{workload_name}' is not a recognized "
                "composite workload."
            )

        # Validate num_training_days.
        if num_training_days < 0:
            raise ValueError("num_training_days must be non-negative.")

    def _strategy_names_to_run(self) -> list[str]:
        """
        Determine which strategy names should be run based on the
        include/exclude lists.

        Returns:
            A list of strategy names to run.
        """
        strategy_names = list(smd.STRATEGIES.keys())
        if self.include_strategy_names:
            strategy_names = self.include_strategy_names
        elif self.exclude_strategy_names:
            strategy_names = [
                name
                for name in strategy_names
                if name not in self.exclude_strategy_names
            ]
        return strategy_names
    
    def output_dir(self) -> str:
        """
        Return the output directory for the current strategy run.

        Returns:
            The output directory path.
        """
        return os.path.join(
            self.outputs_parent_dir(),
            self.workload_name,
            f"{int(self.latency_slo_s)}s_slo",
        )

    def run_one(self, strategy_name: str):
        """
        Run a single strategy by name.

        Parameters:
            strategy_name: The name of the strategy to run.
        """
        # Instantiate the strategy.
        strategy_class = cast(Callable, smd.STRATEGIES[strategy_name]["class"])
        strategy_instance = strategy_class(
            latency_slo_s=self.latency_slo_s,
            slo_violation_rate_threshold=self.slo_violation_rate_threshold,
        )

        # For the chosen workload, after the training period, run the strategy
        # for each day.
        workload = Composite.load(workload_name=self.workload_name)
        num_days = workload.num_days()
        records = []
        for day_idx in range(self.num_training_days, num_days):
            suggested_blueprint, perf_of_suggested_blueprint = (
                strategy_instance.perf_of_suggested_blueprint(
                    workload_name=self.workload_name,
                    day_idx=day_idx,
                    latency_slo_s=self.latency_slo_s,
                )
            )
            records.append(
                {
                    "day_idx": day_idx,
                    "suggested_blueprint_0_rpu": suggested_blueprint.clusters[
                        0
                    ].rpu,  # TODO: Handle multi-cluster blueprints
                    "slo_violation_rate": (
                        perf_of_suggested_blueprint.slo_violation_rate
                    ),
                    "total_cost": perf_of_suggested_blueprint.cost,
                }
            )

        # Save the results as a Parquet file.
        output_dir = self.output_dir()  
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{strategy_name}.parquet")
        records_df = pd.DataFrame(records)
        records_df["workload_name"] = self.workload_name
        records_df["latency_slo_s"] = self.latency_slo_s
        records_df["slo_violation_rate_threshold"] = (
            self.slo_violation_rate_threshold
        )
        records_df["num_training_days"] = self.num_training_days
        records_df["strategy_name"] = strategy_name
        records_df = (
            records_df[
                [
                    "strategy_name",
                    "workload_name",
                    "day_idx",
                    "latency_slo_s",
                    "slo_violation_rate_threshold",
                    "suggested_blueprint_0_rpu",
                    "slo_violation_rate",
                    "total_cost",
                    "num_training_days",
                ]
            ]
            .sort_values(by=["day_idx"])
            .reset_index(drop=True)
        )
        records_df.to_parquet(output_path, index=False)

    def run_all(self):
        """
        Run all strategies determined by the include/exclude lists, and plot the
        results.
        """
        num_cpus = min(
            max(1, mp.cpu_count() - 1), len(self.strategy_names_to_run)
        )
        with mp.Pool(processes=num_cpus) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        self.run_one, self.strategy_names_to_run
                    ),
                    total=len(self.strategy_names_to_run),
                    desc="Running strategies",
                )
            )

        StrategyRunner.plot_results(
            self.workload_name, latency_slo_s=self.latency_slo_s
        )

    @staticmethod
    def plot_results(workload_name: str, latency_slo_s: float):
        """
        Plot the results of all strategy runs for the specified workload.

        Parameters:
            workload_name: The name of the workload to plot results for.
            latency_slo_s: The latency SLO in seconds.
        """

        # Read in all strategy results for the workload.
        output_dir = os.path.join(
            StrategyRunner.outputs_parent_dir(),
            workload_name,
            f"{int(latency_slo_s)}s_slo",
        )
        all_records = []
        for strategy_name in os.listdir(output_dir):
            if strategy_name.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(output_dir, strategy_name))
                all_records.append(df)
        if not all_records:
            print(f"No results found for workload '{workload_name}'.")
            return
        results_df = pd.concat(all_records, ignore_index=True)

        # Generate the plots individually.
        StrategyPlotter.plot_daily_slo_violation_rates(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
        )
        StrategyPlotter.plot_daily_costs(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
        )
        StrategyPlotter.plot_daily_chosen_rpu(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
        )
        summary_df = StrategyPlotter.plot_scatter_slo_violations_vs_cost(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
        )

        # Also generate a single figure with the workload definition and all
        # three plots for easier comparison. Make the first plot have a shorter
        # height.
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [0.5, 1, 1, 1]},
        )
        workload = Composite.load(workload_name=workload_name)
        workload.plot_definition(
            save_path=None,
            ax=axes[0],
            show=False,
            start_day_idx=0,
        )
        StrategyPlotter.plot_daily_slo_violation_rates(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
            ax=axes[1],
        )
        StrategyPlotter.plot_daily_costs(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
            ax=axes[2],
        )
        StrategyPlotter.plot_daily_chosen_rpu(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
            ax=axes[3],
        )
        fig.tight_layout()
        combined_plt_path = os.path.join(
            output_dir, "combined_strategy_plots.png"
        )
        fig.savefig(combined_plt_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Also save these scatterplot data to a CSV for further analysis.
        csv_path = os.path.join(output_dir, "slo_vs_cost_summary.csv")
        summary_df.to_csv(csv_path, index=False)
