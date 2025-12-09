import multiprocessing as mp
import os
from typing import Callable, Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

import autoslo.user.strategies_metadata as smd
import autoslo.utils.paralellism as plu
import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.blueprint_timeseries import BlueprintTimeseries
from autoslo.blueprints.cluster import Cluster
from autoslo.routing.query_router import QueryRouter
from autoslo.strategies.slo_strategy import SLOStrategy
from autoslo.strategies.slo_strategy_performance import SLOStrategyPerformance
from autoslo.user.strategy_plotter import StrategyPlotter
from autoslo.workload_definition.composite import Composite
from autoslo.workload_execution.trace import Trace


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
        training_period_blueprint_name: str,
        training_period_query_router_name: str,
        model_training_run_id: str,
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
            training_period_blueprint_name: The blueprint name used during the
                training period.
            training_period_query_router_name: The query router name used during
                the training period.
            model_training_run_id: The run ID where the model is stored.
        """

        StrategyRunner._validate_args(
            include_strategy_names=include_strategy_names,
            exclude_strategy_names=exclude_strategy_names,
            latency_slo_s=latency_slo_s,
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            workload_name=workload_name,
            num_training_days=num_training_days,
            training_period_blueprint_name=training_period_blueprint_name,
            training_period_query_router_name=training_period_query_router_name,
            model_training_run_id=model_training_run_id,
        )
        self.include_strategy_names = include_strategy_names
        self.exclude_strategy_names = exclude_strategy_names
        self.latency_slo_s = latency_slo_s
        self.slo_violation_rate_threshold = slo_violation_rate_threshold
        self.workload_name = workload_name
        self.num_training_days = num_training_days
        self.training_period_blueprint_name = training_period_blueprint_name
        self.training_period_query_router_name = (
            training_period_query_router_name
        )
        self.model_training_run_id = model_training_run_id

        # Determine which strategies to run based on include/exclude lists.
        self.strategy_names_to_run = self._strategy_names_to_run()

    @staticmethod
    def _validate_args(
        **kwargs,
    ) -> None:
        """
        Validate the provided arguments.

        Parameters:
            **kwargs: The arguments to validate.

        Raises:
            ValueError: If any argument is invalid.
        """
        # Validate include/exclude strategy names.
        if (
            "include_strategy_names" in kwargs
            and "exclude_strategy_names" in kwargs
        ):
            include_strategy_names = kwargs["include_strategy_names"]
            exclude_strategy_names = kwargs["exclude_strategy_names"]
            if include_strategy_names and exclude_strategy_names:
                raise ValueError(
                    "Cannot specify both include_strategy_names and "
                    "exclude_strategy_names."
                )
        if "include_strategy_names" in kwargs:
            include_strategy_names = kwargs["include_strategy_names"]
            if include_strategy_names:
                for name in include_strategy_names:
                    if name not in smd.STRATEGIES:
                        raise ValueError(
                            f"Included strategy name '{name}' is not recognized."
                        )

        if "exclude_strategy_names" in kwargs:
            exclude_strategy_names = kwargs["exclude_strategy_names"]
            if exclude_strategy_names:
                for name in exclude_strategy_names:
                    if name not in smd.STRATEGIES:
                        raise ValueError(
                            f"Excluded strategy name '{name}' is not recognized."
                        )

        # Validate latency_slo_s and slo_violation_rate_threshold.
        if "latency_slo_s" in kwargs:
            if kwargs["latency_slo_s"] <= 0:
                raise ValueError("latency_slo_s must be positive.")

        if "slo_violation_rate_threshold" in kwargs:
            if not (0 <= kwargs["slo_violation_rate_threshold"] <= 1):
                raise ValueError(
                    "slo_violation_rate_threshold must be between 0 and 1."
                )

        # Validate workload_name.
        if "workload_name" in kwargs:
            workload_name = kwargs["workload_name"]
            if not workload_name:
                raise ValueError("workload_name must be a non-empty string.")
            if workload_name not in Composite.all_composite_workload_names():
                raise ValueError(
                    f"workload_name '{workload_name}' is not a recognized "
                    "composite workload."
                )

        # Validate num_training_days.
        if "num_training_days" in kwargs:
            if kwargs["num_training_days"] < 0:
                raise ValueError("num_training_days must be non-negative.")

        # Validate blueprint and query router names during training.
        if "training_period_blueprint_name" in kwargs:
            blueprint_name = kwargs["training_period_blueprint_name"]
            blueprint = Blueprint.from_config(blueprint_name)
            if blueprint_name not in pu.get_blueprint_dicts_from_config():
                raise ValueError(
                    f"training_period_blueprint_name "
                    f"'{blueprint_name}' is not a recognized blueprint name."
                )

        if "training_period_query_router_name" in kwargs:
            query_router_name = kwargs["training_period_query_router_name"]
            try:
                QueryRouter.from_name(
                    query_router_name,
                    blueprint=Blueprint.from_config(blueprint_name),
                )
            except ValueError as e:
                raise ValueError(
                    f"training_period_query_router_name "
                    f"'{query_router_name}' is not a recognized query router "
                    f"name or has invalid parameters: {e}"
                ) from e

        # Validate model_training_run_id.
        if "model_training_run_id" in kwargs:
            model_training_runs_df = pu.ModelLocator.get_runs_df()
            model_training_run_id = kwargs["model_training_run_id"]
            if (
                model_training_run_id
                not in model_training_runs_df["run_id"].values
            ):
                raise ValueError(
                    f"model_training_run_id '{model_training_run_id}' does not "
                    "exist in the model runs."
                )

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

    def _collect_training_period_data(self) -> None:
        """
        Collect and store the trajectory of the workload during the training
        period. For each day in the training period, we retrieve the trace on
        the size defined by rpu_during_training.
        """
        workload = Composite.load(workload_name=self.workload_name)
        records = []
        training_period_blueprint = Blueprint.from_config(
            self.training_period_blueprint_name
        )
        training_period_query_router = QueryRouter.from_name(
            self.training_period_query_router_name,
            blueprint=training_period_blueprint,
        )
        for day_idx in range(self.num_training_days):
            perf: SLOStrategyPerformance = SLOStrategy.evaluate_suggestion(
                workload=workload,
                day_idx=day_idx,
                latency_slo_s=self.latency_slo_s,
                blueprint=training_period_blueprint,
                query_router=training_period_query_router,
            )

            records.append(
                {
                    "strategy_name": "training_period",
                    "workload_name": self.workload_name,
                    "num_training_days": self.num_training_days,
                    "day_idx": day_idx,
                    "latency_slo_s": self.latency_slo_s,
                    "slo_violation_rate_threshold": (
                        self.slo_violation_rate_threshold
                    ),
                    "blueprint_name": (self.training_period_blueprint_name),
                    "query_router_name": (
                        self.training_period_query_router_name
                    ),
                    "num_slo_violations": perf.num_slo_violations(),
                    "num_total_queries": perf.num_total_queries(),
                    "slo_violation_rate": perf.slo_violation_rate(),
                    "total_cost": perf.total_cost(),
                    "total_routing_time_s": perf.total_routing_time_s(),
                }
            )

        # Save the training period data as a Parquet file.
        output_dir = self.output_dir()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            "_".join(
                [
                    "training_period",
                    self.training_period_blueprint_name,
                    self.training_period_query_router_name,
                ]
            )
            + ".parquet",
        )
        records_df = pd.DataFrame(records)
        records_df.to_parquet(output_path, index=False)

    def _run_one(self, strategy_name: str):
        """
        Run a single strategy by name.

        Parameters:
            strategy_name: The name of the strategy to run.
        """
        strategy_class: Optional[Callable] = cast(
            Optional[Callable], smd.STRATEGIES[strategy_name]["class"]
        )
        if strategy_class is None:
            # Skip placeholder strategies like "training_period".
            return

        # Instantiate the strategy.
        strategy_instance = strategy_class(
            latency_slo_s=self.latency_slo_s,
            slo_violation_rate_threshold=self.slo_violation_rate_threshold,
            model_training_run_id=self.model_training_run_id,
        )

        # Initialize past blueprints based on the training period.
        past_blueprints = BlueprintTimeseries.empty()
        training_period_blueprint = Blueprint.from_config(
            self.training_period_blueprint_name
        )
        for day_idx in range(self.num_training_days):
            past_blueprints.set_blueprint_for_period(
                period_idx=day_idx,
                blueprint=training_period_blueprint,
            )

        # For the chosen workload, after the training period, run the strategy
        # for each day.
        workload = Composite.load(workload_name=self.workload_name)
        num_days = workload.num_days()
        records = []
        for day_idx in range(self.num_training_days, num_days):
            # Find the suggestion for this day and log it.
            suggested_blueprint, suggested_query_router = (
                strategy_instance.suggest(
                    workload=workload,
                    day_idx=day_idx,
                    latency_slo_s=self.latency_slo_s,
                    past_blueprints=past_blueprints,
                )
            )
            past_blueprints.set_blueprint_for_period(
                period_idx=day_idx, blueprint=suggested_blueprint
            )

            # Evaluate the performance of the suggested blueprint/query router.
            perf: SLOStrategyPerformance = (
                strategy_instance.evaluate_suggestion(
                    workload=workload,
                    day_idx=day_idx,
                    latency_slo_s=self.latency_slo_s,
                    blueprint=suggested_blueprint,
                    query_router=suggested_query_router,
                )
            )
            records.append(
                {
                    "day_idx": day_idx,
                    "blueprint_name": suggested_blueprint.name,
                    "query_router_name": (suggested_query_router.name),
                    "num_slo_violations": perf.num_slo_violations(),
                    "num_total_queries": perf.num_total_queries(),
                    "slo_violation_rate": (perf.slo_violation_rate()),
                    "total_cost": perf.total_cost(),
                    "total_routing_time_s": (perf.total_routing_time_s()),
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
        records_df["model_training_run_id"] = self.model_training_run_id
        records_df["strategy_name"] = strategy_name
        records_df = (
            records_df[
                [
                    "strategy_name",
                    "workload_name",
                    "num_training_days",
                    "day_idx",
                    "latency_slo_s",
                    "slo_violation_rate_threshold",
                    "blueprint_name",
                    "query_router_name",
                    "model_training_run_id",
                    "num_slo_violations",
                    "num_total_queries",
                    "slo_violation_rate",
                    "total_cost",
                    "total_routing_time_s",
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

        # Retrieve the trajectory of the workload during the training period.
        self._collect_training_period_data()

        # Run the strategies over the test period in parallel.
        with mp.Pool(processes=plu.deg_of_paralellism()) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        self._run_one, self.strategy_names_to_run
                    ),
                    total=len(self.strategy_names_to_run),
                    desc="Running strategies",
                )
            )

        # Plot the results.
        StrategyRunner.plot_results(
            workload_name=self.workload_name,
            latency_slo_s=self.latency_slo_s,
            exclude_strategy_names=self.exclude_strategy_names,
        )

    @staticmethod
    def plot_results(
        workload_name: str,
        latency_slo_s: float,
        exclude_strategy_names: Optional[list[str]] = None,
    ):
        """
        Plot the results of strategy runs for the specified workload.

        Parameters:
            workload_name: The name of the workload to plot results for.
            latency_slo_s: The latency SLO in seconds.
            exclude_strategy_names: List of strategy names to exclude from
                the plots.
        """
        if exclude_strategy_names is None:
            exclude_strategy_names = []

        # Validate input arguments.
        StrategyRunner._validate_args(
            workload_name=workload_name,
            latency_slo_s=latency_slo_s,
            exclude_strategy_names=exclude_strategy_names,
        )

        # Read in all strategy results for the workload.
        output_dir = os.path.join(
            StrategyRunner.outputs_parent_dir(),
            workload_name,
            f"{int(latency_slo_s)}s_slo",
        )
        all_records = []
        for strategy_filename in os.listdir(output_dir):
            strategy_name, ext = os.path.splitext(strategy_filename)
            if (
                ext == ".parquet"
                and strategy_name not in exclude_strategy_names
            ):
                df = pd.read_parquet(
                    os.path.join(output_dir, strategy_filename)
                )
                all_records.append(df)
        if not all_records:
            print(f"No results found for workload '{workload_name}'.")
            return
        results_df = pd.concat(all_records, ignore_index=True)

        # Generate the plots individually.
        # StrategyPlotter.plot_daily_slo_violation_rates(
        #     workload_name=workload_name,
        #     output_dir=output_dir,
        #     results_df=results_df,
        # )
        # StrategyPlotter.plot_daily_costs(
        #     workload_name=workload_name,
        #     output_dir=output_dir,
        #     results_df=results_df,
        # )
        # StrategyPlotter.plot_daily_chosen_rpu(
        #     workload_name=workload_name,
        #     output_dir=output_dir,
        #     results_df=results_df,
        # )
        summary_df = StrategyPlotter.plot_scatter_slo_violations_vs_cost(
            workload_name=workload_name,
            output_dir=output_dir,
            results_df=results_df,
        )

        # Also save these scatterplot data to a CSV for further analysis.
        csv_path = os.path.join(output_dir, "slo_vs_cost_summary.csv")
        summary_df.to_csv(csv_path, index=False)

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
