import argparse

from autoslo.user.strategy_runner import StrategyRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the specified TotalStrategies with given parameters."
    )
    parser.add_argument(
        "-i",
        "--include_strategy_names",
        type=str,
        nargs="*",
        help=(
            "List of strategy names to include for running. If nonempty, only "
            "these strategies will be run. Cannot be used with "
            "--exclude_strategy_names."
        ),
    )
    parser.add_argument(
        "-e",
        "--exclude_strategy_names",
        type=str,
        nargs="*",
        help=(
            "List of strategy names to exclude from running. If nonempty, all "
            "other strategies will be run except these. Cannot be used with "
            "--include_strategy_names."
        ),
    )
    parser.add_argument(
        "-s",
        "--latency_slo_s",
        type=float,
        required=True,
        help="The latency SLO in seconds.",
    )
    parser.add_argument(
        "-v",
        "--slo_violation_rate_threshold",
        type=float,
        default=0.05,
        help="The acceptable SLO violation rate threshold.",
    )
    parser.add_argument(
        "-w",
        "--workload_name",
        type=str,
        required=True,
        help="The workload to run the strategies against.",
    )
    parser.add_argument(
        "-td",
        "--num_training_days",
        type=int,
        default=14,
        help=(
            "The number of training days to use. The strategy is evaluated "
            "only on days after the training period."
        ),
    )
    parser.add_argument(
        "-tb",
        "--training_period_blueprint_name",
        type=str,
        default="single_8",
        help=(
            "During the training period, we assume that the workload is run "
            "with the specified blueprint."
        ),
    )
    parser.add_argument(
        "-tr",
        "--training_period_query_router_name",
        default="RFixed(fixed_cluster_name='cluster_8')",
        type=str,
        help=(
            "During the training period, we assume that the workload is run "
            "with the specified query router."
        ),
    )
    parser.add_argument(
        "-p",
        "--only_plots",
        action="store_true",
        help="Generate only plots without running the strategies.",
    )

    args = parser.parse_args()

    runner = StrategyRunner(
        include_strategy_names=args.include_strategy_names,
        exclude_strategy_names=args.exclude_strategy_names,
        latency_slo_s=args.latency_slo_s,
        slo_violation_rate_threshold=args.slo_violation_rate_threshold,
        workload_name=args.workload_name,
        num_training_days=args.num_training_days,
        training_period_blueprint_name=args.training_period_blueprint_name,
        training_period_query_router_name=args.training_period_query_router_name,
    )
    if not args.only_plots:
        runner.run_all()
    else:
        StrategyRunner.plot_results(
            args.workload_name,
            args.latency_slo_s,
        )
