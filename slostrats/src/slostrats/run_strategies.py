import argparse

from slostrats.user.strategy_runner import StrategyRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the specified TotalStrategies with given parameters."
    )
    parser.add_argument(
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
        "--latency_slo_s",
        type=float,
        required=True,
        help="The latency SLO in seconds.",
    )
    parser.add_argument(
        "--slo_violation_rate_threshold",
        type=float,
        default=0.05,
        help="The acceptable SLO violation rate threshold.",
    )
    parser.add_argument(
        "--workload_name",
        type=str,
        required=True,
        help="The workload to run the strategies against.",
    )
    parser.add_argument(
        "--num_training_days",
        type=int,
        default=14,
        help=(
            "The number of training days to use. The strategy is evaluated "
            "only on days after the training period."
        ),
    )
    parser.add_argument(
        "--rpu_during_training",
        type=int,
        help=(
            "During the training period, we assume a constant blueprint "
            "including a single cluster with the specified RPU."
        ),
        required=True,
    )
    parser.add_argument(
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
        rpu_during_training=args.rpu_during_training,
    )
    if not args.only_plots:
        runner.run_all()
    else:
        StrategyRunner.plot_results(
            args.workload_name,
            args.latency_slo_s,
            args.num_training_days,
            args.rpu_during_training,
        )
