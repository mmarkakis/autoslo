import argparse

from autoslo.user.strategy_runner import StrategyRunner

from tqdm.auto import tqdm


ALL_WORKLOAD_NAMES = [
    "weekly_set",
    "weekly_peak",
    "weekly_random",
    "growth_h_base",
    "growth_h_added",
    "growth_h_noisy",
    "growth_t_base",
    "growth_t_added",
    "growth_t_noisy",
]

ALL_SLO_S = [10, 30, 60, 120, 300]

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

    latency_slo_s_list = []
    workload_name_list = []

    if "latency_slo_s" not in args or args.latency_slo_s is None:
        latency_slo_s_list = ALL_SLO_S
    else:
        latency_slo_s_list = [args.latency_slo_s]
    if "workload_name" not in args or args.workload_name is None:
        workload_name_list = ALL_WORKLOAD_NAMES
    else:
        workload_name_list = [args.workload_name]

    combinations = [
        (workload_name, latency_slo_s)
        for workload_name in workload_name_list
        for latency_slo_s in latency_slo_s_list
    ]

    for i, (workload_name, latency_slo_s) in enumerate(combinations):
        print(
            f"({i+1}/{len(combinations)}) Running strategies for workload "
            f"'{workload_name}' with latency SLO {latency_slo_s}s"
        )

        runner = StrategyRunner(
            include_strategy_names=args.include_strategy_names,
            exclude_strategy_names=args.exclude_strategy_names,
            latency_slo_s=latency_slo_s,
            slo_violation_rate_threshold=args.slo_violation_rate_threshold,
            workload_name=workload_name,
            num_training_days=args.num_training_days,
            training_period_blueprint_name=args.training_period_blueprint_name,
            training_period_query_router_name=args.training_period_query_router_name,
        )
        if not args.only_plots:
            runner.run_all()
        else:
            StrategyRunner.plot_results(
                workload_name,
                latency_slo_s,
            )
