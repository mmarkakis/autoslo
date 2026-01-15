import argparse

from autoslo.blueprint_selection.selector import BlueprintSelector


def main(args):

    selector = BlueprintSelector(
        workload_name=args.workload_name,
        slo_s=args.slo_s,
        slo_violation_rate_threshold=args.slo_violation_rate_threshold,
        iconq_model_id=args.iconq_model_id,
        cluster_name=args.cluster_name,
        init_from_trace=args.init_from_trace,
        use_stage_for_isolated_queries=args.use_stage_for_isolated_queries,
        max_iters=20,
        verbose=True,
        export_video=args.export_video,
        selector_run_id=args.selector_run_id,
        optimize_cumulative_slo_violation_time=args.optimize_cumulative_slo_violation_time,
        slo_violation_amount_threshold_s=args.slo_violation_amount_threshold_s
    )

    selector.solve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy script for testing selector benchmarking on traces."
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        help="The ID of the Iconq model to use.",
        default="1768492742",
    )
    parser.add_argument(
        "--workload_name",
        type=str,
        help="The name of the workload to use.",
        default="tpcds_99templates_25pctheavy_30meaninterarrivals",
    )
    parser.add_argument(
        "--cluster_name",
        type=str,
        help="The name of the cluster to use.",
        default="cluster_8",
    )
    parser.add_argument(
        "--slo_s",
        type=float,
        help="The SLO to meet, in seconds.",
        default=180.0,
    )
    parser.add_argument(
        "--slo_violation_rate_threshold",
        type=float,
        help="The threshold for acceptable SLO violation rate.",
        default=0.00,
    )
    parser.add_argument(
        "--use_stage_for_isolated_queries",
        action="store_true",
        help="Whether to use the StageModel for isolated queries.",
    )
    parser.add_argument(
        "--export_video",
        action="store_true",
        help="Whether to export a video of the selection process.",
    )
    parser.add_argument(
        "--selector_run_id",
        type=str,
        help="Optional run ID to use for the selector run.",
        default=None,
    )
    parser.add_argument(
        "--init_from_trace",
        action="store_true",
        help="Whether to initialize the selector from the trace.",
    )
    parser.add_argument(
        "--optimize_cumulative_slo_violation_time",
        action="store_true",
        help="Whether to optimize for cumulative SLO violation in seconds.",
    )
    parser.add_argument(
        "--slo_violation_amount_threshold_s",
        type=float,
        help="The threshold for acceptable cumulative SLO violation time in seconds.",
        default=30.0,
        )
    args = parser.parse_args()
    main(args)
