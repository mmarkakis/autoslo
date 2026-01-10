import argparse

from autoslo.blueprint_selection.selector import BlueprintSelector


def main(args):

    selector = BlueprintSelector(
        workload_name=args.workload_name,
        slo_s=args.slo_s,
        slo_violation_rate_threshold=args.slo_violation_rate_threshold,
        iconq_model_id=args.iconq_model_id,
        cluster_name=args.cluster_name,
        init_from_trace=True,
    )

    selector.solve(verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy script for testing selector benchmarking on traces."
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        help="The ID of the Iconq model to use.",
        default="1767983798",
    )
    parser.add_argument(
        "--workload_name",
        type=str,
        help="The name of the workload to use.",
        default="tpcds_99templates_25pctheavy_60meaninterarrivals",
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
        default=20.0,
    )
    parser.add_argument(
        "--slo_violation_rate_threshold",
        type=float,
        help="The threshold for acceptable SLO violation rate.",
        default=0.05,
    )
    args = parser.parse_args()
    main(args)
