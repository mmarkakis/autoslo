import argparse

from autoslo.blueprint_selection.selector import BlueprintSelector
from autoslo.blueprint_selection.workload_routing_simulator import (
    WorkloadRoutingSimulator,
)

from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster

def main(args):

    if not args.use_simulator:

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
            selector_run_id=args.run_id,
            optimize_cumulative_slo_violation_time=args.optimize_cumulative_slo_violation_time,
            slo_violation_amount_threshold_s=args.slo_violation_amount_threshold_s,
        )
        if not args.use_v2:
            selector.solve()
        else:
            selector.solve_v2()

    else:
        blueprint = Blueprint.maximal()

        simulator = WorkloadRoutingSimulator(
            workload_name=args.workload_name,
            iconq_model_id=args.iconq_model_id,
            blueprint_name=blueprint.name,
            slo_s=args.slo_s,
            optimize_based_on_slo_violation_amount=args.optimize_cumulative_slo_violation_time,
            slo_violation_rate_threshold=args.slo_violation_rate_threshold,
            slo_violation_amount_threshold_s=args.slo_violation_amount_threshold_s,
            verbose=True,
            export_video=args.export_video,
            video_frame_duration=1.0,
            simulator_run_id=args.run_id,
        )
        simulator.first_pass()


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
        "--run_id",
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
    parser.add_argument(
        "--use_v2",
        action="store_true",
        help="Whether to use the v2 selector implementation.",
    )
    parser.add_argument(
        "--use_simulator",
        action="store_true",
        help="Whether to use the WorkloadRoutingSimulator instead of the selector.",
    )
    args = parser.parse_args()
    main(args)
