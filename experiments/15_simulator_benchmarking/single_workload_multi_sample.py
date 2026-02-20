import argparse
import os
from datetime import datetime

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.workload_routing_simulator import (
    WorkloadRoutingSimulator,
)
from autoslo.blueprints.blueprint import Blueprint
from autoslo.workload_definition.redset_workload import (
    RedsetWorkloadSamplingSpec,
)
from autoslo.workload_definition.tpcds_sampler import TPCDSSampler


def main(args):

    base_sampling_spec = RedsetWorkloadSamplingSpec(
        tpcds_prob_distribution_dir=args.tpcds_prob_distribution_dir,
        seed=42,
        abs_start_time=datetime(2024, 3, 1, 0, 0, 0),
        abs_end_time=datetime(2024, 3, 1, 4, 0, 0),
        real_queries_per_output_queries=4,
        real_s_per_output_s=4,
    )
    blueprint = Blueprint.maximal(max_rpu=32)

    print(
        f"Running {args.num_samples} simulations on workload "
        f"{args.workload_name} with blueprint {blueprint.name} and IconQ "
        f"model {args.iconq_model_id}..."
    )

    # Create the simulator once and reset it for each sample.
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
        video_frame_duration=args.video_frame_duration,
    )

    # Warm up featurization caches before the first sample.
    sampler = TPCDSSampler.from_dir(args.tpcds_prob_distribution_dir)
    tpcds_vocab = list(sampler.column_dict.values())
    simulator._iconq_model.iconq_query_featurizer.warm_up_cache(tpcds_vocab)
    print("Featurization cache warm-up complete.")

    for i in range(args.num_samples):
        sampling_spec = RedsetWorkloadSamplingSpec(
            tpcds_prob_distribution_dir=args.tpcds_prob_distribution_dir,
            seed=base_sampling_spec.seed + i,
            abs_start_time=base_sampling_spec.abs_start_time,
            abs_end_time=base_sampling_spec.abs_end_time,
            real_queries_per_output_queries=base_sampling_spec.real_queries_per_output_queries,
            real_s_per_output_s=base_sampling_spec.real_s_per_output_s,
        )

        start = datetime.now().timestamp()
        simulator.reset()
        simulator.simulate_one(sampling_spec=sampling_spec)
        end = datetime.now().timestamp()
        print(
            f"Iteration ({i+1}/{args.num_samples}): Done after "
            f"{(end-start):.2f} seconds, at simulator run ID: "
            f"{simulator._run_id}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy script for testing simulator benchmarking on traces."
    )
    parser.add_argument(
        "--workload_name",
        type=str,
        help="The name of the workload to simulate on.",
        default="redset_provisioned_cluster12",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="The number of samples to take from the workload for simulation.",
        default=5,
    )
    parser.add_argument(
        "--tpcds_prob_distribution_dir",
        type=str,
        help=(
            "The directory containing the TPCDS probability distributions to "
            "use for sampling."
        ),
        default=os.path.join(
            pu.get_data_path(), "generation_parameters", "dist_16_rpu"
        ),
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        help="The ID of the Iconq model to use.",
        default="1771539369",
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
        default=0.05,
    )
    parser.add_argument(
        "--export_video",
        type=bool,
        default=False,
        help="Whether to export a video of the selection process.",
    )
    parser.add_argument(
        "--video_frame_duration",
        type=float,
        default=1.0,
        help="Duration of each frame in the exported video, in seconds.",
    )
    parser.add_argument(
        "--optimize_cumulative_slo_violation_time",
        type=bool,
        default=True,
        help="Whether to optimize for cumulative SLO violation in seconds.",
    )
    parser.add_argument(
        "--slo_violation_amount_threshold_s",
        type=float,
        help="The threshold for acceptable cumulative SLO violation time in seconds.",
        default=30.0,
    )
    parser.add_argument(
        "--continue_runs",
        action="store_true",
        help="Whether to continue from an existing collection of runs",
    )
    parser.add_argument(
        "--max_rpu",
        type=int,
        default=32,
        help="The maximum RPU to use in the blueprint.",
    )

    args = parser.parse_args()
    main(args)
