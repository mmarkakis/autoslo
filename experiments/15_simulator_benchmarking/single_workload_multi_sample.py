import argparse
import os
from datetime import datetime

import autoslo.utils.paths as pu
from autoslo.workload_execution.workload_simulator import (
    WorkloadSimulator,
)
from autoslo.clusters.blueprint import Blueprint
from autoslo.capacity.policy_tuner import DynamicClusterConfig
from autoslo.workload_definition.redset_workload import (
    RedsetWorkloadSamplingSpec,
)
from autoslo.workload_definition.tpcds_sampler import TPCDSSampler


def main(args):

    base_sampling_spec = RedsetWorkloadSamplingSpec(
        tpcds_prob_distribution_dir=args.tpcds_prob_distribution_dir,
        seed=42,
        abs_start_time=datetime(2024, 4, 1, 0, 0, 0),
        abs_end_time=datetime(2024, 4, 2, 0, 0, 0),
        real_queries_per_output_queries=24,
        real_s_per_output_s=24,
    )

    # --- Mode-specific setup ------------------------------------------------
    dynamic_config: DynamicClusterConfig | None = None
    blueprint_label: str  # human-readable label used for experiment naming

    if args.dynamic:
        initial_rpus = tuple(args.initial_rpus)
        allowed_rpu_sizes = tuple(args.allowed_rpu_sizes)
        dynamic_config = DynamicClusterConfig(
            initial_rpus=initial_rpus,
            allowed_rpu_sizes=allowed_rpu_sizes,
            spin_up_delay_s=args.spin_up_delay_s,
        )
        blueprint_name = "dynamic"
        blueprint_label = (
            f"dynamic_init({'_'.join(str(r) for r in initial_rpus)})"
            f"_allowed({'_'.join(str(r) for r in sorted(allowed_rpu_sizes))})"
        )
    else:
        blueprint = Blueprint.maximal(max_rpu=args.max_rpu)
        blueprint_name = blueprint.name
        blueprint_label = blueprint.name

    # Derive experiment_name from CLI arg or auto-generate from key params.
    experiment_name = args.experiment_name or (
        f"{args.workload_name}__{blueprint_label}__slo{args.slo_s}"
    )

    print(
        f"Running {args.num_samples} simulations on workload "
        f"{args.workload_name} with blueprint '{blueprint_label}' and IconQ "
        f"model {args.iconq_model_id}..."
    )
    if args.dynamic:
        print(
            f"Dynamic mode: initial_rpus={dynamic_config.initial_rpus}, "
            f"allowed_rpu_sizes={dynamic_config.allowed_rpu_sizes}, "
            f"spin_up_delay_s={dynamic_config.spin_up_delay_s}, "
            f"eta_crit={args.eta_crit}, "
            f"idle_periods={args.idle_periods_before_tear_down}, "
            f"poll_interval_s={args.capacity_poll_interval_s}"
        )
    print(f"Experiment name: {experiment_name}")
    

    # Create the simulator once and reset it for each sample.
    simulator = WorkloadSimulator(
        workload_name=args.workload_name,
        iconq_model_id=args.iconq_model_id,
        blueprint_name=blueprint_name,
        slo_s=args.slo_s,
        slo_dict_filename=args.slo_dict_filename,
        optimize_based_on_slo_violation_amount=args.optimize_cumulative_slo_violation_time,
        slo_violation_rate_threshold=args.slo_violation_rate_threshold,
        slo_violation_amount_threshold_s=args.slo_violation_amount_threshold_s,
        verbose=True,
        export_video=args.export_video,
        video_frame_duration=args.video_frame_duration,
        experiment_name=experiment_name,
        overwrite_experiment=args.overwrite_experiment,
        dynamic_cluster_config=dynamic_config,
        eta_crit=args.eta_crit,
        idle_periods_before_tear_down=args.idle_periods_before_tear_down,
        capacity_poll_interval_s=args.capacity_poll_interval_s,
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
        default=20,
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
        "--slo_dict_filename",
        type=str,
        default=None,
        help=(
            "Filename (not full path) of a YAML file under "
            "data/generation_parameters/ mapping template IDs to per-template "
            "SLO values in seconds.  E.g. 'slo_dict.yml'.  When given, "
            "routing and violation stats use per-template SLOs instead of the "
            "global --slo_s for overridden templates."
        ),
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
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Name of the experiment group for this batch of runs. "
            "Runs will be stored under simulator_runs/<experiment_name>/. "
            "Defaults to '<workload>__<blueprint>__slo<slo_s>'."
        ),
    )
    parser.add_argument(
        "--overwrite_experiment",
        action="store_true",
        help=(
            "Whether to overwrite an existing experiment with the same name. "
            "If False and an experiment with the same name exists, a unique "
            "suffix is appended to the new experiment's name to avoid "
            "overwriting."
        ),  
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
        help="The maximum RPU to use in the blueprint (static mode only).",
    )

    # --- Dynamic mode -------------------------------------------------------
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help=(
            "Enable dynamic cluster provisioning mode.  When set, the "
            "simulator starts with --initial_rpus clusters and uses the "
            "capacity controller to spin up / tear down clusters at runtime. "
            "Mutually exclusive with static blueprint mode."
        ),
    )
    parser.add_argument(
        "--initial_rpus",
        type=int,
        nargs="+",
        default=[8],
        metavar="RPU",
        help=(
            "RPU sizes for clusters available from the start of the "
            "simulation (dynamic mode only).  Pass multiple values for "
            "multiple initial clusters, e.g. --initial_rpus 8 16."
        ),
    )
    parser.add_argument(
        "--allowed_rpu_sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        metavar="RPU",
        help=(
            "RPU sizes the capacity controller may spin up dynamically "
            "(dynamic mode only).  E.g. --allowed_rpu_sizes 4 8 16 32."
        ),
    )
    parser.add_argument(
        "--spin_up_delay_s",
        type=float,
        default=120.0,
        help=(
            "Simulated delay in seconds between requesting a cluster "
            "spin-up and it becoming available (dynamic mode only)."
        ),
    )
    parser.add_argument(
        "--eta_crit",
        type=float,
        default=0.1,
        help=(
            "SLO headroom threshold below which a new cluster is spun up. "
            "headroom = (slo - latency) / slo; values <= eta_crit trigger "
            "spin-up (dynamic mode only)."
        ),
    )
    parser.add_argument(
        "--idle_periods_before_tear_down",
        type=int,
        default=15,
        help=(
            "Number of consecutive idle polling periods before a cluster "
            "is torn down (dynamic mode only)."
        ),
    )
    parser.add_argument(
        "--capacity_poll_interval_s",
        type=float,
        default=60.0,
        help=(
            "How often (seconds of simulated time) the capacity controller "
            "checks SLO headroom (dynamic mode only)."
        ),
    )

    args = parser.parse_args()
    main(args)
