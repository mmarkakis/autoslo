import argparse

from autoslo.blueprint_selection.selector import BlueprintSelector

from tqdm.auto import tqdm


def main(args):

    pct_heavy_options = [0, 10, 25, 50]
    mean_interarrival_options = [10, 30, 60, 120]
    rpus = [8]

    total = len(pct_heavy_options) * len(mean_interarrival_options) * len(rpus)
    curr = 0

    for pct_heavy in pct_heavy_options:
        for mean_interarrival in mean_interarrival_options:
            for rpu in rpus:
                workload_name = f"tpcds_99templates_{pct_heavy:02d}pctheavy_{mean_interarrival}meaninterarrivals"

                cluster_name = f"cluster_{rpu}"

                print(
                    f"({curr+1}/{total}) Running selection on workload "
                    f"{workload_name} with cluster {cluster_name} and IconQ "
                    f"model {args.iconq_model_id}..."
                )

                selector = BlueprintSelector(
                    workload_name=workload_name,
                    slo_s=args.slo_s,
                    slo_violation_rate_threshold=args.slo_violation_rate_threshold,
                    iconq_model_id=args.iconq_model_id,
                    cluster_name=cluster_name,
                    init_from_trace=True,
                    use_stage_for_isolated_queries=args.use_stage_for_isolated_queries,
                    max_iters=20,
                    verbose=False,
                    export_video=args.export_video,
                    video_frame_duration=args.video_frame_duration,
                )

                selector.solve()
                print(f"\tDone, at selector run ID: {selector._run_id}")
                curr += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy script for testing selector benchmarking on traces."
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        help="The ID of the Iconq model to use.",
        default="1768080208",
    )
    parser.add_argument(
        "--slo_s",
        type=float,
        help="The SLO to meet, in seconds.",
        default=120.0,
    )
    parser.add_argument(
        "--slo_violation_rate_threshold",
        type=float,
        help="The threshold for acceptable SLO violation rate.",
        default=0.05,
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
        "--video_frame_duration",
        type=float,
        default=1.0,
        help="Duration of each frame in the exported video, in seconds.",
    )
    args = parser.parse_args()
    main(args)
