import argparse

from autoslo.blueprint_selection.selector import BlueprintSelector

from tqdm.auto import tqdm
import yaml
import autoslo.utils.paths as pu
import os
from datetime import datetime


def main(args):

    pct_heavy_options = [0, 10, 25, 50]
    mean_interarrival_options = [120, 60, 30, 10]
    rpus = [8]

    total = len(pct_heavy_options) * len(mean_interarrival_options) * len(rpus)
    curr = 0

    bookkeeping_filename = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "08_selector_benchmarking",
        "selector_benchmarking_bookkeeping.yml",
    )
    bookkeeping = {}
    if args.continue_runs and os.path.exists(bookkeeping_filename):
        with open(bookkeeping_filename, "r") as f:
            bookkeeping = yaml.safe_load(f)
        existing_runs = bookkeeping.get("runs", [])
        completed_set = set(
            (
                run["pct_heavy"],
                run["mean_interarrival"],
                run["rpu"],
            )
            for run in existing_runs
        )
    else:
        bookkeeping = {
            "iconq_model_id": args.iconq_model_id,
            "slo_s": args.slo_s,
            "slo_violation_rate_threshold": args.slo_violation_rate_threshold,
            "use_stage_for_isolated_queries": args.use_stage_for_isolated_queries,
            "export_video": args.export_video,
            "video_frame_duration": args.video_frame_duration,
            "init_from_trace": args.init_from_trace,
            "optimize_cumulative_slo_violation_time": args.optimize_cumulative_slo_violation_time,
            "slo_violation_amount_threshold_s": args.slo_violation_amount_threshold_s,
            "use_v2": args.use_v2,
            "runs": [],
        }
        
        with open(bookkeeping_filename, "w") as f:
            yaml.dump(bookkeeping, f, sort_keys=False)

    
    for mean_interarrival in mean_interarrival_options:
        for pct_heavy in pct_heavy_options:

            workload_name = f"tpcds_99templates_{pct_heavy:02d}pctheavy_{mean_interarrival}meaninterarrivals"

            for rpu in rpus:

                if args.continue_runs and (pct_heavy, mean_interarrival, rpu) in completed_set:
                    print(
                        f"({curr+1}/{total}) Skipping already completed run for "
                        f"workload {workload_name} with rpu {rpu}."
                    )
                    curr += 1
                    continue

                cluster_name = f"cluster_{rpu}"

                print(
                    f"({curr+1}/{total}) Running selection on workload "
                    f"{workload_name} with cluster {cluster_name} and IconQ "
                    f"model {args.iconq_model_id}..."
                )
                start = datetime.now().timestamp()
                selector = BlueprintSelector(
                    workload_name=workload_name,
                    slo_s=args.slo_s,
                    slo_violation_rate_threshold=args.slo_violation_rate_threshold,
                    iconq_model_id=args.iconq_model_id,
                    cluster_name=cluster_name,
                    init_from_trace=args.init_from_trace,
                    use_stage_for_isolated_queries=args.use_stage_for_isolated_queries,
                    max_iters=20,
                    verbose=True,
                    export_video=args.export_video,
                    video_frame_duration=args.video_frame_duration,
                    optimize_cumulative_slo_violation_time=args.optimize_cumulative_slo_violation_time,
                    slo_violation_amount_threshold_s=args.slo_violation_amount_threshold_s,
                )

                if not args.use_v2:
                    selector.solve()
                else:
                    selector.solve_v2()
                end = datetime.now().timestamp()
                print(
                    f"\tDone after {(end-start):.2f} seconds, at selector run ID: {selector._run_id}"
                )
                curr += 1

                # Update bookkeeping
                bookkeeping["runs"].append(
                    {
                        "pct_heavy": pct_heavy,
                        "mean_interarrival": mean_interarrival,
                        "rpu": rpu,
                        "workload_name": workload_name,
                        "cluster_name": cluster_name,
                        "selector_run_id": selector._run_id,
                        "start_timestamp": start,
                        "end_timestamp": end,
                    }
                )

                with open(bookkeeping_filename, "w") as f:
                    yaml.dump(bookkeeping, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy script for testing selector benchmarking on traces."
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        help="The ID of the Iconq model to use.",
        required=True,
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
        "--use_stage_for_isolated_queries",
        type=bool,
        default=True,
        help="Whether to use the StageModel for isolated queries.",
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
        "--init_from_trace",
        type=bool,
        default=False,
        help="Whether to initialize the selector from the trace.",
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
        "--use_v2",
        action="store_true",
        help="Whether to use the v2 selector implementation.",
    )
    parser.add_argument(
        "--continue_runs", 
        action='store_true', 
        help='Whether to continue from an existing collection of runs'
    )   
    args = parser.parse_args()
    main(args)
