import argparse
import os
from datetime import datetime

import yaml

import autoslo.utils.paths as pu
from autoslo.workload_execution.workload_simulator import (
    WorkloadSimulator,
)
from autoslo.blueprints.blueprint import Blueprint


def main(args):

    pct_heavy_options = [0, 10, 25, 50]
    mean_interarrival_options = [120, 60, 30, 10]

    total = len(pct_heavy_options) * len(mean_interarrival_options)
    curr = 0

    blueprint = Blueprint.maximal(max_rpu=args.max_rpu)

    bookkeeping_filename = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "15_simulator_benchmarking",
        "simulator_benchmarking_bookkeeping.yml",
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
            )
            for run in existing_runs
        )
    else:
        bookkeeping = {
            "iconq_model_id": args.iconq_model_id,
            "slo_s": args.slo_s,
            "blueprint_max_rpu": args.max_rpu,
            "blueprint_name": blueprint.name,
            "slo_violation_rate_threshold": args.slo_violation_rate_threshold,
            "export_video": args.export_video,
            "video_frame_duration": args.video_frame_duration,
            "optimize_cumulative_slo_violation_time": args.optimize_cumulative_slo_violation_time,
            "slo_violation_amount_threshold_s": args.slo_violation_amount_threshold_s,
            "runs": [],
        }

        with open(bookkeeping_filename, "w") as f:
            yaml.dump(bookkeeping, f, sort_keys=False)

    blueprint = Blueprint.maximal(max_rpu=32)

    for mean_interarrival in mean_interarrival_options:
        for pct_heavy in pct_heavy_options:

            workload_name = f"tpcds_99templates_{pct_heavy:02d}pctheavy_{mean_interarrival}meaninterarrivals"

            if (
                args.continue_runs
                and (pct_heavy, mean_interarrival) in completed_set
            ):
                print(
                    f"({curr+1}/{total}) Skipping already completed run for "
                    f"workload {workload_name} with blueprint {blueprint.name}."
                )
                curr += 1
                continue

            print(
                f"({curr+1}/{total}) Running simulation on workload "
                f"{workload_name} with blueprint {blueprint.name} and IconQ "
                f"model {args.iconq_model_id}..."
            )
            start = datetime.now().timestamp()
            simulator = WorkloadSimulator(
                workload_name=workload_name,
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

            simulator.first_pass()
            end = datetime.now().timestamp()
            print(
                f"\tDone after {(end-start):.2f} seconds, at simulator run ID: {simulator._run_id}"
            )
            curr += 1

            # Update bookkeeping
            bookkeeping["runs"].append(
                {
                    "pct_heavy": pct_heavy,
                    "mean_interarrival": mean_interarrival,
                    "blueprint_name": blueprint.name,
                    "workload_name": workload_name,
                    "simulator_run_id": simulator._run_id,
                    "start_timestamp": start,
                    "end_timestamp": end,
                }
            )

            with open(bookkeeping_filename, "w") as f:
                yaml.dump(bookkeeping, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy script for testing simulator benchmarking on traces."
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
