import argparse
from autoslo.workload_execution.selector_based_runner import run_using_selector
import os
import yaml
import asyncio
import autoslo.filesystem.path_utils as pu

from datetime import datetime


def main(use_model: bool = False):

    out_path = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "08_selector_benchmarking",
        "seqnum_runner_bookkeeping.yml",
    )
    out_d: dict[str, list] = {"runs": []}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            out_d = yaml.safe_load(f)

    bookkeeping_path = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "08_selector_benchmarking",
        "selector_benchmarking_bookkeeping.yml",
    )

    with open(bookkeeping_path, "r") as f:
        bookkeeping = yaml.safe_load(f)

    total = len(bookkeeping["runs"])

    for i, run in enumerate(bookkeeping["runs"]):
        print(
            f"({i+1}/{total}) Running workload {run['workload_name']} based on selector run {run['selector_run_id']}"
        )

        # Check if such a run already exists in out_d
        already_done = False
        for completed_run in out_d["runs"]:
            if (
                completed_run["workload_name"] == run["workload_name"]
                and completed_run["use_model"] == use_model
            ):
                print(
                    f"Run for workload {run['workload_name']} with use_model={use_model} already completed, skipping."
                )
                already_done = True
                break
        if already_done:
            print("Skipping...")
            continue

        selector_run_id = run["selector_run_id"]

        arg_namespace = argparse.Namespace(
            tpcds_scale_factor=1000,
            selector_run_id=selector_run_id,
            use_model=use_model,
            maxconns=400,
        )
        start = datetime.now().timestamp()
        asyncio.run(run_using_selector(arg_namespace))
        end = datetime.now().timestamp()

        out_d["runs"].append(
            {
                "workload_name": run["workload_name"],
                "pct_heavy": run["pct_heavy"],
                "mean_interarrival": run["mean_interarrival"],
                "selector_run_id": selector_run_id,
                "use_model": use_model,
                "start_timestamp": start,
                "end_timestamp": end,
                "time_taken_s": end - start,
            }
        )
        with open(out_path, "w") as f:
            yaml.dump(out_d, f, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run workloads using selector-based routing with sequence number mapping."
    )
    parser.add_argument(
        "--use_model",
        action="store_true",
        help="Whether to use model-based routing (RModelBased) instead of sequence number routing (RSeqNum).",
    )
    args = parser.parse_args()

    main(args.use_model)
