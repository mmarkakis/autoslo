import argparse
import asyncio
from datetime import datetime

from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_execution.run_stats_collector import RunStatsCollector
from autoslo.workload_execution.query_runner import QueryRunner

FIRST_PART_OPTIONS = [
    #"heavy", 
    "light"
    ]
SECOND_PART_OPTIONS = [
    #"compliant", 
    "disruptive_v2"
]


async def run_all_interference_workloads(base_args: argparse.Namespace):
    """
    Run all interference workloads sequentially with a 10-minute pause between each run.

    Parameters:
        base_args: Base arguments to pass to each QueryRunner instance.
    """

    run_ids = []

    # Start by running just the example workload a couple of times
    # to get the workgroup to resume.
    example_workload_name = "benchmarking_workload_1_1_5"
    example_args = argparse.Namespace(
        **vars(base_args), workload_name=example_workload_name
    )
    print(
        f"{datetime.now()} Running example workload {example_workload_name}..."
    )
    run_id = await QueryRunner(example_args).run()
    run_ids.append(run_id)
    print(f"{datetime.now()} Sleeping for 2 minutes...")
    await asyncio.sleep(2 * 60)
    print(
        f"{datetime.now()} Running example workload {example_workload_name} again..."
    )
    run_id = await QueryRunner(example_args).run()
    run_ids.append(run_id)

    if not base_args.test_run:
        # Wait for a bit to get clean stats.
        print(f"{datetime.now()} Sleeping for 5 minutes...")
        await asyncio.sleep(5 * 60)

        # Now run all the interference workloads.
        for first_part in FIRST_PART_OPTIONS:
            for second_part in SECOND_PART_OPTIONS:
                workload_name = f"interference_{first_part}_{second_part}"

                print(
                    f"{datetime.now()} Running interference workload {workload_name}..."
                )
                full_args = argparse.Namespace(
                    **vars(base_args), workload_name=workload_name
                )
                run_id = await QueryRunner(full_args).run()
                run_ids.append(run_id)

                print(f"{datetime.now()} Sleeping for 10 minutes...")
                await asyncio.sleep(10 * 60)

    # Now get the statistics out as well.
    print(f"{datetime.now()} Collecting stats for all runs...")
    stats_collector = RunStatsCollector(run_ids=run_ids)
    await stats_collector.collect_stats(skip_write_on_mismatch=True)
    print(f"{datetime.now()} Done collecting stats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run queries from an interference workload in an open loop."
    )
    parser.add_argument(
        "--tpcds_scale_factor",
        type=int,
        default=1000,
        help="TPC-DS scale factor to run against.",
    )
    parser.add_argument(
        "--blueprint_name",
        type=str,
        default="single_8",
        help="Blueprint name to run on.",
    )
    parser.add_argument(
        "--query_router_name",
        type=str,
        default="RFixed(fixed_cluster_name='cluster_8')",
        help="Name of the QueryRouter to use.",
    )
    parser.add_argument(
        "--maxconns",
        type=int,
        default=1000,
        help="Maximum number of connections in the connection pool.",
    )
    parser.add_argument(
        "--closed_loop",
        action="store_true",
        help="If set, run in closed loop (wait for each query to finish before starting the next).",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="If set, run only a short test run with a few queries.",
    )
    base_args = parser.parse_args()
    asyncio.run(run_all_interference_workloads(base_args))
