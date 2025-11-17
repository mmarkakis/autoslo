import argparse
import asyncio
import os
from datetime import datetime
import shutil

import autoslo.utils.paths as pu
from autoslo.workload_execution.query_runner import QueryRunner
from autoslo.workload_execution.collect_stats import StatsCollector
import yaml

NUM_TEMPLATES_OPTIONS = [99]
PCT_HEAVY_OPTIONS = [0, 10, 25, 50]
MEAN_INTERARRIVAL_TIME_S_OPTIONS = [120, 60, 30, 10]


async def run_all_chunk_workloads(base_args: argparse.Namespace):
    """
    Run all chunk workloads sequentially with a 10-minute pause between each run.

    Parameters:
        base_args: Base arguments to pass to each QueryRunner instance.
    """

    # Start by running just the example workload a couple of times to get the workgroup to resume.
    example_workload_path = os.path.join(
        pu.DATA_PATH, "benchmarking_workloads", "benchmarking_workload_1_1_5.parquet"
    )
    example_args = argparse.Namespace(
        **vars(base_args), trace_path=example_workload_path
    )
    print(f"{datetime.now()} Running example workload {example_workload_path}...")
    await QueryRunner(example_args).run()
    print(f"{datetime.now()} Sleeping for 2 minutes...")
    await asyncio.sleep(2 * 60)
    print(
        f"{datetime.now()} Running example workload {example_workload_path} again..."
    )
    await QueryRunner(example_args).run()
    print(f"{datetime.now()} Sleeping for 5 minutes...")
    await asyncio.sleep(5 * 60)

    if not base_args.test_run:
        # Now run all the chunk workloads.
        chunk_ids = []
        run_ids = []
        for num_templates in NUM_TEMPLATES_OPTIONS:
            for pct_heavy in PCT_HEAVY_OPTIONS:
                for (
                    mean_interarrival_time_s
                ) in MEAN_INTERARRIVAL_TIME_S_OPTIONS:
                    chunk_id = f"tpcds_{num_templates}templates_{pct_heavy:02d}pctheavy_{mean_interarrival_time_s:02d}meaninterarrivals"
                    chunk_ids.append(chunk_id)
                    chunk_workload_path = os.path.join(
                        pu.DATA_PATH, "chunks", f"{chunk_id}", "chunk_workload.parquet"
                    )
                    print(
                        f"{datetime.now()} Running chunk workload {chunk_workload_path}..."
                    )
                    full_args = argparse.Namespace(
                        **vars(base_args), trace_path=chunk_workload_path
                    )
                    run_id = await QueryRunner(full_args).run()
                    run_ids.append(run_id)

                    print(f"{datetime.now()} Sleeping for 10 minutes...")
                    await asyncio.sleep(10 * 60)

    # Now get the statistics out as well.
    print(f"{datetime.now()} Collecting stats for all runs...")
    stats_collector = StatsCollector(
        conn_info_path=base_args.conn_info_path,
        only_endpoint=base_args.endpoint_name,
    )
    await stats_collector.collect_stats(skip_write_on_mismatch=True)
    print(f"{datetime.now()} Done collecting stats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run queries from a benchmarking workload in an open loop."
    )

    parser.add_argument(
        "--conn_info_path",
        type=str,
        default=os.path.join(pu.AUTOSLO_ROOT, "config", "conn.yml"),
        help="Path to the YAML file containing the connection info for psycopg2.",
    )
    parser.add_argument(
        "--tpcds_scale_factor",
        type=int,
        default=1000,
        help="TPC-DS scale factor to run against.",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        default="16",
        help="Endpoint name to run on.",
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
    asyncio.run(run_all_chunk_workloads(base_args))
