import argparse
import asyncio
import os
from datetime import datetime

import chunkbench.path_utils as pu
from chunkbench.execution.query_runner import QueryRunner
from chunkbench.execution.collect_stats import StatsCollector

NUM_TEMPLATES_OPTIONS = [99]
PCT_HEAVY_OPTIONS = [0, 10, 25, 50]
MEAN_INTERARRIVAL_TIME_S_OPTIONS = [120, 60, 30, 10]


async def run_all_chunk_traces(base_args: argparse.Namespace):
    """
    Run all chunk traces sequentially with a 10-minute pause between each run.

    Parameters:
        base_args: Base arguments to pass to each QueryRunner instance.
    """

    # Start by running just the example trace a couple of times to get the workgroup to resume.
    example_trace_path = os.path.join(
        pu.DATA_PATH, "benchmarking_traces", "benchmarking_trace_1_1_5.parquet"
    )
    example_args = argparse.Namespace(**vars(base_args), trace_path=example_trace_path)
    print(f"{datetime.now()} Running example trace {example_trace_path}...")
    await QueryRunner(example_args).run()
    print(f"{datetime.now()} Sleeping for 2 minutes...")
    await asyncio.sleep(2 * 60)
    print(f"{datetime.now()} Running example trace {example_trace_path} again...")
    await QueryRunner(example_args).run()
    print(f"{datetime.now()} Sleeping for 5 minutes...")
    await asyncio.sleep(5 * 60)

    # Now run all the chunk traces.
    for num_templates in NUM_TEMPLATES_OPTIONS:
        for pct_heavy in PCT_HEAVY_OPTIONS:
            for mean_interarrival_time_s in MEAN_INTERARRIVAL_TIME_S_OPTIONS:
                chunk_id = f"tpcds_{num_templates}templates_{pct_heavy:02d}pctheavy_{mean_interarrival_time_s:02d}meaninterarrivals"
                chunk_trace_path = os.path.join(
                    pu.DATA_PATH, "chunk_traces", f"{chunk_id}.parquet"
                )
                print(f"{datetime.now()} Running chunk trace {chunk_trace_path}...")
                full_args = argparse.Namespace(
                    **vars(base_args), trace_path=chunk_trace_path
                )
                await QueryRunner(full_args).run()

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
        description="Run queries from a benchmarking trace in an open loop."
    )

    parser.add_argument(
        "--conn_info_path",
        type=str,
        default=os.path.join(pu.CHUNKBENCH_ROOT, "config", "conn.yml"),
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
    base_args = parser.parse_args()
    asyncio.run(run_all_chunk_traces(base_args))
