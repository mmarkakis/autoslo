import argparse
import asyncio
import os
from datetime import datetime
import shutil

import chunkload.utils.paths as pu
from chunkload.execution.query_runner import QueryRunner
from chunkload.execution.collect_stats import StatsCollector
import yaml

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
    example_args = argparse.Namespace(
        **vars(base_args), trace_path=example_trace_path
    )
    print(f"{datetime.now()} Running example trace {example_trace_path}...")
    await QueryRunner(example_args).run()
    print(f"{datetime.now()} Sleeping for 2 minutes...")
    await asyncio.sleep(2 * 60)
    print(
        f"{datetime.now()} Running example trace {example_trace_path} again..."
    )
    await QueryRunner(example_args).run()
    print(f"{datetime.now()} Sleeping for 5 minutes...")
    await asyncio.sleep(5 * 60)

    # Now run all the chunk traces.
    chunk_ids = []
    run_ids = []
    for num_templates in NUM_TEMPLATES_OPTIONS:
        for pct_heavy in PCT_HEAVY_OPTIONS:
            for mean_interarrival_time_s in MEAN_INTERARRIVAL_TIME_S_OPTIONS:
                chunk_id = f"tpcds_{num_templates}templates_{pct_heavy:02d}pctheavy_{mean_interarrival_time_s:02d}meaninterarrivals"
                chunk_ids.append(chunk_id)
                chunk_workload_path = os.path.join(
                    pu.DATA_PATH, f"{chunk_id}", "chunk_workload.parquet"
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

    # Finally, for each run, copy the stats to the correct chunk directory.
    distribute_stats_to_chunks(chunk_ids, base_args.endpoint_name)


def distribute_stats_to_chunks(chunk_ids: list[str], endpoint_name: str):
    """
    For each chunk, copy the stats from the most recent run on the given
    endpoint to the chunk directory.

    Parameters:
        chunk_ids: List of chunk IDs to distribute stats for.
        endpoint_name: Name of the endpoint to filter stats by.
    """

    for chunk_id in chunk_ids:
        # Find the most recent run for this chunk on the given endpoint.
        run_id = pu.RunLocator.get_run_id(
            trace_path=chunk_id, endpoint_name=endpoint_name
        )
        if not run_id:
            print(
                f"No run found for chunk {chunk_id} on endpoint "
                f"{endpoint_name}, skipping."
            )
            continue
        run_id = run_id[-1]

        # Copy the sys_query_history.parquet file to the chunk directory.
        sys_query_history_path = os.path.join(
            pu.RUNS_PATH, run_id, "sys_query_history.parquet"
        )
        chunk_dir = os.path.join(pu.DATA_PATH, "chunks", chunk_id)
        shutil.copy(
            sys_query_history_path,
            os.path.join(
                chunk_dir, f"sys_query_history_{endpoint_name}.parquet"
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run queries from a benchmarking trace in an open loop."
    )

    parser.add_argument(
        "--conn_info_path",
        type=str,
        default=os.path.join(pu.CHUNKLOAD_ROOT, "config", "conn.yml"),
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
