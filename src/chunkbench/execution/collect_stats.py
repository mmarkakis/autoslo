import argparse
import asyncio
import os
from typing import Optional

import pandas as pd
import psycopg2 as pg2
import yaml

import chunkbench.path_utils as pu
from chunkbench.execution.conn_utils import form_hostname

SYS_QUERY_HISTORY_QUERY = """
    SELECT *
    FROM sys_query_history
    WHERE query_text LIKE '--{}%'
    ORDER BY start_time ASC;
"""


class StatsCollector:

    def __init__(
        self,
        conn_info_path: str,
        force_starting_at: Optional[str] = None,
        only_endpoint: Optional[str] = None,
    ):
        """
        Initialize the StatsCollector with connection information.

        Parameters:
            conn_info_path: Path to the YAML file containing the connection info for psycopg2.
            force_starting_at: If specified, recollect stats for runs starting at this run ID (inclusive).
            only_endpoint: If specified, only collect stats for runs that used this endpoint name.
        """

        self.conn_info_path = conn_info_path
        self.force_starting_at = force_starting_at
        self.only_endpoint = only_endpoint

        # Validate connection info.
        if not os.path.exists(self.conn_info_path):
            raise FileNotFoundError(
                f"Connection info file {self.conn_info_path} does not exist."
            )
        with open(self.conn_info_path, "r") as f:
            self.all_conn_info = yaml.safe_load(f)

        # Compile the run information.
        self.compile_run_information()

    def compile_run_information(self):
        """
        Go through the data directory and compile information about the runs.
        """
        self.run_params = {}
        for run_dir in os.listdir(os.path.join(pu.DATA_PATH, "runs")):
            run_path = os.path.join(pu.DATA_PATH, "runs", run_dir)
            if not os.path.isdir(run_path):
                continue

            has_range_and_matches = (self.force_starting_at is not None) and (
                run_dir >= self.force_starting_at
            )

            no_range_and_missing = (self.force_starting_at is None) and not any(
                fname.endswith(".parquet") for fname in os.listdir(run_path)
            )

            if has_range_and_matches or no_range_and_missing:
                # Store the relevant information for later processing.
                with open(os.path.join(run_path, "run_params.yml"), "r") as f:
                    run_params = yaml.safe_load(f)

                if (self.only_endpoint is None) or (
                    run_params["endpoint_name"] == self.only_endpoint
                ):
                    self.run_params[run_dir] = run_params

    async def collect_stats(self, skip_write_on_mismatch: bool = False):
        for run_id, run_params in self.run_params.items():
            print(f"Collecting stats for run {run_id}...")
            # Open connection.
            endpoint_name = run_params["endpoint_name"]
            if endpoint_name not in self.all_conn_info["endpoints"]:
                print(
                    f"Endpoint name {endpoint_name} not found in connection info. Skipping."
                )
                continue
            conn_info = self.all_conn_info["endpoints"][endpoint_name]
            d = {
                "dbname": conn_info["dbname"],
                "user": conn_info["user"],
                "password": conn_info["password"],
                "port": conn_info["port"],
                "host": form_hostname(
                    conn_info["workgroup_name"],
                    self.all_conn_info["aws_account_id"],
                    self.all_conn_info["aws_region"],
                ),
            }
            conn = pg2.connect(**d)

            # Query the sys_query_history table.
            cur = conn.cursor()
            query = SYS_QUERY_HISTORY_QUERY.format(run_id)
            print(f"\tExecuting query:\n{query}")
            cur.execute(query)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            cur.close()
            conn.close()
            num_rows = len(df)
            print(f"\tRetrieved {num_rows} rows from sys_query_history.")
            mismatch = num_rows != run_params["num_queries"]
            if mismatch:
                print(
                    f"\tNumber of rows {num_rows} does not match expected {run_params['num_queries']}."
                )

            # Write out the stats.
            if not (skip_write_on_mismatch and mismatch):
                out_path = os.path.join(
                    pu.DATA_PATH, "runs", run_id, "sys_query_history.parquet"
                )
                df.to_parquet(out_path, index=False)
                print(f"\tWrote stats to {out_path}.")
            else:
                print(f"\tSkipping write due to mismatch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the execution statistics for any runs without stats."
    )
    parser.add_argument(
        "--conn_info_path",
        type=str,
        default=os.path.join(pu.CHUNKBENCH_ROOT, "config", "conn.yml"),
        help="Path to the YAML file containing the connection info for psycopg2.",
    )
    parser.add_argument(
        "--force_starting_at",
        type=str,
        default=None,
        help="If specified, recollect stats for runs starting at this run ID (inclusive).",
    )
    parser.add_argument(
        "--only_endpoint",
        type=str,
        default=None,
        help="If specified, only collect stats for runs that used this endpoint name.",
    )
    parser.add_argument(
        "--skip_write_on_mismatch",
        action="store_true",
        help="If set, skip writing out stats if the number of collected rows does not match the expected number of queries.",
    )
    args = parser.parse_args()
    collector = StatsCollector(
        args.conn_info_path, args.force_starting_at, args.only_endpoint
    )
    asyncio.run(collector.collect_stats(args.skip_write_on_mismatch))
