import argparse
import asyncio
import os

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

    def __init__(self, conn_info_path: str):
        """
        Initialize the StatsCollector with connection information.

        Parameters:
            conn_info_path: Path to the YAML file containing the connection info for psycopg2.
        """

        self.conn_info_path = conn_info_path

        # Validate connection info.
        if not os.path.exists(self.conn_info_path):
            raise FileNotFoundError(
                f"Connection info file {self.conn_info_path} does not exist."
            )
        with open(args.conn_info_path, "r") as f:
            self.all_conn_info = yaml.safe_load(f)

        # Compile the run information.
        self.compile_run_information()

    def compile_run_information(self):
        """
        Go through the data directory and compile information about the runs without stats.
        """
        self.run_params = {}
        for run_dir in os.listdir(os.path.join(pu.DATA_PATH, "runs")):
            run_path = os.path.join(pu.DATA_PATH, "runs", run_dir)
            if not os.path.isdir(run_path):
                continue

            # Check if stats files exist and skip. They should be any parquet files.
            if any(fname.endswith(".parquet") for fname in os.listdir(run_path)):
                continue

            # Store the relevant information for later processing.
            with open(os.path.join(run_path, "run_params.yml"), "r") as f:
                self.run_params[run_dir] = yaml.safe_load(f)

    async def collect_stats(self):
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
            cur.execute(SYS_QUERY_HISTORY_QUERY.format(run_id))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            cur.close()
            conn.close()
            num_rows = len(df)
            print(f"\tRetrieved {num_rows} rows from sys_query_history.")
            if num_rows != run_params["num_queries"]:
                print(
                    f"\tNumber of rows {num_rows} does not match expected {run_params['num_queries']}."
                )

            # Write out the stats.
            out_path = os.path.join(
                pu.DATA_PATH, "runs", run_id, "sys_query_history.parquet"
            )
            df.to_parquet(out_path, index=False)
            print(f"\tWrote stats to {out_path}.")


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

    args = parser.parse_args()
    collector = StatsCollector(args.conn_info_path)
    asyncio.run(collector.collect_stats())
