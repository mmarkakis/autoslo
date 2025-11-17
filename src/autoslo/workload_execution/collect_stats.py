import argparse
import asyncio
import os
from typing import Optional

import pandas as pd
import psycopg2 as pg2
import yaml

import autoslo.utils.paths as pu
from autoslo.execution.conn_utils import form_hostname

SYS_QUERY_HISTORY_QUERY = """
    SELECT *
    FROM sys_query_history
    WHERE query_text LIKE '--{}%'
    ORDER BY start_time ASC;
"""

SYS_QUERY_EXPLAIN_QUERY = """
    SELECT *
    FROM sys_query_explain
    WHERE query_id BETWEEN {} AND {};
"""

SYS_QUERY_DETAIL_QUERY = """
    SELECT *
    FROM sys_query_detail
    WHERE start_time BETWEEN '{}' AND '{}';
"""

SYS_EXTERNAL_QUERY_DETAIL_QUERY = """
    SELECT *
    FROM sys_external_query_detail
    WHERE start_time BETWEEN '{}' AND '{}';
"""

SYS_SERVERLESS_USAGE_QUERY = """
    SELECT *
    FROM sys_serverless_usage
    WHERE start_time BETWEEN '{}' AND '{}';
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

    async def run_one_query(
        self, conn: pg2.extensions.connection, query: str
    ) -> tuple[pd.DataFrame, int]:
        """
        Run a single query and return the results as a DataFrame.

        Parameters:
            conn: An open psycopg2 connection.
            query: The SQL query to execute.

        Returns:
            df: A DataFrame containing the query results.
            num_rows: The number of rows returned by the query.
        """
        with conn.cursor() as cur:
            print(f"\tExecuting query:\n{query}")
            cur.execute(query)
            rows = cur.fetchall()
            cols = (
                [desc[0] for desc in cur.description] if cur.description else []
            )
        num_rows = len(rows)
        print(f"\tRetrieved {num_rows} rows.")
        return pd.DataFrame(rows, columns=cols), num_rows

    def write_out_table(self, run_id, df: pd.DataFrame, table_name: str):
        """
        Write out a DataFrame to a parquet file.

        Parameters:
            run_id: The ID of the run.
            df: The DataFrame to write.
            table_name: The name of the table/file.
        """
        out_path = os.path.join(
            pu.DATA_PATH, "runs", run_id, f"{table_name}.parquet"
        )
        df.to_parquet(out_path, index=False)
        print(f"\tWrote {table_name} to {out_path}.")

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

            # Query sys_query_history and possibly write out the results.
            sys_query_history_df, sys_query_history_df_len = (
                await self.run_one_query(
                    conn, SYS_QUERY_HISTORY_QUERY.format(run_id)
                )
            )
            mismatch = sys_query_history_df_len != run_params["num_queries"]
            if mismatch:
                print(
                    f"\tNumber of rows {sys_query_history_df_len} does not "
                    f"match expected {run_params['num_queries']}."
                )
            if not (skip_write_on_mismatch and mismatch):
                self.write_out_table(
                    run_id, sys_query_history_df, "sys_query_history"
                )
            else:
                print(f"\tSkipping write due to mismatch.")
                continue

            # Derive the query_id and start_time ranges for further queries.
            if sys_query_history_df_len == 0:
                print(
                    f"\tNo rows retrieved, skipping further stats collection."
                )
                continue
            min_query_id = sys_query_history_df["query_id"].min()
            max_query_id = sys_query_history_df["query_id"].max()
            min_time = sys_query_history_df["start_time"].min() - pd.Timedelta(
                minutes=1
            )
            max_time = sys_query_history_df["end_time"].max() + pd.Timedelta(
                minutes=3
            )

            # Query sys_query_explain and write out the results.
            sys_query_explain_df, _ = await self.run_one_query(
                conn,
                SYS_QUERY_EXPLAIN_QUERY.format(min_query_id, max_query_id),
            )
            self.write_out_table(
                run_id, sys_query_explain_df, "sys_query_explain"
            )

            # Query sys_query_detail and write out the results.
            sys_query_detail_df, _ = await self.run_one_query(
                conn,
                SYS_QUERY_DETAIL_QUERY.format(min_time, max_time),
            )
            self.write_out_table(
                run_id, sys_query_detail_df, "sys_query_detail"
            )

            # Query sys_external_query_detail and write out the results.
            sys_external_query_detail_df, _ = await self.run_one_query(
                conn,
                SYS_EXTERNAL_QUERY_DETAIL_QUERY.format(min_time, max_time),
            )
            self.write_out_table(
                run_id,
                sys_external_query_detail_df,
                "sys_external_query_detail",
            )

            # Query sys_serverless_usage and write out the results.
            sys_serverless_usage_df, _ = await self.run_one_query(
                conn,
                SYS_SERVERLESS_USAGE_QUERY.format(min_time, max_time),
            )
            self.write_out_table(
                run_id, sys_serverless_usage_df, "sys_serverless_usage"
            )

            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the execution statistics for any runs without stats."
    )
    parser.add_argument(
        "--conn_info_path",
        type=str,
        default=os.path.join(pu.AUTOSLO_ROOT, "config", "conn.yml"),
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
