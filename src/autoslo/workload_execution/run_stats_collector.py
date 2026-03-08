import argparse
import asyncio
import os

import pandas as pd
import psycopg2 as pg2
import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from autoslo.workload_execution.conn_utils import ConnWithSetup

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


class RunStatsCollector:

    def __init__(
        self,
        run_ids: list[str],
        force: bool = False,
    ):
        """
        Initialize the StatsCollector.

        Parameters:
            run_ids: List of run IDs to collect stats for.
            force: If True, force recollection of stats for the specified runs,
                even if they already exist.
        """
        self.run_ids = run_ids
        self.force = force

        # Validate connection info.
        self.conn_info_path = pu.get_conn_info_path()
        with open(self.conn_info_path, "r") as f:
            self.all_conn_info = yaml.safe_load(f)

        # Compile the run information.
        self.compile_run_information()

    def compile_run_information(self):
        """
        Go through the data directory and compile information about the runs.
        """
        self.run_params = {}
        for run_id in self.run_ids:
            run_path = os.path.join(pu.get_runs_path(), run_id)
            if not os.path.isdir(run_path):
                print(f"Run path {run_path} does not exist, skipping.")
                continue

            # Determine if we should collect stats for this run.
            should_collect = (self.force) or not any(
                (fname.startswith('sys') and fname.endswith(".parquet")) for fname in os.listdir(run_path)
            )
            if not should_collect:
                print(f"Stats already exist for run {run_id}, skipping.")
                continue

            # Store the relevant information for later processing.
            with open(os.path.join(run_path, "run_params.yml"), "r") as f:
                self.run_params[run_id] = yaml.safe_load(f)

    async def run_one_and_write_out(
        self,
        conn: pg2.extensions.connection,
        query: str,
        run_id: str,
        table_name: str,
        cluster_name: str,
    ) -> tuple[pd.DataFrame, int]:
        """
        Run a single query and return the results as a DataFrame.

        Parameters:
            conn: An open psycopg2 connection.
            query: The SQL query to execute.
            run_id: The ID of the run.
            table_name: The name of the statistics table.
            cluster_name: The name of the cluster.

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
        df = pd.DataFrame(rows, columns=cols)

        # Write out the DataFrame to a parquet file.
        out_path = os.path.join(
            pu.get_runs_path(), run_id, f"{table_name}+{cluster_name}.parquet"
        )
        df.to_parquet(out_path, index=False)
        print(f"\tWrote {table_name} to {out_path}.")

        return df, num_rows

    async def collect_stats(self, skip_write_on_mismatch: bool = False):
        bp_configs = pu.get_blueprint_dicts_from_config()
        cluster_configs = pu.get_cluster_dicts_from_config()

        for run_id, run_params in self.run_params.items():
            print(f"Collecting stats for run {run_id}...")

            bp_name = run_params["blueprint_name"]
            cluster_names = bp_configs[bp_name]["cluster_names"]

            for cluster_name in cluster_names:
                # Build connection from config.
                ci = ClusterConnInfo.from_dict(cluster_configs[cluster_name])
                conn = pg2.connect(
                    host=ci.host,
                    port=ci.port,
                    user=ci.user,
                    password=ci.password,
                    dbname=ci.dbname,
                    connection_factory=lambda dsn, **kw: ConnWithSetup(
                        dsn, search_path="public", **kw
                    ),
                )

                # Query sys_query_history and write out the results.
                sys_query_history_df, sys_query_history_df_len = (
                    await self.run_one_and_write_out(
                        conn,
                        SYS_QUERY_HISTORY_QUERY.format(run_id),
                        run_id,
                        "sys_query_history",
                        cluster_name,
                    )
                )

                # Derive the query_id and start_time ranges for further queries.
                if sys_query_history_df_len == 0:
                    print(
                        f"\tNo rows found, skipping further stats collection."
                    )
                    continue
                min_query_id = sys_query_history_df["query_id"].min()
                max_query_id = sys_query_history_df["query_id"].max()
                min_time = sys_query_history_df[
                    "start_time"
                ].min() - pd.Timedelta(minutes=1)
                max_time = sys_query_history_df[
                    "end_time"
                ].max() + pd.Timedelta(minutes=3)

                # Query the rest of the stats tables and write out the results.
                await self.run_one_and_write_out(
                    conn,
                    SYS_QUERY_EXPLAIN_QUERY.format(min_query_id, max_query_id),
                    run_id,
                    "sys_query_explain",
                    cluster_name,
                )
                await self.run_one_and_write_out(
                    conn,
                    SYS_QUERY_DETAIL_QUERY.format(min_time, max_time),
                    run_id,
                    "sys_query_detail",
                    cluster_name,
                )
                await self.run_one_and_write_out(
                    conn,
                    SYS_EXTERNAL_QUERY_DETAIL_QUERY.format(min_time, max_time),
                    run_id,
                    "sys_external_query_detail",
                    cluster_name,
                )
                await self.run_one_and_write_out(
                    conn,
                    SYS_SERVERLESS_USAGE_QUERY.format(min_time, max_time),
                    run_id,
                    "sys_serverless_usage",
                    cluster_name,
                )

                # Close the connection.
                conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the execution statistics for any runs without stats."
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        nargs="+",
        required=True,
        help="List of run IDs to collect stats for.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, force recollection of stats even if they already exist.",
    )
    parser.add_argument(
        "--skip_write_on_mismatch",
        action="store_true",
        help=(
            "If set, skip writing out stats if the number of collected rows "
            "does not match the expected number of queries."
        ),
    )
    args = parser.parse_args()
    collector = RunStatsCollector(run_ids=args.run_ids, force=args.force)
    asyncio.run(collector.collect_stats(args.skip_write_on_mismatch))
