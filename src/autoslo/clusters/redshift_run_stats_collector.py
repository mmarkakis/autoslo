import logging
import os
from typing import Optional

import pandas as pd
import psycopg2

from autoslo.clusters.cluster_conn_info import ClusterConnInfo
from autoslo.workload_execution.conn_utils import ConnWithSetup
from autoslo.workload_execution.run_stats_collector import (
    SYS_EXTERNAL_QUERY_DETAIL_QUERY,
    SYS_QUERY_DETAIL_QUERY,
    SYS_QUERY_EXPLAIN_QUERY,
    SYS_QUERY_HISTORY_QUERY,
    SYS_SERVERLESS_USAGE_QUERY,
)
import autoslo.utils.paths as pu


class RedshiftRunStatsCollector:

    @staticmethod
    def collect_cluster_stats(
        cluster_name: str, conn_info: ClusterConnInfo, run_id: str
    ) -> None:
        """Stats-collector callback invoked during cluster tear-down.

        Opens a fresh connection to *cluster_name*, queries the five
        Redshift system tables used by :class:`RunStatsCollector`, and
        writes each result as a Parquet file in the current run
        directory.

        This method is synchronous and may block for a significant
        amount of time (system tables can take minutes to flush).  It
        is invoked by :meth:`ManagedClusterPool._finalize_removal`
        while the cluster is still alive.
        """
        logging.info(
            "Collecting stats for cluster %s (run %s) ...",
            cluster_name,
            run_id,
        )
        try:
            conn = psycopg2.connect(
                host=conn_info.host,
                port=conn_info.port,
                user=conn_info.user,
                password=conn_info.password,
                dbname=conn_info.dbname,
                connection_factory=lambda dsn, **kw: ConnWithSetup(
                    dsn, search_path="public", **kw
                ),
            )
        except Exception:
            logging.exception(
                "Failed to connect to %s for stats collection.",
                cluster_name,
            )
            return

        try:
            # 1. sys_query_history — anchor table.
            history_df = RedshiftRunStatsCollector._query_to_parquet(
                conn,
                SYS_QUERY_HISTORY_QUERY.format(run_id),
                "sys_query_history",
                cluster_name,
                run_id
            )
            if history_df is None or history_df.empty:
                logging.warning(
                    "No sys_query_history rows for cluster %s, run %s. "
                    "Skipping remaining system tables.",
                    cluster_name,
                    run_id,
                )
                return

            # Derive query-id and time ranges for the remaining tables.
            min_qid = int(history_df["query_id"].min())
            max_qid = int(history_df["query_id"].max())
            min_time = history_df["start_time"].min() - pd.Timedelta(minutes=1)
            max_time = history_df["end_time"].max() + pd.Timedelta(minutes=3)

            # 2–5. remaining system tables.
            for query_sql, table_name in [
                (
                    SYS_QUERY_EXPLAIN_QUERY.format(min_qid, max_qid),
                    "sys_query_explain",
                ),
                (
                    SYS_QUERY_DETAIL_QUERY.format(min_time, max_time),
                    "sys_query_detail",
                ),
                (
                    SYS_EXTERNAL_QUERY_DETAIL_QUERY.format(min_time, max_time),
                    "sys_external_query_detail",
                ),
                (
                    SYS_SERVERLESS_USAGE_QUERY.format(min_time, max_time),
                    "sys_serverless_usage",
                ),
            ]:
                RedshiftRunStatsCollector._query_to_parquet(
                    conn, query_sql, table_name, cluster_name, run_id
                )
        finally:
            try:
                conn.close()
            except Exception:
                pass

        logging.info("Stats collection complete for cluster %s.", cluster_name)

    @staticmethod
    def _query_to_parquet(
        conn,
        query: str,
        table_name: str,
        cluster_name: str,
        run_id: str
    ) -> Optional[pd.DataFrame]:
        """Execute *query*, write result as Parquet, return the DataFrame."""
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                cols = (
                    [desc[0] for desc in cur.description]
                    if cur.description
                    else []
                )
            df = pd.DataFrame(rows, columns=cols)
            out_path = os.path.join(
                pu.get_runs_path(),
                run_id,
                f"{table_name}+{cluster_name}.parquet",
            )
            df.to_parquet(out_path, index=False)
            logging.info("Wrote %d rows to %s", len(df), out_path)
            return df
        except Exception:
            logging.exception(
                "Failed to query %s for cluster %s.",
                table_name,
                cluster_name,
            )
            return None
