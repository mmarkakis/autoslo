import argparse
import decimal
import os
from datetime import datetime
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd
import psycopg2 as pg2
import yaml
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster import Cluster

TABLE_STATS_QUERY = """
SELECT
    ti.table AS table_name,
    ti.size AS num_blocks,
    ti.tbl_rows AS num_rows,
    COUNT(a.attname) AS num_columns
FROM 
    svv_table_info ti
LEFT JOIN 
    pg_attribute a ON a.attrelid = ti.table_id
    AND a.attnum > 0
WHERE 
    ti.schema = '{schema_name}'  
GROUP BY 
    ti.table,ti.size, ti.tbl_rows
ORDER BY 
    ti.table;
"""
COLUMN_STATS_QUERY_1 = """
SELECT
    pgs.tablename AS table_name,
    pgs.attname AS column_name,
    pgs.null_frac AS null_fraction,
    pgs.avg_width AS avg_width,
    pgs.n_distinct AS num_distinct,
    pgs.correlation AS correlation
FROM 
    pg_stats AS pgs
WHERE
    pgs.attname NOT IN ('insertxid', 'deletexid') AND
    pgs.schemaname = '{schema_name}'
ORDER BY
    pgs.tablename ASC, pgs.attname ASC
;
"""
COLUMN_STATS_QUERY_2 = """
SELECT 
    c.table_name AS table_name,
    c.column_name AS column_name,
    COALESCE(c.character_maximum_length, c.numeric_precision) AS max_width,
    c.data_type as data_type,
    CASE
        WHEN 
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                sortkey1, 
                '^AUTO[(]SORTKEY', 
                ''
            ), 
            '[()]', 
            ''
        ) = column_name THEN 1
        ELSE 0
    END AS is_sorted,
    ti.tbl_rows as num_rows
FROM 
    svv_columns AS c
JOIN
    svv_table_info AS ti 
    ON c.table_catalog = ti.database AND c.table_schema = ti.schema AND c.table_name = ti.table
WHERE 
    c.table_schema = '{schema_name}'
ORDER BY
    c.table_name ASC, c.column_name ASC
;
"""
COLUMN_NUMERIC_STATS_QUERY = """
SELECT 
    AVG({column_name}) AS mean,
    PERCENTILE_CONT(0) WITHIN GROUP (ORDER BY {column_name}) AS p0,
    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY {column_name}) AS p10,
    PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY {column_name}) AS p20,
    PERCENTILE_CONT(0.3) WITHIN GROUP (ORDER BY {column_name}) AS p30,
    PERCENTILE_CONT(0.4) WITHIN GROUP (ORDER BY {column_name}) AS p40,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name}) AS p50,
    PERCENTILE_CONT(0.6) WITHIN GROUP (ORDER BY {column_name}) AS p60,
    PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY {column_name}) AS p70,
    PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY {column_name}) AS p80,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY {column_name}) AS p90,
    PERCENTILE_CONT(1.0) WITHIN GROUP (ORDER BY {column_name}) AS p100
FROM 
    {schema_name}.{table_name}
;
"""
COLUMN_STRING_STATS_QUERY = """ 
    SELECT 
        {column_name} AS value,
        COUNT(*) AS occ_count
    FROM 
        {schema_name}.{table_name}
    GROUP BY 
        {column_name}
    HAVING
        occ_count > {min_string_occ_count}
    ORDER BY 
        occ_count DESC
    LIMIT {max_string_vals};
"""

CATEGORICAL_THRESHOLD = 10000
MAX_STRING_VALS = 100
MIN_STRING_OCC_FREQUENCY = 0.001


class RedshiftColType(Enum):
    """
    An enumeration of the possible data types for a column in Redshift.
    """

    BIGINT = "bigint"
    VARCHAR = "character varying"

    def __str__(self):
        return self.value


class DBStatsCollector:

    def __init__(
        self,
        cluster_name: str,
        schema_name: str,
        analyze: bool = False,
        force: bool = False,
    ) -> None:
        """
        Initialize the DBStatsCollector.

        Parameters:
            cluster_name: The name of the cluster to connect to.
            schema_name: The name of the database to collect stats for.
            analyze: Whether to run ANALYZE on the database before collecting
                stats.
            force: Whether to force re-collection of statistics (per combination
                of cluster and database) even if they already exist.
        """
        self.cluster_name = cluster_name
        self.schema_name = schema_name
        self.analyze = analyze
        self.force = force
        self.creation_time = datetime.now()

    def query_to_dict(
        self,
        conn: pg2.extensions.connection,
        query: str,
    ) -> list[dict[str, Any]]:
        """
        Run a single query and return the results as a dictionary.

        Parameters:
            conn: An open psycopg2 connection.
            query: The SQL query to execute.

        Returns:
            A list of dictionaries containing the query results, where each
                dictionary corresponds to a row with column names as keys.
        """

        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            cols = (
                [desc[0] for desc in cur.description] if cur.description else []
            )
        results = [dict(zip(cols, row)) for row in rows]

        # Cast any Decimal types to float for YAML serialization.
        for row in results:
            for key, value in row.items():
                if isinstance(value, decimal.Decimal):
                    row[key] = float(value)

        return results

    def query_to_df(
        self,
        conn: pg2.extensions.connection,
        query: str,
    ) -> pd.DataFrame:
        """
        Run a single query and return the results as a DataFrame.

        Parameters:
            conn: An open psycopg2 connection.
            query: The SQL query to execute.

        Returns:
            A DataFrame containing the query results.
        """
        results_dict = self.query_to_dict(conn, query)
        return pd.DataFrame(results_dict)

    def collect_stats(self) -> None:
        """
        Get the table and column statistics and save them to a yml file.
        """

        # Check whether the intended output already exists.
        stats_path = os.path.join(
            pu.get_data_path(),
            "db_stats",
            f"{self.cluster_name}_{self.schema_name}.yml",
        )
        if os.path.exists(stats_path) and not self.force:
            return

        # Create a cluster and open a connection.
        cluster = Cluster.from_config(self.cluster_name)
        conn = cluster.conn_pool(
            minconn=1, maxconn=1, search_path=self.schema_name
        ).getconn()

        # Run an analyze if requested.
        if self.analyze:
            with conn.cursor() as cur:
                print(
                    f"{datetime.now()} Running ANALYZE on schema {self.schema_name} ..."
                )
                cur.execute("ANALYZE;")
                print(f"{datetime.now()} \tANALYZE completed.")

        # Get the table and column statistics.
        table_stats = self.get_table_stats(conn)
        column_stats = self.get_column_stats(conn)

        # Dump everything to a yaml file.
        d = {
            "cluster_name": self.cluster_name,
            "schema_name": self.schema_name,
            "analyze": self.analyze,
            "collection_time": self.creation_time.isoformat(),
            "collection_timestamp": int(self.creation_time.timestamp()),
            "table_stats": table_stats,
            "column_stats": column_stats,
        }
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w") as f:
            yaml.safe_dump(d, f)

    def get_table_stats(
        self,
        conn: pg2.extensions.connection,
    ) -> dict[str, dict[str, Any]]:
        """
        Get the table statistics from the database.

        Parameters:
            conn: The connection to the database.

        Returns:
            table_stats: The table statistics dictionary.
        """
        print(
            f"{datetime.now()} Getting table stats for schema {self.schema_name} ..."
        )
        results_dict = self.query_to_dict(
            conn, TABLE_STATS_QUERY.format(schema_name=self.schema_name)
        )
        table_stats = {row["table_name"]: row for row in results_dict}
        print(
            f"{datetime.now()} \tTable stats obtained for {len(table_stats)} tables."
        )
        return table_stats

    def get_column_stats(
        self,
        conn: pg2.extensions.connection,
    ) -> dict[str, dict[str, Any]]:
        """
        Recover the column statistics from the database.

        Parameters:
            conn: The connection to the database.

        Returns:
            column_stats: The column statistics dictionary.
        """
        print(
            f"{datetime.now()} Getting column stats for schema {self.schema_name} ..."
        )
        # Run the two queries to get the base column stats and join them.
        # column_stats_1 comes from pg_stats which lacks some columns, thus
        # right join.
        column_stats_1 = self.query_to_df(
            conn, COLUMN_STATS_QUERY_1.format(schema_name=self.schema_name)
        )
        column_stats_2 = self.query_to_df(
            conn, COLUMN_STATS_QUERY_2.format(schema_name=self.schema_name)
        )
        column_stats_df = column_stats_1.merge(
            column_stats_2, on=["table_name", "column_name"], how="right"
        )
        column_stats = (
            column_stats_df.sort_values(by=["table_name", "column_name"])
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

        # Derive additional information depending on the type of column.
        print(f"{datetime.now()} \tGetting detailed column stats ...")
        for i, row_dict in tqdm(
            enumerate(column_stats), total=len(column_stats)
        ):
            data_type = row_dict["data_type"]

            if data_type == RedshiftColType.BIGINT.value:
                column_numeric_stats = self.query_to_dict(
                    conn,
                    COLUMN_NUMERIC_STATS_QUERY.format(
                        schema_name=self.schema_name,
                        table_name=row_dict["table_name"],
                        column_name=row_dict["column_name"],
                    ),
                )
                assert len(column_numeric_stats) == 1
                column_stats[i] = row_dict | column_numeric_stats[0]

            elif data_type == RedshiftColType.VARCHAR.value:
                min_string_occ_count = int(
                    MIN_STRING_OCC_FREQUENCY * float(row_dict["num_rows"])
                )
                column_string_stats = self.query_to_dict(
                    conn,
                    COLUMN_STRING_STATS_QUERY.format(
                        schema_name=self.schema_name,
                        table_name=row_dict["table_name"],
                        column_name=row_dict["column_name"],
                        min_string_occ_count=str(min_string_occ_count),
                        max_string_vals=str(MAX_STRING_VALS),
                    ),
                )
                row_dict["common_string_vals_frequencies"] = {
                    r["value"]: r["occ_count"] / row_dict["num_rows"]
                    for r in column_string_stats
                }

            # Replace nulls, NaNs, and infinities with -1.
            for key, value in column_stats[i].items():
                if value is None:
                    column_stats[i][key] = -1
                elif isinstance(value, float):
                    if pd.isna(value) or pd.isnull(value) or np.isinf(value):
                        column_stats[i][key] = -1

        # Pivot the nesting to be table_name -> column_name -> stats.
        print(f"{datetime.now()} \tStructuring column stats ...")
        column_stats_dict: dict[str, dict[str, dict[str, Any]]] = {}
        for row_dict in column_stats:
            table_name = row_dict["table_name"]
            column_name = row_dict["column_name"]
            if table_name not in column_stats_dict:
                column_stats_dict[table_name] = {}
            del row_dict["table_name"]
            del row_dict["column_name"]
            column_stats_dict[table_name][column_name] = cast(
                dict[str, Any], row_dict
            )

        print(f"{datetime.now()} \tColumn stats obtained.")
        return column_stats_dict


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Run ANALYZE on the database",
    )
    parser.add_argument(
        "--scale_factor",
        "-s",
        type=int,
        help=("The scale factor to collect stats for."),
        default=1000,
    )
    parser.add_argument(
        "--cluster_name",
        "-c",
        type=str,
        help=("Which cluster to collect stats for. "),
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help=(
            "If set, force re-collection of statistics (per combination of "
            "cluster and database) even if they already exist."
        ),
    )
    args = parser.parse_args()

    # Determine cluster and scale factor to get stats for.
    with open(pu.get_conn_info_path(), "r") as f:
        config = yaml.safe_load(f)
    scale_factors_dict = config["scale_factors"]
    if args.scale_factor not in scale_factors_dict:
        raise ValueError(
            f"Scale factor {args.scale_factor} not found in configuration."
        )
    schema_name = scale_factors_dict[args.scale_factor]

    clusters_dict = config["clusters"]
    if args.cluster_name not in clusters_dict:
        raise ValueError(
            f"Cluster name {args.cluster_name} not found in configuration."
        )
    cluster_name = args.cluster_name

    # Collect stats for the specified cluster and scale factor.
    collector = DBStatsCollector(
        cluster_name=cluster_name,
        schema_name=schema_name,
        analyze=args.analyze,
    )
    collector.collect_stats()
