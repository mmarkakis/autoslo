import argparse
from datetime import datetime

from psycopg2 import sql

from autoslo.blueprints.cluster import Cluster

EXTERNAL_SCHEMA = "ext_tpcds1000"
TARGET_SCHEMA = "tpcds1000"  # internal schema to create tables in


def get_external_tables(conn):
    """Return list of external table names."""
    query = """
        SELECT tablename
        FROM svv_external_tables
        WHERE schemaname = %s
        ORDER BY tablename;
    """
    with conn.cursor() as cur:
        cur.execute(query, (EXTERNAL_SCHEMA,))
        return [row[0] for row in cur.fetchall()]


def ensure_schema(conn):
    """Create target schema if needed."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(TARGET_SCHEMA)
            )
        )
    conn.commit()


def table_exists(conn, table_name):
    query = """
        SELECT 1
        FROM pg_table_def
        WHERE schemaname = %s
          AND tablename = %s
        LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(query, (TARGET_SCHEMA, table_name))
        return cur.fetchone() is not None


def materialize_table(conn, table_name):
    stmt = sql.SQL(
        """
        CREATE TABLE {}.{}
        DISTSTYLE AUTO
        SORTKEY AUTO
        AS
        SELECT *
        FROM {}.{};
    """
    ).format(
        sql.Identifier(TARGET_SCHEMA),
        sql.Identifier(table_name),
        sql.Identifier(EXTERNAL_SCHEMA),
        sql.Identifier(table_name),
    )

    with conn.cursor() as cur:
        print(f"Creating {TARGET_SCHEMA}.{table_name} ...")
        cur.execute(stmt)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Internalize external tables into Redshift."
    )
    parser.add_argument(
        "--cluster_name",
        type=str,
        help="The name of the cluster to connect to.",
        required=True,
    )
    args = parser.parse_args()

    cluster = Cluster.from_config(args.cluster_name)
    conn = cluster.conn_pool(
        minconn=1, maxconn=1, search_path=EXTERNAL_SCHEMA
    ).getconn()

    ensure_schema(conn)
    tables = get_external_tables(conn)

    print(f"Found {len(tables)} external tables")

    for table in tables:
        print(f"{datetime.now()} Processing table {table}...")
        if table_exists(conn, table):
            print(f"Skipping {table} (already exists)")
            continue

        materialize_table(conn, table)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
