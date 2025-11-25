import multiprocessing as mp
import os
from functools import partial

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

import autoslo.utils.paralellism as plu
import autoslo.utils.paths as pu

MS_IN_S = 1_000


def group_one(df: pd.DataFrame, group_column: str, out_path: str):
    """
    Groups the input DataFrame by the specified column and computes various
    aggregate statistics for each group. The resulting aggregated DataFrame is
    then saved to the specified output path in Parquet format.

    Parameters:
        df: Input DataFrame containing query execution data.
        group_column: The column name to group by (e.g., 'hour_bin', 'day_bin').
        out_path: The path to save the aggregated DataFrame in Parquet format.
    """
    agg_df = (
        df.groupby(group_column)
        .agg(
            instance_id=("instance_id", "first"),
            num_queries=("arrival_timestamp", "size"),
            unique_cluster_sizes=(
                "cluster_size",
                lambda x: list(x[x.notna()].unique()),
            ),
            unique_cluster_size_count=("cluster_size", "nunique"),
            nan_cluster_size_num_queries=(
                "cluster_size",
                lambda x: x.isna().sum(),
            ),
            duration_s_p95=(
                "total_duration_ms",
                lambda x: x.quantile(0.95) / MS_IN_S,
            ),
            duration_s_p99=(
                "total_duration_ms",
                lambda x: x.quantile(0.99) / MS_IN_S,
            ),
            duration_s_sum=("total_duration_ms", lambda x: x.sum() / MS_IN_S),
            was_aborted_mean=("was_aborted", "mean"),
            was_cached_mean=("was_cached", "mean"),
            num_permanent_tables_accessed_mean=(
                "num_permanent_tables_accessed",
                "mean",
            ),
            num_external_tables_accessed_mean=(
                "num_external_tables_accessed",
                "mean",
            ),
            num_system_tables_accessed_mean=(
                "num_system_tables_accessed",
                "mean",
            ),
            mbytes_scanned_mean=("mbytes_scanned", "mean"),
            mbytes_scanned_p95=("mbytes_scanned", lambda x: x.quantile(0.95)),
            mbytes_scanned_p99=("mbytes_scanned", lambda x: x.quantile(0.99)),
            mbytes_spilled_mean=("mbytes_spilled", "mean"),
            mbytes_spilled_p95=("mbytes_spilled", lambda x: x.quantile(0.95)),
            mbytes_spilled_p99=("mbytes_spilled", lambda x: x.quantile(0.99)),
            num_joins_mean=("num_joins", "mean"),
            num_joins_p95=("num_joins", lambda x: x.quantile(0.95)),
            num_joins_p99=("num_joins", lambda x: x.quantile(0.99)),
            num_scans_mean=("num_scans", "mean"),
            num_scans_p95=("num_scans", lambda x: x.quantile(0.95)),
            num_scans_p99=("num_scans", lambda x: x.quantile(0.99)),
            num_aggregations_mean=("num_aggregations", "mean"),
            num_aggregations_p95=(
                "num_aggregations",
                lambda x: x.quantile(0.95),
            ),
            num_aggregations_p99=(
                "num_aggregations",
                lambda x: x.quantile(0.99),
            ),
        )
        .reset_index()
    )
    agg_df["nan_cluster_size_duration_s_sum"] = (
        df[df["cluster_size"].isna()]
        .groupby(group_column)["total_duration_ms"]
        .sum()
        .reset_index(drop=True)
        / MS_IN_S
    )

    # Also for each separate value of "query_type" in df, create a separate
    # column with the fraction of queries of that type in the bin.
    query_type_dummies = pd.get_dummies(df["query_type"], prefix="query_type")
    query_type_dummies[group_column] = df[group_column]
    query_type_fraction = (
        query_type_dummies.groupby(group_column).mean().reset_index()
    )
    agg_df = agg_df.merge(query_type_fraction, on=group_column, how="left")

    agg_df.to_parquet(out_path, index=False)


def process_one(file_name: str, in_dir: str, out_dir: str):
    """
    Process a single Parquet file by reading it, performing aggregations,
    and saving the results to the output directory.

    Parameters:
        file_name: The name of the Parquet file to process.
        in_dir: The input directory containing the Parquet files.
        out_dir: The output directory to save the processed files.
    """
    in_path = os.path.join(in_dir, file_name)
    stem, _ = os.path.splitext(file_name)

    pa.set_cpu_count(plu.inner_level_num_cpus())

    df = pd.read_parquet(
        in_path,
        columns=[
            "arrival_timestamp",
            "queue_duration_ms",
            "execution_duration_ms",
            "cluster_size",
            "was_aborted",
            "was_cached",
            "num_permanent_tables_accessed",
            "num_external_tables_accessed",
            "num_system_tables_accessed",
            "mbytes_scanned",
            "mbytes_spilled",
            "num_joins",
            "num_scans",
            "num_aggregations",
            "query_type",
        ],
        engine="pyarrow",
    )
    df["hour_bin"] = df["arrival_timestamp"].dt.floor("h")
    df["day_bin"] = df["arrival_timestamp"].dt.floor("D")
    df["total_duration_ms"] = (
        df["queue_duration_ms"] + df["execution_duration_ms"]
    )

    df["num_permanent_tables_accessed"] = df[
        "num_permanent_tables_accessed"
    ].fillna(0)
    df["num_external_tables_accessed"] = df[
        "num_external_tables_accessed"
    ].fillna(0)
    df["num_system_tables_accessed"] = df["num_system_tables_accessed"].fillna(
        0
    )
    df["mbytes_scanned"] = df["mbytes_scanned"].fillna(0)
    df["mbytes_spilled"] = df["mbytes_spilled"].fillna(0)
    df["num_joins"] = df["num_joins"].fillna(0)
    df["num_scans"] = df["num_scans"].fillna(0)
    df["num_aggregations"] = df["num_aggregations"].fillna(0)

    df["instance_id"] = int(stem)

    group_one(
        df, "hour_bin", os.path.join(out_dir, "hourly_agg", f"{stem}.parquet")
    )
    group_one(
        df, "day_bin", os.path.join(out_dir, "daily_agg", f"{stem}.parquet")
    )


def compose(out_dir: str, granularity: str):
    """
    Compose multiple Parquet files into a single DataFrame and save the result.

    Parameters:
        out_dir: The directory containing the Parquet files to compose.
        granularity: The granularity of the Parquet files to compose (e.g., 'hourly').
    """
    column_sets = {}
    for i in range(200):
        path = os.path.join(out_dir, f"{granularity}_agg", f"{i}.parquet")
        column_sets[i] = pq.read_schema(path).names

    superset = set().union(*column_sets.values())

    dfs = []
    pa.set_cpu_count(plu.inner_level_num_cpus())
    for i in tqdm(range(200), desc=f"Composing {granularity} data"):
        path = os.path.join(out_dir, f"{granularity}_agg", f"{i}.parquet")
        df = pd.read_parquet(path, engine="pyarrow")
        for col in superset:
            if col not in df.columns:
                assert col.startswith("query_type")
                df[col] = 0.0
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_parquet(
        os.path.join(out_dir, f"all_{granularity}.parquet"), index=False
    )


def process_all(in_dir: str, out_dir: str):
    """
    Process all Parquet files in the input directory by reading, aggregating,
    and saving the results to the output directory.

    Parameters:
        in_dir: The input directory containing the Parquet files.
        out_dir: The output directory to save the processed files.
    """
    file_names = []  # List all file names in the in_dir directory
    for file_name in os.listdir(in_dir):
        if file_name.endswith(".parquet"):
            file_names.append(file_name)

    process_one_partial = partial(process_one, in_dir=in_dir, out_dir=out_dir)

    with mp.Pool(plu.deg_of_paralellism()) as pool:
        try:
            for _ in tqdm(
                pool.imap_unordered(process_one_partial, file_names),
                total=len(file_names),
                desc="Processing Redset summary files",
            ):
                pass
        except KeyboardInterrupt:
            # Allow Ctrl-C to terminate worker pool promptly.
            pool.terminate()
            raise


if __name__ == "__main__":

    in_dir = os.path.join(pu.get_redset_raw_path(), "provisioned", "parts")
    out_dir = os.path.join(
        pu.get_data_path(), "redset_byproducts", "provisioned"
    )
    os.makedirs(os.path.join(out_dir, "hourly_agg"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "daily_agg"), exist_ok=True)

    process_all(in_dir, out_dir)
    compose(out_dir, "hourly")
    compose(out_dir, "daily")
