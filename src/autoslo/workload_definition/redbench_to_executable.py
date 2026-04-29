import argparse
import json
import os

import pandas as pd
import rich

import autoslo.filesystem.path_utils as pu

from autoslo.workload_definition.query import QueryTextId


def main(
    workload_dir: str, new_q_per_template_min: int, new_q_per_template_max: int
) -> None:
    # Read in the workload queries from 'queries.json', either in the workload_dir
    # or in a subdir starting with `matching_` or `generation`.
    queries_json_path = os.path.join(workload_dir, "queries.json")
    if not os.path.exists(queries_json_path):
        subdirs = os.listdir(workload_dir)
        matching_subdirs = [
            d
            for d in subdirs
            if d.startswith("matching_") or d.startswith("generation")
        ]
        if len(matching_subdirs) == 0:
            raise FileNotFoundError(
                f"No 'queries.json' found in {workload_dir} or its subdirectories."
            )
        elif len(matching_subdirs) > 1:
            raise ValueError(
                f"Multiple subdirectories starting with 'matching_' or 'generation_' found in {workload_dir}."
            )
        queries_json_path = os.path.join(
            workload_dir, matching_subdirs[0], "queries.json"
        )
        if not os.path.exists(queries_json_path):
            raise FileNotFoundError(
                f"No 'queries.json' found in {workload_dir} or its subdirectories."
            )
    with open(queries_json_path, "r") as f:
        queries = json.load(f)

    # Read in records.
    q_per_template_width = new_q_per_template_max - new_q_per_template_min + 1
    records = []
    for query in queries:
        if query["query_type"] != "select":
            continue
        original_q_per_template = int(query["q_in_template"])
        reassigned_q_per_template = (
            original_q_per_template % q_per_template_width
        ) + new_q_per_template_min
        combined = "#".join(
            [
                "ext_tpcds1000",
                f"{int(query['template']):03d}",
                f"{reassigned_q_per_template:03d}",
            ]
        )
        record = {
            "query_id": query["redset_query"]["query_id"],
            "abs_start_time_str": query["arrival_timestamp"],
            "query_text_id": combined,
            "repetition_id": query["query_hash"],
        }

        records.append(record)
    df = pd.DataFrame.from_records(records)

    # Convert as needed and save to Parquet.
    df["abs_start_time"] = pd.to_datetime(df["abs_start_time_str"])
    df = df.drop(columns=["abs_start_time_str"])
    output_path = os.path.join(workload_dir, "workload.parquet")
    df.to_parquet(output_path, index=False)
    output_path_2 = os.path.join(
        pu.get_workloads_dir(),
        "ext_tpcds1000",
        f"{os.path.basename(workload_dir)}.parquet",
    )
    df.to_parquet(output_path_2, index=False)

    # Pretty print some stats about the workload in a table.
    rich.print(
        f"Workload '{os.path.basename(workload_dir)}' converted to executable format:"
    )
    stats_table = rich.table.Table(title="Workload Stats")
    stats_table.add_column("Stat", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="magenta")
    stats_table.add_row("Total Queries", str(len(df)))
    stats_table.add_row(
        "Unique Query Templates",
        str(
            df["query_text_id"]
            .apply(lambda x: QueryTextId(x).template_id)
            .nunique()
        ),
    )
    stats_table.add_row(
        "Unique Template+Query Index",
        str(df["query_text_id"].nunique()),
    )
    stats_table.add_row(
        "Time Range",
        f"{df['abs_start_time'].min()} to {df['abs_start_time'].max()}",
    )
    stats_table.add_row(
        "Mean Inter-Arrival Time (seconds)",
        str(df["abs_start_time"].diff().dt.total_seconds().mean()),
    )
    stats_table.add_row(
        "Mean Queries per Day",
        str(
            len(df)
            / (df["abs_start_time"].max() - df["abs_start_time"].min()).days
        ),
    )
    rich.print(stats_table)

    # Also save a "plan_collection" workload that contains each unique query once,
    # so that we can collect their
    unique_queries = df.drop_duplicates(subset=["query_text_id"])
    warmup_df = pd.concat([unique_queries] * 3, ignore_index=True)
    warmup_df = warmup_df.sample(
        frac=1, random_state=42, replace=False
    ).reset_index(drop=True)
    warmup_df["abs_start_time"] = pd.Timestamp("2024-01-01 00:00:00")
    warmup_output_path = os.path.join(workload_dir, "warmup_workload.parquet")
    warmup_df.to_parquet(warmup_output_path, index=False)
    warmup_output_path_2 = os.path.join(
        pu.get_workloads_dir(),
        "ext_tpcds1000",
        f"{os.path.basename(workload_dir)}_warmup.parquet",
    )
    warmup_df.to_parquet(warmup_output_path_2, index=False)

    # Print some stats about the warm-up workload as well.
    rich.print(
        f"Warm-up workload for '{os.path.basename(workload_dir)}' created with 3 copies of each unique query:"
    )
    warmup_stats_table = rich.table.Table(title="Warm-up Workload Stats")
    warmup_stats_table.add_column("Stat", style="cyan", no_wrap=True)
    warmup_stats_table.add_column("Value", style="magenta")
    warmup_stats_table.add_row("Total Queries", str(len(warmup_df)))
    warmup_stats_table.add_row(
        "Unique Query Templates",
        str(
            warmup_df["query_text_id"]
            .apply(lambda x: QueryTextId(x).template_id)
            .nunique()
        ),
    )
    warmup_stats_table.add_row(
        "Unique Template+Query Index",
        str(warmup_df["query_text_id"].nunique()),
    )
    rich.print(warmup_stats_table)

    # Now we also need to retrieve the corresponding query texts for each unique
    # query and save them in a separate file.
    query_text_records = []
    for _, row in unique_queries.iterrows():
        schema_name = QueryTextId(row["query_text_id"]).schema_name
        query_text_file = os.path.join(
            pu.QUERIES_PATH,
            schema_name,
            f"{row['query_text_id']}.sql",
        )
        with open(query_text_file, "r") as f:
            query_text = f.read()
        query_text_records.append(
            {
                "query_text_id": row["query_text_id"],
                "query_text": query_text,
            }
        )
    query_text_df = pd.DataFrame.from_records(query_text_records)
    query_text_df = query_text_df.drop_duplicates(subset=["query_text_id"])
    query_text_df = query_text_df.sort_values(by=["query_text_id"]).reset_index(
        drop=True
    )
    query_text_output_path = os.path.join(workload_dir, "query_texts.parquet")
    query_text_df.to_parquet(query_text_output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RedBench workload definition to executable format"
    )
    parser.add_argument(
        "--workload_name",
        type=str,
        required=True,
        help="Name of the Redbench workload.",
    )
    parser.add_argument(
        "--new_q_per_template_min",
        type=int,
        default=1,
        help="Minimum query per template in the new workload.",
    )
    parser.add_argument(
        "--new_q_per_template_max",
        type=int,
        default=3,
        help="Maximum query per template in the new workload.",
    )
    args = parser.parse_args()

    workload_name = args.workload_name
    workload_dir = os.path.join(
        pu.get_data_path(), "redset_executable_workloads", workload_name
    )
    main(workload_dir, args.new_q_per_template_min, args.new_q_per_template_max)
