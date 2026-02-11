import os

import argparse
import json
import yaml

import pandas as pd

import autoslo.utils.paths as pu

tpch_table_names = [
    "region",
    "nation",
    "customer",
    "lineitem",
    "orders",
    "partsupp",
    "part",
    "supplier",
]


def read_in_query_templates(template_dir: str) -> dict[str, str]:
    """
    Read in the CAB query templates from the specified directory.

    Parameters:
        template_dir: Path to the directory containing the query templates.

    Returns:
        A dictionary mapping template IDs to their SQL strings.
    """

    templates: dict[int, str] = {}
    for filename in os.listdir(template_dir):
        # Check that the format is <number>.sql
        if not filename.endswith(".sql") or not filename[:-4].isdigit():
            continue

        template_id = int(filename.split(".")[0])
        with open(os.path.join(template_dir, filename), "r") as f:
            template_sql = f.read()

        templates[template_id] = template_sql

    return templates


def generate_query_rows(stream_id, queries, templates, database_id):
    query_rows = []

    for i, query in enumerate(queries):
        template_id = query["query_id"]
        query_num_within_template = 0  # Actually create this
        rel_start_ms = query["start"]
        arguments = query.get("arguments", [])

        sql = templates[template_id]

        # Replace table placeholders.
        for table_name in tpch_table_names:
            placeholder = f":{table_name}"
            actual_table_name = f"{table_name}_{database_id}"
            sql = sql.replace(placeholder, actual_table_name)

        # Replace arguments.
        for i in range(len(arguments)):
            placeholder = f"${i+1}"
            if type(arguments[i]) == str:
                actual_value = f"'{arguments[i]}'"
            else:
                actual_value = str(arguments[i])
            sql = sql.replace(placeholder, actual_value)
            sql = sql.replace(":split:", "")

        # Create the query row.
        sql = (
            f"-- Filename: query{template_id:03d}_{query_num_within_template:03d}.sql\n"
            + sql
        )
        query_row = {
            "query_id": f"{stream_id}_{i}",
            "stream_id": stream_id,
            "rel_start_time_s": rel_start_ms / 1000.0,
            "query_template": template_id,
            "query_num_within_template": query_num_within_template,
            "query_text": sql,
        }
        query_rows.append(query_row)

    return query_rows


def main(cab_root_dir: str):

    # Define paths.
    cab_query_streams_path = os.path.join(
        cab_root_dir, "benchmark-gen", "query_streams"
    )
    template_dir = os.path.join(cab_root_dir, "benchmark-run", "sql_redshift")

    # Process the configuration parameters.
    generation_params_path = os.path.join(
        cab_query_streams_path, "generation_parameters.json"
    )
    with open(generation_params_path, "r") as f:
        generation_params = json.load(f)

    cab_factor = generation_params["cab_factor"]
    hours = generation_params["total_duration_in_hours"]
    seed_multiplier = generation_params["seed_multiplier"]
    num_dbs = generation_params["database_count"]
    num_streams = generation_params.get("num_streams", num_dbs)
    workload_name = f"cab_factor{cab_factor}_hours{hours}_seed{seed_multiplier}_dbs{num_dbs}_streams{num_streams}"

    # Write them out.
    output_dir = os.path.join(
        pu.get_data_path(), "cab_workloads", workload_name
    )
    out_params_path = os.path.join(output_dir, "generation_parameters.yml")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_params_path, "w") as f:
        yaml.dump(generation_params, f)

    # Process each of the CAB query stream files.
    all_query_rows = []
    templates = read_in_query_templates(template_dir)
    for filename in os.listdir(cab_query_streams_path):
        if not filename.startswith("query_stream_"):
            continue

        stream_id = int(filename.split("_")[-1].split(".")[0])

        with open(os.path.join(cab_query_streams_path, filename), "r") as f:
            stream_data = json.load(f)

        query_rows = generate_query_rows(
            stream_id,
            stream_data["queries"],
            templates,
            stream_data["database_id"],
        )
        all_query_rows.extend(query_rows)

    # Prepare and write out final dataframe.
    all_queries_df = pd.DataFrame(all_query_rows)
    all_queries_df = all_queries_df.sort_values(
        by="rel_start_time_s"
    ).reset_index(drop=True)
    output_path = os.path.join(output_dir, f"{workload_name}.parquet")
    all_queries_df.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--cab_root_dir",
        type=str,
        required=True,
        help="Path to the root of the CAB code.",
    )
    args = parser.parse_args()

    main(args.cab_root_dir)
