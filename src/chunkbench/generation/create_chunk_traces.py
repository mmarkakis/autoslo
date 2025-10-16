import argparse
import os
import numpy as np
import yaml

import pandas as pd
from tqdm.auto import tqdm

import chunkbench.path_utils as pu

from typing import Any


def create_one_chunk_trace(chunk_generation_spec_path: str, out_path: str) -> dict[str, Any]:
    """
    Create a chunk trace from a single generation spec.

    Parameters:
        chunk_generation_spec_path: Path to the chunk generation spec file.
        out_path: Path to save the generated chunk trace.

    Returns:
        A dictionary of summary statistics about the generated chunk trace.
    """
    # Read in the generation spec and validate.
    with open(chunk_generation_spec_path, "r") as f:
        spec = yaml.safe_load(f)

    assert spec["schema"] == "tpcds", "Only TPC-DS schema is supported for now."
    assert spec["schema"] in pu.HEAVY_TEMPLATES_FILES, f"No heavy templates file for schema {spec['schema']}."
    assert spec["num_templates"] <= 99, "TPC-DS has only 99 templates."


    # Retrieve the query texts for the specified number of templates.
    query_texts = {}
    for template_id in range(1, spec["num_templates"] + 1):
        template_str = f"query{template_id:03d}"
        template_dir = os.path.join(pu.QUERIES_PATH, template_str)
        if not os.path.exists(template_dir):
            print(f"Template directory {template_dir} does not exist. Skipping.")
            continue

        query_texts[template_id] = []

        for query_num in range(1, spec["num_queries_per_template"] + 1):
            with open(
                os.path.join(template_dir, f"{template_str}_{query_num:03d}.sql"), "r"
            ) as f:
                query_text = f.read()
            query_texts[template_id].append(query_text)

    # Determine which templates are heavy and which are light.
    heavy_templates = set()
    with open(pu.HEAVY_TEMPLATES_FILES[spec["schema"]], "r") as f:
        for line in f:
            template_id = int(line.strip())
            if template_id <= spec["num_templates"]:
                heavy_templates.add(template_id)
    light_templates = set(range(1, spec["num_templates"] + 1)) - heavy_templates

    # Generate the chunk trace.
    chunk_trace = []
    current_time_s = 0.0
    np.random.seed(spec["random_seed"])
    query_id = 0
    while current_time_s < spec["chunk_duration_s"]:
        # Create record for current query.
        pick_heavy = ((np.random.rand() * 100) < spec["pct_heavy"])
        if pick_heavy:
            template_id = np.random.choice(list(heavy_templates))
        else:
            template_id = np.random.choice(list(light_templates))
        query_num_within_template = np.random.randint(0, spec["num_queries_per_template"])
        query_text = query_texts[template_id][query_num_within_template]

        chunk_trace.append({
            "chunk_id": spec["chunk_id"],
            "query_id": query_id,
            "rel_start_time_s": current_time_s,
            "query_template": template_id,
            "query_num_within_template": query_num_within_template,
            "query_text": query_text,
        })
        query_id += 1

        # Update current time.
        interarrival_time_s = max(
            0.0,
            np.random.normal(
                loc=spec["mean_interarrival_time_s"],
                scale=spec["stddev_interarrival_time_s"],
            ),
        )
        current_time_s += interarrival_time_s

    # Write out the chunk trace as a Parquet file.
    df = pd.DataFrame(chunk_trace)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Sanity check statistics from the file we just wrote out and return.
    return {
        "chunk_id": spec["chunk_id"],
        "chunk_duration_s": spec["chunk_duration_s"],
        "expected__num_queries": int(spec["chunk_duration_s"] / spec["mean_interarrival_time_s"]),
        "actual__num_queries": df.shape[0],
        "expected__num_templates": spec["num_templates"],
        "actual__num_templates": df['query_template'].nunique(),
        "expected__num_queries_per_template": spec["num_queries_per_template"],
        "actual__num_queries_per_template": df.groupby('query_template')['query_num_within_template'].nunique().max(),
        "expected__pct_heavy": spec["pct_heavy"],
        "actual__pct_heavy": 100 * df[df['query_template'].isin(heavy_templates)].shape[0] / df.shape[0],
        "expected__mean_interarrival_time_s": spec["mean_interarrival_time_s"],
        "actual__mean_interarrival_time_s": df['rel_start_time_s'].diff().mean(),
        "expected__stddev_interarrival_time_s": spec["stddev_interarrival_time_s"],
        "actual__stddev_interarrival_time_s": df['rel_start_time_s'].diff().std(),
    }


def create_all_chunk_traces(specs_dir: str, out_dir: str) -> None:
    """
    Create chunk traces from all generation specs in the specified directory.

    Parameters:
        specs_dir: Directory containing chunk generation specs.
        out_dir: Directory to save generated chunk traces.
    """
    # Generate all chunk traces.
    os.makedirs(out_dir, exist_ok=True)
    l = []
    for spec_file in tqdm(os.listdir(specs_dir)):
        if not spec_file.endswith(".yml"):
            continue
        spec_path = os.path.join(specs_dir, spec_file)
        trace_out_path = os.path.join(out_dir, spec_file.replace("yml", "parquet"))
        stats = create_one_chunk_trace(spec_path, trace_out_path)
        l.append(stats)

    # Postprocess and save summary statistics.
    df = pd.DataFrame(l)
    df = df.sort_values(by=["chunk_id"]).reset_index(drop=True)
    df.columns = pd.MultiIndex.from_tuples(
        [("_".join(col.split("__")[1:]), col.split("__")[0]) for col in df.columns],
        names=["var", "type"]
    )
    summary_out_path = os.path.join(out_dir, "summary_stats.parquet")
    df.to_parquet(summary_out_path, index=False)
    print(f"Wrote out summary statistics to {summary_out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create chunk traces from generation specs."
    )
    parser.add_argument(
        "--specs-dir",
        type=str,
        default=os.path.join(pu.DATA_PATH, "chunk_generation_specs"),
        help="Directory containing chunk generation specs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(pu.DATA_PATH, "chunk_traces"),
        help="Directory to save generated chunk traces.",
    )
    args = parser.parse_args()

    create_all_chunk_traces(args.specs_dir, args.out_dir)
