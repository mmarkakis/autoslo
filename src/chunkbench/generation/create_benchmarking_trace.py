import argparse
import os
import random

import pandas as pd
from tqdm.auto import tqdm

import chunkbench.path_utils as pu


def create_benchmarking_trace(
    queries_path: str,
    num_templates: int,
    queries_per_template: int,
    copies_per_query: int,
    shuffle: bool = False,
    seed: int = 42,
):
    """
    Create a benchmarking trace by selecting a specified number of queries per TPC-DS template.

    Parameters:
        queries_path: Path to the directory containing TPC-DS queries.
        num_templates: Number of templates to include in the trace.
        queries_per_template: Number of queries to select per template.
        copies_per_query: Number of copies of each query to include in the trace.
        shuffle: Whether to shuffle the trace.
        seed: Random seed for shuffling.
    """

    # Generate the entries of the benchmarking trace.
    l = []
    for template_id in tqdm(range(1, num_templates + 1), desc="Processing templates"):
        template_str = f"query{template_id:03d}"
        template_dir = os.path.join(queries_path, template_str)
        if not os.path.exists(template_dir):
            print(f"Template directory {template_dir} does not exist. Skipping.")
            continue

        for query_num in range(1, queries_per_template + 1):
            with open(
                os.path.join(template_dir, f"{template_str}_{query_num:03d}.sql"), "r"
            ) as f:
                query_text = f.read()
            for _ in range(copies_per_query):
                l.append(
                    {
                        "query_id": "",  # Placeholder, will be filled later.
                        "rel_start_time_s": 0,
                        "query_template": template_id,
                        "query_num_within_template": query_num,
                        "query_text": query_text,
                    }
                )

    # Shuffle if needed and add in query_ids.
    if shuffle:
        random.seed(seed)
        random.shuffle(l)
    df = pd.DataFrame(l)
    df["query_id"] = df.apply(
        lambda row: f"{row.name}/{row['query_template']}/{row['query_num_within_template']}",
        axis=1,
    )

    # Write out as a parquet file.
    out_dir = os.path.join(pu.DATA_PATH, "benchmarking_traces")
    os.makedirs(out_dir, exist_ok=True)
    shuffle_suffix = f"_shuffled_{seed}" if shuffle else ""
    out_path = os.path.join(
        out_dir,
        f"benchmarking_trace_{num_templates}_{queries_per_template}_{copies_per_query}{shuffle_suffix}.parquet",
    )
    df.to_parquet(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_templates",
        "-nt",
        type=int,
        default=99,
        help="Number of templates to include in the trace.",
    )
    parser.add_argument(
        "--queries_per_template",
        "-qpt",
        type=int,
        default=10,
        help="Number of queries per template to include in the trace.",
    )
    parser.add_argument(
        "--copies_per_query",
        "-cpq",
        type=int,
        default=5,
        help="Number of copies of each query to include in the trace.",
    )
    parser.add_argument(
        "--shuffle", "-s", action="store_true", help="Whether to shuffle the trace."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling."
    )
    args = parser.parse_args()

    create_benchmarking_trace(
        pu.QUERIES_PATH,
        args.num_templates,
        args.queries_per_template,
        args.copies_per_query,
        args.shuffle,
        args.seed,
    )
