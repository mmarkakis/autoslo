import os
import pickle

import pandas as pd
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.forecasting.arrival_classifier import ArrivalClassifier
from autoslo.workload_definition.query import Query

import pyarrow as pa
import multiprocessing as mp

cluster_type = "provisioned"

columns = [
    "database_id",
    "query_id",
    "feature_fingerprint",
    "query_type",
    "num_permanent_tables_accessed",
    "num_external_tables_accessed",
    "num_system_tables_accessed",
    "read_table_ids",
    "write_table_ids",
    "arrival_timestamp",
]


def process_cluster(cluster_id: int):
    path = pu.get_redset_raw_data(
        cluster_type=cluster_type, cluster_id=cluster_id
    )
    out_dir = os.path.join(
        pu.get_data_path(),
        "redset_byproducts",
        cluster_type,
        str(cluster_id),
    )
    os.makedirs(out_dir, exist_ok=True)
    workload_df_path = os.path.join(out_dir, f"workload.parquet")

    try:
        workload_df = pd.read_parquet(workload_df_path)
    except Exception:
        df = pd.read_parquet(path, columns=columns)

        df["template_str"] = df.apply(
            lambda row: "_".join(
                [
                    str(row["database_id"]),
                    str(row["feature_fingerprint"]),
                    str(row["query_type"]),
                    str(row["num_permanent_tables_accessed"]),
                    str(row["num_external_tables_accessed"]),
                    str(row["num_system_tables_accessed"]),
                    str(row["read_table_ids"]),
                    str(row["write_table_ids"]),
                ]
            ),
            axis=1,
        )

        mapping = {}
        for i, template_str in enumerate(sorted(df["template_str"].unique())):
            mapping[template_str] = i
        df["query_template"] = df["template_str"].map(mapping)

        mapping_path = os.path.join(out_dir, f"template_mapping.pkl")
        with open(mapping_path, "wb") as f:
            pickle.dump(mapping, f)

        df["workload_id"] = f"redset_{cluster_type}_{cluster_id}"
        df["rel_start_time_s"] = df["arrival_timestamp"].apply(
            lambda x: x.timestamp()
        )
        df["query_num_within_template"] = 0
        df["query_text"] = ""

        workload_df = df[
            [
                "workload_id",
                "query_id",
                "rel_start_time_s",
                "query_template",
                "query_num_within_template",
                "query_text",
            ]
        ]
        workload_df_path = os.path.join(out_dir, f"workload.parquet")
        workload_df.to_parquet(workload_df_path, index=False)

    queries_path = os.path.join(out_dir, f"queries.pkl")
    try:
        with open(queries_path, "rb") as f:
            queries = pickle.load(f)
    except Exception:
        queries = []
        for i, row in workload_df.iterrows():
            queries.append(
                Query(
                    query_id=row["query_id"],
                    tpcds_temp_and_q_idx=f"{row['query_template']}",
                    start_time_s=row["rel_start_time_s"],
                )
            )
        with open(queries_path, "wb") as f:
            pickle.dump(queries, f)

    classification_path = os.path.join(out_dir, f"template_classification.pkl")
    if not os.path.exists(classification_path):

        classifier = ArrivalClassifier(queries=queries, verbose=False)
        classifier.classify_arrivals()

        with open(classification_path, "wb") as f:
            pickle.dump(
                {
                    "classification": classifier._template_classification,
                    "details": classifier._template_details,
                },
                f,
            )

if __name__ == "__main__":
    cluster_ids = list(range(200))
    threads_per_arrow = 4
    num_jobs = mp.cpu_count() // threads_per_arrow

    pa.set_cpu_count(threads_per_arrow)
    pa.set_io_thread_count(threads_per_arrow)

    with mp.Pool(processes=num_jobs) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_cluster, cluster_ids),
                total=len(cluster_ids),
            )
        )
