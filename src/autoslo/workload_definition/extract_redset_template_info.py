import logging
import multiprocessing as mp
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import seaborn as sns
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.forecasting.arrival_classifier import ArrivalClassifier
from autoslo.workload_definition.query import Query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="extract_redset_template_info.log",
    filemode="a",
)

logger = logging.getLogger(__name__)

cluster_type = "provisioned"

columns = [
    "database_id",
    "query_id",
    "feature_fingerprint",
    "was_cached",
    "query_type",
    "arrival_timestamp",
    "execution_duration_ms",
    "queue_duration_ms",
]


def process_cluster(cluster_id: int) -> bool:

    start_time = pd.Timestamp.now()

    logger.info(f"Started processing cluster {cluster_id}.")

    path = pu.get_redset_raw_data(
        cluster_type=cluster_type, cluster_id=cluster_id
    )
    out_dir = os.path.join(
        pu.get_data_path(),
        "redset_byproducts",
        cluster_type,
        "per_cluster",
        str(cluster_id),
    )
    workload_df_path = os.path.join(out_dir, f"workload.parquet")

    try:
        workload_df = pd.read_parquet(workload_df_path)
    except Exception:
        df = pd.read_parquet(path, columns=columns)

        # Remove cached results, non-SELECT queries, and rows with null
        # feature_fingerprint.
        # mask = (
        #     (~df["was_cached"])
        #     & (df["query_type"].str.lower() == "select")
        #     & (df["feature_fingerprint"].notna())
        # )
        # df = df[mask]
        if len(df) == 0:
            logger.info(
                f"Skipping cluster {cluster_id} because it has no valid "
                f"queries after pre-filtering."
            )
            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            logger.info(
                f"Finished processing cluster {cluster_id} in {duration}."
            )
            return False

        # Only keep the cluster if no day has fewer than 100 queries, and no
        # day has more than 10k queries.
        df["date"] = df["arrival_timestamp"].dt.date
        date_counts = df["date"].value_counts()
        min_date = df["date"].min()
        max_date = df["date"].max()
        date_counts = date_counts.reindex(
            pd.date_range(min_date, max_date), fill_value=0
        )
        if (date_counts < 100).any():
            logger.info(
                f"Skipping cluster {cluster_id} because it does not have at "
                f"least 100 queries on each day."
            )
            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            logger.info(
                f"Finished processing cluster {cluster_id} in {duration}."
            )
            return False
        if (date_counts > 10000).any():
            logger.info(
                f"Skipping cluster {cluster_id} because it has more than "
                f"10k queries on at least one day."
            )
            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            logger.info(
                f"Finished processing cluster {cluster_id} in {duration}."
            )
            return False

        if len(date_counts) < 14:
            logger.info(
                f"Skipping cluster {cluster_id} because it has fewer than 14 "
                f"days of data."
            )
            end_time = pd.Timestamp.now()
            duration = end_time - start_time
            logger.info(
                f"Finished processing cluster {cluster_id} in {duration}."
            )
            return False

        df["query_template"] = df[["database_id", "feature_fingerprint"]].agg(
            lambda x: f"{x['database_id']}_{x['feature_fingerprint']}",
            axis=1,
        )
        df["latency_s"] = (
            df["execution_duration_ms"] + df["queue_duration_ms"]
        ) / 1000

        workload_df = df[
            [
                "arrival_timestamp",
                "query_id",
                "query_template",
                "latency_s",
                "query_type",
            ]
        ]
        os.makedirs(out_dir, exist_ok=True)
        workload_df.to_parquet(workload_df_path)

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
                    abs_start_time=row["arrival_timestamp"],
                )
            )
        os.makedirs(out_dir, exist_ok=True)
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

    # Plot arrival patterns.
    logger.info(f"Plotting arrival patterns for cluster {cluster_id}.")
    plot(workload_df, out_dir, cluster_id)

    end_time = pd.Timestamp.now()
    duration = end_time - start_time
    logger.info(f"Finished processing cluster {cluster_id} in {duration}.")
    return True


def plot(workload_df: pd.DataFrame, out_dir: str, cluster_id: int):

    df = workload_df.copy()
    df["arrival_hour"] = df["arrival_timestamp"].dt.floor("h")
    df["hour_of_day"] = df["arrival_timestamp"].dt.hour
    df["day"] = df["arrival_timestamp"].dt.date
    df["on_weekend"] = df["arrival_timestamp"].dt.dayofweek >= 5
    df["day_of_week"] = df["arrival_timestamp"].dt.dayofweek
    df["week"] = df["arrival_timestamp"].dt.isocalendar().week

    # Plot a 2*2 grid of plots.
    fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharey="row")

    # Top left: number of queries per iso week, as a box plot. The x-axis should
    # be the week number, and the y-axis should be the number of queries. Each
    # point in the box plot corresponds to a day, and the value of the point is
    # the number of queries on that day.
    daily_counts_s = df.groupby(["week", "day"]).size()
    daily_counts = daily_counts_s.reset_index(name="query_count")
    sns.boxplot(x="week", y="query_count", data=daily_counts, ax=axs[0, 0])
    axs[0, 0].set_xlabel("Week Number")
    axs[0, 0].set_ylabel("Number of Queries")
    axs[0, 0].set_title(f"Weekly Query Counts")

    # Top right: number of queries per day of week.
    daily_counts_s = df.groupby(["day_of_week", "day"]).size()
    daily_counts = daily_counts_s.reset_index(name="query_count")
    sns.boxplot(
        x="day_of_week", y="query_count", data=daily_counts, ax=axs[0, 1]
    )
    axs[0, 1].set_xlabel("Day of Week")
    axs[0, 1].set_ylabel("Number of Queries")
    axs[0, 1].set_title(f"Daily Query Counts")
    axs[0, 1].set_xticks(range(0, 7))
    axs[0, 1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    # Middle left: number of select queries per hour of day.
    hourly_counts_s = (
        df[df["query_type"] == "select"].groupby(["hour_of_day", "day"]).size()
    )
    hourly_counts = hourly_counts_s.reset_index(name="query_count")
    sns.boxplot(
        x="hour_of_day", y="query_count", data=hourly_counts, ax=axs[1, 0]
    )
    axs[1, 0].set_xlabel("Hour of Day")
    axs[1, 0].set_ylabel("Number of Queries")
    axs[1, 0].set_title(f"Hourly SELECT Query Counts")
    axs[1, 0].set_xticks(range(0, 24))
    axs[1, 0].set_yscale("log")

    # Middle right: number of non-select queries per hour of day.
    hourly_counts_s = (
        df[df["query_type"] != "select"].groupby(["hour_of_day", "day"]).size()
    )
    hourly_counts = hourly_counts_s.reset_index(name="query_count")
    sns.boxplot(
        x="hour_of_day",
        y="query_count",
        data=hourly_counts,
        ax=axs[1, 1],
        color="orange",
    )
    axs[1, 1].set_xlabel("Hour of Day")
    axs[1, 1].set_ylabel("Number of Queries")
    axs[1, 1].set_title(f"Hourly Non-SELECT Query Counts")
    axs[1, 1].set_xticks(range(0, 24))
    axs[1, 1].set_yscale("log")

    # Bottom left: latency distribution of select queries per hour of day.
    sns.boxplot(
        x="hour_of_day",
        y="latency_s",
        data=df[df["query_type"] == "select"],
        ax=axs[2, 0],
    )
    axs[2, 0].set_xlabel("Hour of Day")
    axs[2, 0].set_ylabel("Latency (s)")
    axs[2, 0].set_title(f"Hourly SELECT Query Latencies")
    axs[2, 0].set_yscale("log")

    # Bottom right: latency distribution of non-select queries per hour of day.
    sns.boxplot(
        x="hour_of_day",
        y="latency_s",
        data=df[df["query_type"] != "select"],
        ax=axs[2, 1],
        color="orange",
    )
    axs[2, 1].set_xlabel("Hour of Day")
    axs[2, 1].set_ylabel("Latency (s)")
    axs[2, 1].set_title(f"Hourly Non-SELECT Query Latencies")
    axs[2, 1].set_yscale("log")

    plt.suptitle(
        f"Query Arrival Patterns for Cluster {cluster_id}", fontsize=16
    )
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"arrival_patterns.png")
    plt.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    cluster_ids = list(range(200))
    threads_per_arrow = 4
    num_jobs = mp.cpu_count() // threads_per_arrow

    pa.set_cpu_count(threads_per_arrow)
    pa.set_io_thread_count(threads_per_arrow)

    with mp.Pool(processes=num_jobs) as pool:
        x = list(
            tqdm(
                pool.imap_unordered(process_cluster, cluster_ids),
                total=len(cluster_ids),
            )
        )

    logger.info(
        f"Successfully processed {sum(x)} out of {len(cluster_ids)} clusters."
    )
