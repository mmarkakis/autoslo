import heapq
import os
import pickle
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, TypeAlias, cast

import networkx as nx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import autoslo.utils.paralellism as plu
import autoslo.utils.paths as pu
from autoslo.blueprints.cluster import Cluster
from autoslo.query_plans.parse_plan import parse_one_plan, plan_summary


class Trace:
    """
    A query execution trace. This class is used to abstract away the details of
    the trace data as collected from the database engine.
    """

    REQUIRED_COLUMNS = {
        "sys_query_history": [
            "query_id",
            "start_time",
            "end_time",
            "elapsed_time",
            "status",
            "result_cache_hit",
            "query_type",
        ],
        "sys_query_detail": [
            "query_id",
            "step_name",
            "output_bytes",
            "table_name",
        ],
        "sys_query_explain": [
            "query_id",
            "plan_node_id",
            "plan_node",
        ],
    }

    REDSHIFT_BILLING_THRESHOLD_S = 60
    REDSHIFT_BILLING_GRANULARITY_S = 1
    REDSHIFT_ELAPSED_TIME_UNIT = "us"  # microseconds
    BYTES_IN_MEGABYTE = 1_000_000

    REDSHIFT_SYSTEM_TABLE_SUBSTRINGS = [
        "sys_",
        "svv_",
        "stl_",
        "stv_",
        "svcs_",
        "svl_",
    ]
    REDSHIFT_PERMANENT_TABLE_SUBSTRINGS = ["tpcds1000"]

    TPCDSTempAndQIdx: TypeAlias = str
    """Represents the template number and the number of the query within it."""

    def __init__(self, run_id: str):
        """
        Create a Trace instance from a run directory, reading in only the
        required columns from each parquet file. If a cached version of the
        trace exists, it will be used instead.

        Parameters:
            run_id: The ID of the run directory containing the Parquet files.

        Returns:
            A Trace instance.

        Raises:
            ValueError: If no sys_query_history Parquet file is found in the
                run directory.
        """
        self._run_id = run_id
        self._uuid = uuid.uuid4()

        # A map from table_name to [a map of cluster_name to DataFrame].
        self._dfs: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        self._original_start = datetime.now()

        run_dir = os.path.join(pu.get_runs_path(), run_id)
        pq_filenames = [
            f for f in os.listdir(run_dir) if f.endswith(".parquet")
        ]
        for filename in pq_filenames:
            parts = filename.split(".")[0].split("+")
            if parts[0] in Trace.REQUIRED_COLUMNS.keys():
                table_name, cluster_name = parts[0], parts[1]
                df = Trace._read_with_colcheck(
                    os.path.join(run_dir, filename),
                    Trace.REQUIRED_COLUMNS[table_name],
                )
                if len(df) == 0:
                    continue

                # N.B.: We may create longer traces by appending multiple copies
                # of the same run of the same chunk to each other. To maintain
                # proper joins across the sys tables of each copy, we append
                # a UUID to each query_id.
                df["query_id"] = df.apply(
                    lambda r: f"{cluster_name}_{r['query_id']}#{self._uuid}",
                    axis=1,
                )

                if table_name == "sys_query_explain":
                    df = (
                        df[df["plan_node"].str.contains("XN")]
                        .sort_values(["query_id", "plan_node_id"])
                        .reset_index(drop=True)
                    )

                self._dfs[table_name][cluster_name] = df

                if table_name == "sys_query_history":
                    min_start_time = df["start_time"].min()
                    if min_start_time < self._original_start:
                        self._original_start = min_start_time

    @property
    def run_id(self) -> str:
        """
        Get the run ID of the trace.
        """
        return self._run_id

    @staticmethod
    def _read_with_colcheck(path: str, column_list: list[str]) -> pd.DataFrame:
        """
        Check if the Parquet file at the specified path contains the required
        columns, and read it into a DataFrame if so.

        Parameters:
            path: The path to the Parquet file.
            column_list: The list of columns to check for.

        Returns:
            A pandas DataFrame containing the data from the Parquet file.

        Raises:
            ValueError: If any of the specified columns are missing.
        """
        missing_columns = [
            col for col in column_list if col not in pq.read_schema(path).names
        ]
        if len(missing_columns) > 0:
            raise ValueError(
                f"DataFrame at {path} is missing required columns: "
                f"{', '.join(missing_columns)}"
            )
        pa.set_cpu_count(plu.inner_level_num_cpus())
        return pd.read_parquet(path, columns=column_list, engine="pyarrow")

    def normalize_start_to(self, new_start: datetime) -> "Trace":
        """
        Normalize start and end times of the trace so that the earliest start
        time is equal to `new_start`.

        Parameters:
            new_start: The new start time for the earliest query in the trace.

        Returns:
            A new Trace instance with normalized start times.
        """

        # We use the start and end times from sys_query_history as the source of
        # truth for the trace.
        for cluster_name, df in self._dfs["sys_query_history"].items():
            earliest_start = df["start_time"].min()
            shift = pd.Timestamp(new_start) - earliest_start
            normalized_df = df.copy()
            normalized_df["start_time"] = normalized_df["start_time"] + shift
            normalized_df["end_time"] = normalized_df["end_time"] + shift
            self._dfs["sys_query_history"][cluster_name] = normalized_df

        return self

    def reset_start(self) -> "Trace":
        """
        Reset start and end times of the trace to their original values.

        Returns:
            A new Trace instance with original start times.
        """
        return self.normalize_start_to(self._original_start)

    def append(self, other: "Trace") -> "Trace":
        """
        Append another Trace instance to this one. That is, treat the queries
        in the `other` trace as occurring after the queries in this trace.
        Insert the specified time gap between the two traces.

        Parameters:
            other: The other Trace instance to append.

        Returns:
            A new Trace instance containing the combined trace data.

        Raises:
            ValueError: If the earliest start time of the other trace is before
                the latest end time of this trace.
        """

        latest_end_time = self.latest_query_end_time
        other_earliest_start = other.earliest_query_start_time
        if other_earliest_start < latest_end_time:
            raise ValueError(
                "Cannot append trace: the earliest start time of the other "
                "trace is before the latest end time of this trace."
            )

        # Append the dataframes per cluster and table.
        for table_name, clusters in self._dfs.items():
            if table_name not in other._dfs:
                continue
            other_clusters = other._dfs[table_name]
            for cluster_name, df in clusters.items():
                if cluster_name not in other_clusters:
                    continue
                combined_df = pd.concat(
                    [
                        df,
                        other_clusters[cluster_name],
                    ]
                ).reset_index(drop=True)
                self._dfs[table_name][cluster_name] = combined_df

        return self

    @property
    def earliest_query_start_time(self) -> datetime:
        """
        Get the earliest start time across all clusters in the trace.

        Returns:
            The earliest start time as a datetime object.
        """
        earliest_start = None
        for df in self._dfs["sys_query_history"].values():
            cluster_earliest = df["start_time"].min()
            if earliest_start is None or cluster_earliest < earliest_start:
                earliest_start = cluster_earliest
        if earliest_start is None:
            raise ValueError("No sys_query_history data found in the trace.")
        return earliest_start

    @property
    def latest_query_end_time(self) -> datetime:
        """
        Get the latest end time across all clusters in the trace.

        Returns:
            The latest end time as a datetime object.
        """
        latest_end = None
        for df in self._dfs["sys_query_history"].values():
            cluster_latest = df["end_time"].max()
            if latest_end is None or cluster_latest > latest_end:
                latest_end = cluster_latest
        if latest_end is None:
            raise ValueError("No sys_query_history data found in the trace.")
        return latest_end

    @property
    def num_queries(self) -> int:
        """
        Get the total number of queries in the trace.

        Returns:
            The total number of queries.
        """
        total_queries = 0
        for df in self._dfs["sys_query_history"].values():
            total_queries += len(df)
        return total_queries

    @property
    def latencies_s(self) -> pd.Series:
        """
        Get the latencies of the queries in the trace, in seconds.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the `query_ids` property.

        Returns:
            A pandas Series where the index is the query IDs and the values are
                the latencies in seconds.
        """
        conversion_factor = pd.Timedelta(
            1, Trace.REDSHIFT_ELAPSED_TIME_UNIT  # type: ignore
        ).total_seconds()

        series = []
        for df in self._dfs["sys_query_history"].values():
            s = (
                df.set_index("query_id")["elapsed_time"].astype("float")
                * conversion_factor
            )
            series.append(s)

        return pd.concat(series).reindex(self.query_ids)

    @property
    def costs(self) -> list[float]:
        """
        Placeholder method to return costs associated with each cluster used
        in the trace.

        Returns:
            A list of costs associated with each cluster used in the trace.

        """
        return [
            self.cost_of_cluster(cluster_name)
            for cluster_name in self.seen_clusters
        ]

    @property
    def seen_clusters(self) -> list[str]:
        """
        Get the list of unique cluster names seen in the trace.

        Returns:
            A list of unique cluster names.
        """
        return list(self._dfs["sys_query_history"].keys())

    @property
    def routing_times_s(self) -> list[float]:
        """
        Placeholder method to return the time taken to route each query.

        Returns:
            A list of routing times in seconds.
        """
        # FIXME: Placeholder implementation - routing strategy should record
        # this and write it out.
        return [0.0 for _ in range(self.num_queries)]

    def cost_of_cluster(self, cluster_name: str) -> float:
        """
        Calculate the cost incurred by a specific cluster in the trace.

        Parameters:
            cluster_name: The name of the cluster for which to calculate the
                cost.

        Returns:
            The total cost incurred by the specified cluster.
        """
        if cluster_name not in self.seen_clusters:
            return 0.0

        df = self._dfs["sys_query_history"][cluster_name]
        billed_s = Trace._billed_s(df["start_time"], df["end_time"])
        cluster = Cluster.from_config(cluster_name)
        return billed_s * cluster.cost_per_second

    @staticmethod
    def _round_up(value: float, granularity: float) -> float:
        """
        Round up a value to the nearest multiple of granularity.

        Parameters:
            value: The value to round up.
            granularity: The granularity to round up to.
        """
        return granularity * ((value + granularity - 1) // granularity)

    @staticmethod
    def _billed_s(
        start_times: pd.Series,
        end_times: pd.Series,
        threshold_s: float = REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = REDSHIFT_BILLING_GRANULARITY_S,
    ) -> float:
        """
        Get the total billed time implied by the given start and end times,
        according to the specified billing threshold and granularity.

        Parameters:
            start_times: A pandas Series of start times.
            end_times: A pandas Series of end times.
            threshold_s: The billing threshold in seconds. This is the minimum
                time that will be billed - i.e. all smaller intervals are
                rounded up to this threshold.
            granularity_s: The billing granularity in seconds. This is the time
                interval to which the billed time is rounded up.

        Returns:
            The total billed time for the trace, in seconds.
        """

        if len(start_times) != len(end_times):
            raise ValueError(
                "start_times and end_times must have the same length."
            )
        if any(start_times > end_times):
            raise ValueError(
                "All start_times must be less than or equal to "
                "their corresponding end_times."
            )

        total_billed_s = 0.0
        if start_times.empty:
            return total_billed_s

        current_interval_start = start_times.iloc[0]
        current_interval_end = max(
            end_times.iloc[0],
            current_interval_start + pd.Timedelta(seconds=threshold_s),
        )

        for start_time, end_time in zip(start_times, end_times):
            if start_time <= current_interval_end:
                current_interval_end = max(current_interval_end, end_time)
            else:
                interval_duration_s = (
                    current_interval_end - current_interval_start
                ).total_seconds()
                interval_billed_s = max(
                    threshold_s,
                    Trace._round_up(interval_duration_s, granularity_s),
                )
                total_billed_s += interval_billed_s
                current_interval_start = start_time
                current_interval_end = max(
                    end_time,
                    current_interval_start + pd.Timedelta(seconds=threshold_s),
                )

        interval_duration_s = (
            current_interval_end - current_interval_start
        ).total_seconds()
        interval_billed_s = max(
            threshold_s, Trace._round_up(interval_duration_s, granularity_s)
        )
        total_billed_s += interval_billed_s

        return total_billed_s

    @property
    def query_ids(self) -> list[str]:
        """
        Get the query IDs of the queries in the trace.

        Returns:
            A pandas Series containing the query IDs.
        """
        all_query_ids = []
        for df in self._dfs["sys_query_history"].values():
            all_query_ids.extend(list(df["query_id"].unique()))

        return all_query_ids

    @staticmethod
    def extract_temp_and_q_idxs(query_text: str) -> "TPCDSTempAndQIdx":
        """
        Extract the TPC-DS template and query index from the given query text.

        Parameters:
            query_text: The text of the query.

        Returns:
            A tuple containing the template number and the query index.

        Raises:
            ValueError: If the query text does not contain a valid TPC-DS
                template and query index following the required format.
        """
        try:
            return query_text.split("\\n")[1][-11:-4].strip()

        except Exception as e:
            raise ValueError(
                "Query text does not contain a valid TPC-DS template and "
                "query index following the required format."
            ) from e

    @staticmethod
    def extract_temp(temp_and_q_idx: "TPCDSTempAndQIdx") -> int:
        """
        Extract the TPC-DS template number from the given template and query
        index string.

        Parameters:
            temp_and_q_idx: The TPC-DS template and query index string.

        Returns:
            The template number as an integer.
        """
        return int(str(temp_and_q_idx).split("_")[0])

    @staticmethod
    def extract_q_idx(temp_and_q_idx: "TPCDSTempAndQIdx") -> int:
        """
        Extract the TPC-DS query index from the given template and query index
        string.

        Parameters:
            temp_and_q_idx: The TPC-DS template and query index string.

        Returns:
            The query index as an integer.
        """
        return int(str(temp_and_q_idx).split("_")[1])

    @property
    def tpcds_temp_and_q_idxs(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the TPCDS template and query indices associated with each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the `query_ids` property.
        """
        # Check if there is a cached version of the TPCDS template and query
        # indices.
        run_dir = os.path.join(pu.get_runs_path(), self.run_id)
        tpcds_temp_and_q_idxs_path = os.path.join(
            run_dir, "tpcds_temp_and_q_idxs.parquet"
        )
        if os.path.exists(tpcds_temp_and_q_idxs_path):
            # Set the uuid to the query IDs in the cached series.
            concatenated = cast(
                pd.Series,
                pd.read_parquet(tpcds_temp_and_q_idxs_path).squeeze("columns"),
            )
            concatenated.index = pd.Index(
                [f"{q.split('#')[0]}#{self._uuid}" for q in concatenated.index]
            )
            return concatenated

        # If not, compute the TPCDS template and query indices.
        series = []
        for cluster_name, df in self._dfs["sys_query_history"].items():
            # Read another local copy of the query history for this cluster,
            # including the query text field.
            df_with_query_text = pd.read_parquet(
                os.path.join(
                    run_dir, f"sys_query_history+{cluster_name}.parquet"
                ),
                columns=["query_id", "query_text"],
            )

            # Adjust the query IDs and derive the TPCDS template and query
            # indices.
            df_with_query_text["query_id"] = df_with_query_text.apply(
                lambda r: f"{cluster_name}_{r['query_id']}#{self._uuid}",
                axis=1,
            )
            df_with_query_text["tup"] = df_with_query_text["query_text"].apply(
                self.extract_temp_and_q_idxs
            )
            s = df_with_query_text.set_index("query_id")["tup"]
            series.append(s)

        # Cache the TPCDS template and query indices for future use.
        concatenated = pd.concat(series).reindex(self.query_ids)
        concatenated.to_frame().to_parquet(
            tpcds_temp_and_q_idxs_path, index=True
        )
        return pd.Series(concatenated)

    def mbytes_scanned(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the total MB scanned per query.
        """
        series = []
        query_ids = self.query_ids
        for df in self._dfs["sys_query_detail"].values():
            condition = df["query_id"].isin(query_ids) & (
                df["step_name"] == "scan"
            )
            s = df[condition].groupby("query_id")["output_bytes"].sum()
            series.append(s)

        concatenated = pd.Series(
            pd.concat(series).reindex(query_ids) / Trace.BYTES_IN_MEGABYTE
        )
        return concatenated

    def arrival_times(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the arrival times (start times in SYS_QUERY_HISTORY) of each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the `query_ids` property.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("query_id")["start_time"]
            s = pd.to_datetime(s)
            series.append(s)

        return pd.concat(series).reindex(self.query_ids)

    def completion_times(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the completion times (end times in SYS_QUERY_HISTORY) of each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the `query_ids` property.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("query_id")["end_time"]
            s = pd.to_datetime(s)
            series.append(s)

        return pd.concat(series).reindex(self.query_ids)

    def query_plans(self) -> dict[str, Any]:
        """
        Parse the query plans for each query in the trace and return a
        dictionary mapping the query IDs to their parsed plans.
        """

        d = {}

        # Find out the name of the schema.
        run_params_path = os.path.join(
            pu.get_runs_path(), self._run_id, "run_params.yml"
        )
        with open(run_params_path, "r") as f:
            run_params = yaml.safe_load(f)
        schema_name = run_params["schema_name"]
        workload_name = run_params["workload_name"]

        # Load any pre-parsed plans.
        parsed_plans_path = os.path.join(
            pu.get_data_path(), "parsed_query_plans", f"{schema_name}.pkl"
        )
        parsed_plans: dict[Trace.TPCDSTempAndQIdx, Any] = {}
        if os.path.exists(parsed_plans_path):
            with open(parsed_plans_path, "rb") as f:
                parsed_plans = pickle.load(f)

        # Determine any queries still to be parsed and exit early if none.
        query_ids_to_parse_per_cluster: dict[
            str, list[tuple[str, Trace.TPCDSTempAndQIdx]]
        ] = defaultdict(list)
        for query_id, temp_and_q_idx in self.tpcds_temp_and_q_idxs.items():
            query_id = cast(str, query_id)
            temp_and_q_idx
            if temp_and_q_idx in parsed_plans:
                d[query_id] = parsed_plans[temp_and_q_idx]
            else:
                cluster = query_id.rsplit("_", maxsplit=1)[0]
                query_ids_to_parse_per_cluster[cluster].append(
                    (query_id, temp_and_q_idx)
                )
        if all(len(v) == 0 for v in query_ids_to_parse_per_cluster.values()):
            return d
        
        # Parse the remaining queries.
        for cluster_name, query_ids in query_ids_to_parse_per_cluster.items():
            explain_df = self._dfs["sys_query_explain"][cluster_name]

            for query_id, temp_and_q_idx in query_ids:
                query_df = explain_df[explain_df["query_id"] == query_id]
                if len(query_df) == 0:
                    continue
                plan_steps = query_df.sort_values("plan_node_id")[
                    "plan_node"
                ].tolist()
                verbose_plan, _, _ = parse_one_plan(plan_steps, analyze=False)
                alias_dict: dict[str, Optional[str]] = {}
                verbose_plan.parse_lines_recursively(
                    schema_name=schema_name,
                    alias_dict=alias_dict,
                )
                verbose_plan.parse_columns_bottom_up(
                    alias_dict=alias_dict,
                )

                # Get the tables.
                tables, _, _ = plan_summary(verbose_plan)
                verbose_plan_dict = verbose_plan.as_serializable()
                verbose_plan_dict["tables"] = sorted(list(tables))
                verbose_plan_dict["num_tables"] = len(tables)

                d[cast(str, query_id)] = verbose_plan_dict
                parsed_plans[temp_and_q_idx] = verbose_plan_dict

        # Cache the parsed plans for future use.
        with open(parsed_plans_path, "wb") as f:
            pickle.dump(parsed_plans, f)
        with open(parsed_plans_path.replace(".pkl", ".yml"), "w") as f:
            yaml.dump(parsed_plans, f, sort_keys=False)

        return d

    def was_aborted(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        booleans indicating whether each query was aborted.

        Note: This implementation assumes that each query is considered aborted
        if its status in SYS_QUERY_HISTORY does not contain "success". We don't
        do exact string match because there may be trailing whitespace.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = ~df.set_index("query_id")["status"].str.contains("success")
            series.append(s)

        return pd.concat(series).reindex(self.query_ids)

    def was_cached(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        booleans indicating whether each query was served from the result cache.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("query_id")["result_cache_hit"]
            series.append(s)

        return pd.concat(series).reindex(self.query_ids)

    def query_type(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the type of each query (e.g., 'SELECT', 'INSERT', etc.).
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("query_id")["query_type"]
            series.append(s)
        return pd.concat(series).reindex(self.query_ids)

    def _count_distinct_table_names_containing(
        self, substrings: list[str]
    ) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the count of distinct table names containing any of the specified
        substrings associated with each query.

        Parameters:
            substring: The substring to search for in table names.
        """
        series = []
        query_ids = self.query_ids
        joined_substrings = "|".join(substrings)
        for df in self._dfs["sys_query_detail"].values():
            condition = df["query_id"].isin(query_ids)
            s = (
                df[condition]
                .groupby("query_id")["table_name"]
                .apply(
                    lambda x: x.str.contains(joined_substrings, na=False).sum()
                )
            )
            series.append(s)
        return pd.concat(series).reindex(query_ids, fill_value=0)

    def num_external_tables(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the count of distinct external table names associated with each query.
        """
        # FIXME: Assume we don't have external tables for now.
        return pd.Series(0, index=self.query_ids)

    def num_system_tables(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the count of distinct system table names associated with each query.
        """
        return self._count_distinct_table_names_containing(
            Trace.REDSHIFT_SYSTEM_TABLE_SUBSTRINGS
        )

    def num_permanent_tables(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the count of distinct permanent table names associated with each query.
        """
        return self._count_distinct_table_names_containing(
            Trace.REDSHIFT_PERMANENT_TABLE_SUBSTRINGS
        )

    def _count_word_in_plan_rows(self, word: str) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the count of occurrences of the specified word in the plan nodes.

        FIXME: This lumps together the features from all clusters in the trace.
        We may want to separate them per cluster in the future to properly
        support multi-cluster traces, or at least document this behavior.
        """
        series = []
        query_ids = self.query_ids
        for df in self._dfs["sys_query_explain"].values():
            condition = df["query_id"].isin(query_ids)
            s = (
                df[condition]["plan_node"]
                .str.lower()
                .str.contains(word.lower(), na=False)
                .groupby(df["query_id"])
                .sum()
            )
            series.append(s)
        return pd.concat(series).reindex(query_ids)

    def num_joins(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the number of joins.
        """
        return self._count_word_in_plan_rows("join")

    def num_scans(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the number of scans.
        """
        return self._count_word_in_plan_rows("scan")

    def num_aggregates(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the number of aggregates.
        """
        return self._count_word_in_plan_rows("aggregate")

    def rpu_per_cluster(self) -> dict[str, int]:
        """
        Returns the RPU corresponding to each cluster of the blueprint on
        which this trace was executed.

        Returns:
            A dictionary mapping cluster names to their respective RPUs.
        """
        # Find out the name of the blueprint.
        run_params_path = os.path.join(
            pu.get_runs_path(), self._run_id, "run_params.yml"
        )
        with open(run_params_path, "r") as f:
            run_params = yaml.safe_load(f)
        blueprint_name = run_params["blueprint_name"]

        # For this blueprint, find out the cluster names and their RPUs.
        bp_cluster_names = pu.get_blueprint_dicts_from_config()[blueprint_name][
            "cluster_names"
        ]
        all_cluster_names_to_rpu = {
            k: v["rpu"] for k, v in pu.get_cluster_dicts_from_config().items()
        }
        bp_cluster_names_to_rpu = {
            name: all_cluster_names_to_rpu[name] for name in bp_cluster_names
        }
        return bp_cluster_names_to_rpu

    

