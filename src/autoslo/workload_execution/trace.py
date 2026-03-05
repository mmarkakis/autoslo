import os
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from intervaltree import Interval  # type: ignore[import]

import autoslo.utils.paralellism as plu
import autoslo.utils.paths as pu
from autoslo.blueprints.cluster import Cluster
from autoslo.query_plans.parse_plan import parse_one_plan, plan_summary
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import QueryTextId
from autoslo.workload_definition.query_plan_registry import QueryPlanRegistry


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
            "query_text",
            "error_message",
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
            "child_query_sequence",
        ],
    }

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

        # For caching.
        self._query_ids: list[str] = []

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
    def cluster_name_from_query_id(query_id: str) -> str:
        """
        Extract the cluster name from a query ID.

        Parameters:
            query_id: The query ID from which to extract the cluster name.

        Returns:
            The cluster name as a string.
        """
        return query_id.rsplit("_", maxsplit=1)[0]

    @staticmethod
    def redshift_query_id_from_query_id(query_id: str) -> str:
        """
        Extract the Redshift query ID from a query ID.

        Parameters:
            query_id: The query ID from which to extract the Redshift query ID.

        Returns:
            The Redshift query ID as a string.
        """
        return query_id.rsplit("_", maxsplit=1)[-1].split("#", maxsplit=1)[0]

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
        query_intervals = [
            Interval(begin=start.timestamp(), end=end.timestamp())
            for start, end in zip(df["start_time"], df["end_time"])
        ]
        billed_s = Billing.billed_s(query_intervals=query_intervals)
        rpu = Cluster.rpu_for_cluster_name(cluster_name)
        return billed_s * Cluster.cost_per_second_for_rpu(rpu)

    @property
    def query_ids(self) -> list[str]:
        """
        Get the query IDs of the queries in the trace.

        Returns:
            A pandas Series containing the query IDs.
        """
        if len(self._query_ids) == 0:
            for df in self._dfs["sys_query_history"].values():
                self._query_ids.extend(list(df["query_id"].unique()))

        return self._query_ids

    @property
    def seq_nums(self) -> pd.Series:
        """
        Get the sequence numbers of the queries in the trace.

        Returns:
            A pandas Series containing the sequence numbers.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("query_id")["query_text"].apply(
                lambda x: int(x.split("\\n")[0].split("/")[1])
            )
            series.append(s)

        return pd.concat(series).reindex(self.query_ids)

    @staticmethod
    def extract_query_text_id(
        query_text: str,
        schema_name: str,
        has_prepended_run_information: bool = True,
    ) -> QueryTextId:
        """
        Extract the query text ID from the given TPC-DS query text.

        The ID is encoded in the SQL comment prepended by the TPC-DS query
        generator (e.g. ``"42_001"``).  The returned :class:`QueryTextId`
        combines *schema_name* with the extracted template/index pair in the
        canonical ``"schema#template#index"`` format.

        Parameters:
            query_text: The text of the query.
            schema_name: The schema the query belongs to (e.g.
                ``"ext_tpcds1000"``).  Used to construct the full
                :class:`QueryTextId`.
            has_prepended_run_information: Whether the query text has prepended
                run information.

        Returns:
            The :class:`QueryTextId` for the query.

        Raises:
            ValueError: If the query text does not contain a valid TPC-DS
                query text ID following the required format.
        """
        try:
            idx = 1 if has_prepended_run_information else 0
            raw_id = query_text.split("\\n")[idx][-11:-4].strip()
            template_id, query_index = raw_id.split("_", 1)
            return QueryTextId(f"{schema_name}#{template_id}#{query_index}")

        except Exception as e:
            raise ValueError(
                "Query text does not contain a valid TPC-DS query text ID "
                "following the required format."
            ) from e

    @property
    def query_text_ids(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        :class:`~autoslo.workload_definition.query.QueryTextId` objects
        associated with each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the ``query_ids`` property.

        The cache file stores the ``.value`` strings in the canonical
        ``"schema#template#index"`` format.  Old cache files that used the
        bare ``"template_index"`` format are detected and invalidated
        automatically.
        """
        run_dir = os.path.join(pu.get_runs_path(), self.run_id)
        cache_path = os.path.join(run_dir, "query_text_ids.parquet")

        # Read schema_name once — needed for both cache-miss and stale-check.
        run_params_path = os.path.join(run_dir, "run_params.yml")
        with open(run_params_path, "r") as f:
            run_params = yaml.safe_load(f)
        schema_name = run_params["schema_name"]

        if os.path.exists(cache_path):
            concatenated = cast(
                pd.Series,
                pd.read_parquet(cache_path).squeeze("columns"),
            )
            # Stale-cache guard: old format stored bare "template_index"
            # strings without a '#'.  Detect and invalidate.
            if not concatenated.empty and "#" not in str(concatenated.iloc[0]):
                os.remove(cache_path)
            else:
                # Re-attach the current UUID so joins against live query IDs work.
                concatenated.index = pd.Index(
                    [f"{q.split('#')[0]}#{self._uuid}" for q in concatenated.index]
                )
                return concatenated.map(QueryTextId)

        # Compute query text IDs from the raw query texts.
        series = []
        for cluster_name, df in self._dfs["sys_query_history"].items():
            df_with_query_text = pd.read_parquet(
                os.path.join(
                    run_dir, f"sys_query_history+{cluster_name}.parquet"
                ),
                columns=["query_id", "query_text"],
            )
            df_with_query_text["query_id"] = df_with_query_text.apply(
                lambda r: f"{cluster_name}_{r['query_id']}#{self._uuid}",
                axis=1,
            )
            df_with_query_text["query_text_id"] = df_with_query_text[
                "query_text"
            ].apply(
                lambda t: Trace.extract_query_text_id(t, schema_name).value
            )
            s = df_with_query_text.set_index("query_id")["query_text_id"]
            series.append(s)

        concatenated = pd.concat(series).reindex(self.query_ids)
        concatenated.to_frame().to_parquet(cache_path, index=True)
        return concatenated.map(QueryTextId)

    def query_is_non_overlapping(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        booleans indicating whether each query does not overlap with any other
        query in the trace.
        """
        non_overlapping_dict: dict[str, bool] = {}
        for df in self._dfs["sys_query_history"].values():
            events = []
            for _, row in df.iterrows():
                events.append((row["start_time"], "start", row["query_id"]))
                events.append((row["end_time"], "end", row["query_id"]))

            # Sort events by time, with 'end' events before 'start' events at
            # the same time.
            events.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

            active_queries: set[str] = set()
            for event_time, event_type, query_id in events:
                if event_type == "start":
                    if len(active_queries) == 0:
                        non_overlapping_dict[query_id] = True
                    else:
                        non_overlapping_dict[query_id] = False
                        for active_query in active_queries:
                            non_overlapping_dict[active_query] = False
                    active_queries.add(query_id)
                elif event_type == "end":
                    active_queries.remove(query_id)

        return pd.Series(non_overlapping_dict).reindex(self.query_ids)

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

    @property
    def routing_decisions(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the routing decisions (cluster names) of each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the `query_ids` property.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("query_id", drop=False)["query_id"].apply(
                lambda x: Trace.cluster_name_from_query_id(x)
            )
            series.append(s)

        return pd.concat(series).reindex(self.query_ids)

    def query_plans(self, ignore_caching: bool = False) -> dict[str, Any]:
        """
        Parse the query plans for each query in the trace and return a
        dictionary mapping the query IDs to their parsed plans.

        Parameters:
            ignore_caching: If True, ignore any cached parsed plans and
                re-parse all query plans. Also don't dump the parsed plans to
                cache after parsing.
        """

        d = {}

        # Find out the name of the schema.
        run_params_path = os.path.join(
            pu.get_runs_path(), self._run_id, "run_params.yml"
        )
        with open(run_params_path, "r") as f:
            run_params = yaml.safe_load(f)
        schema_name = run_params["schema_name"]

        # Determine any queries still to be parsed and exit early if none.
        query_ids_to_parse_per_cluster: dict[
            str, list[tuple[str, QueryTextId]]
        ] = defaultdict(list)
        for query_id, query_text_id in self.query_text_ids.items():
            query_id = cast(str, query_id)
            cached_plan = (
                None
                if ignore_caching
                else QueryPlanRegistry.get(query_text_id)
            )
            if cached_plan is not None:
                d[query_id] = cached_plan
            else:
                cluster = query_id.rsplit("_", maxsplit=1)[0]
                query_ids_to_parse_per_cluster[cluster].append(
                    (query_id, query_text_id)
                )
        if all(len(v) == 0 for v in query_ids_to_parse_per_cluster.values()):
            return d

        # Parse the remaining queries.
        new_plans: dict[str, Any] = {}
        for cluster_name, query_ids in query_ids_to_parse_per_cluster.items():
            explain_df = self._dfs["sys_query_explain"][cluster_name]

            for query_id, query_text_id in query_ids:
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
                new_plans[query_text_id] = verbose_plan_dict

        # Persist newly parsed plans to the QueryPlanRegistry.
        if not ignore_caching and new_plans:
            QueryPlanRegistry.update(schema_name, new_plans, save=True)

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

    def error_messages(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the error messages associated with each query.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("query_id")["error_message"]
            series.append(s)
        return pd.concat(series).reindex(self.query_ids).str.strip()

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

    def sys_query_explain_rows_per_query(self) -> dict[str, pd.DataFrame]:
        """
        Return a dictionary mapping query IDs to their corresponding rows in
        SYS_QUERY_EXPLAIN. Ignore any rows corresponding to preempted child 
        queries as indicated by the error messages.
        """
        d: dict[str, pd.DataFrame] = {}
        error_messages = self.error_messages()

        for df in self._dfs["sys_query_explain"].values():

            for query_id, query_df in df.groupby("query_id"):
                query_id = cast(str, query_id)  # Make mypy happy
                if query_id not in self.query_ids:
                    continue

                # If the error message specifies a preempted child query, skip
                # the rows for that child query.
                error_message = error_messages[query_id].strip()
                if (
                    len(error_message) > 0
                    and ("child_sequence:" in error_message)
                    and ("user's request" not in error_message)
                ):
                    problematic_child = int(
                        error_message.split("child_sequence:")[-1][0]
                    )
                    query_df = query_df[
                        query_df["child_query_sequence"] != problematic_child
                    ]

                # Sort by child query sequence and plan node ID.
                d[query_id] = (
                    query_df.sort_values(
                        ["child_query_sequence", "plan_node_id"]
                    )
                    .reset_index(drop=True)
                    .copy()
                )
        return d
