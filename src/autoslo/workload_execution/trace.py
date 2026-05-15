import os
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import autoslo.filesystem.path_utils as pu
import autoslo.tuner.parallelism as plu
from autoslo.clusters.cluster import Cluster
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.query_plans.parse_plan import parse_one_plan, plan_summary
from autoslo.workload_definition.query import ClusterAwareQueryId, QueryTextId
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
        "sys_serverless_usage": [
            "start_time",
            "end_time",
            "charged_seconds",
            "charged_extra_compute_for_automatic_optimization_seconds",
        ],
    }

    REDSHIFT_ELAPSED_TIME_UNIT = "us"  # microseconds
    BYTES_IN_MEGABYTE = 1_000_000

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

        # For caching.
        self._cluster_aware_query_ids: list[ClusterAwareQueryId] = []

        # A map from table_name to [a map of cluster_name to DataFrame].
        self._dfs: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        self._original_start = datetime.now()

        run_dir = os.path.join(pu.get_runs_path(), run_id)
        pq_filenames = [
            f for f in os.listdir(run_dir) if f.endswith(".parquet")
        ]

        # Pass 1: build redshift_query_id -> cluster_aware_query_id mapping
        # and the reverse cluster lookup by parsing the WorkloadRunner SQL
        # comment (``--{run_id}/{query_id}\n{sql}``) from every
        # sys_query_history file.
        self._redshift_to_cluster_aware: dict[str, ClusterAwareQueryId] = {}
        for filename in pq_filenames:
            parts = filename.split(".")[0].split("+")
            if len(parts) != 2 or parts[0] != "sys_query_history":
                continue
            cluster_name = parts[1]
            df_ids = pd.read_parquet(
                os.path.join(run_dir, filename),
                columns=["query_id", "query_text"],
            )
            if len(df_ids) == 0:
                continue
            wq_ids = df_ids["query_text"].apply(
                lambda t: (
                    t.split("\\n")[0].split("/")[1]
                    if t.startswith(f"--{self._run_id}/")
                    else None
                )
            )
            for raw_id, wq_id in zip(df_ids["query_id"].astype(str), wq_ids):
                if wq_id is None:
                    continue
                cluster_aware_id = ClusterAwareQueryId.make(cluster_name, wq_id)
                self._redshift_to_cluster_aware[raw_id] = cluster_aware_id

        # Pass 2: load all required parquet files.  For files that carry a
        # query_id column the raw Redshift integer is replaced with the
        # corresponding TaggedQueryId from the mapping built in pass 1.
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

                if "query_id" in df.columns:
                    known = (
                        df["query_id"]
                        .astype(str)
                        .isin(self._redshift_to_cluster_aware)
                    )
                    df = df[known].copy()
                    if len(df) == 0:
                        continue
                    df["cluster_aware_query_id"] = df["query_id"].apply(
                        lambda raw_id: self._redshift_to_cluster_aware[
                            str(raw_id)
                        ]
                    )
                    df = df.rename(columns={"query_id": "redshift_query_id"})

                if table_name == "sys_query_explain":
                    df = (
                        df[df["plan_node"].str.contains("XN")]
                        .sort_values(["cluster_aware_query_id", "plan_node_id"])
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
    def server_side_latencies_s(self) -> pd.Series:
        """Redshift server-side latencies (elapsed_time), indexed by
        ClusterAwareQueryId."""
        conversion_factor = pd.Timedelta(
            1, Trace.REDSHIFT_ELAPSED_TIME_UNIT  # type: ignore
        ).total_seconds()

        series = []
        for df in self._dfs["sys_query_history"].values():
            s = (
                df.set_index("cluster_aware_query_id")["elapsed_time"].astype(
                    "float"
                )
                * conversion_factor
            )
            series.append(s)

        return pd.concat(series).reindex(self.cluster_aware_query_ids)

    @property
    def structured_log(self) -> Optional[StructuredLog]:
        """Lazily loaded StructuredLog for this run, or None if absent."""
        if not hasattr(self, "_structured_log"):
            path = (
                Path(pu.get_runs_path())
                / self._run_id
                / "structured_log.parquet"
            )
            self._structured_log: Optional[StructuredLog] = (
                StructuredLog.load(path) if path.exists() else None
            )
        return self._structured_log

    def _client_side_index(self) -> dict[str, ClusterAwareQueryId]:
        """Map workload query_id string -> ClusterAwareQueryId."""
        return {caqid.query_id: caqid for caqid in self.cluster_aware_query_ids}

    @property
    def client_side_latencies_s(self) -> pd.Series:
        """Client-side latencies (COMPLETION - ARRIVAL) from the structured log.

        Raises ValueError if no structured log is present.
        """
        if self.structured_log is None:
            raise ValueError(
                f"No structured_log.parquet for run {self._run_id!r}"
            )
        idx = self._client_side_index()
        df = self.structured_log.query_latencies()
        s = df.set_index("query_id")["latency_s"].rename(index=idx)
        return s.reindex(self.cluster_aware_query_ids)

    @property
    def client_side_arrival_times_s(self) -> pd.Series:
        """Client-side arrival timestamps (seconds relative to run start).

        Raises ValueError if no structured log is present.
        """
        if self.structured_log is None:
            raise ValueError(
                f"No structured_log.parquet for run {self._run_id!r}"
            )
        idx = self._client_side_index()
        df = self.structured_log.query_latencies()
        s = df.set_index("query_id")["arrival_s"].rename(index=idx)
        return s.reindex(self.cluster_aware_query_ids)

    @property
    def client_side_completion_times_s(self) -> pd.Series:
        """Client-side completion timestamps (seconds relative to run start).

        Raises ValueError if no structured log is present.
        """
        if self.structured_log is None:
            raise ValueError(
                f"No structured_log.parquet for run {self._run_id!r}"
            )
        idx = self._client_side_index()
        df = self.structured_log.query_latencies()
        s = df.set_index("query_id")["completion_s"].rename(index=idx)
        return s.reindex(self.cluster_aware_query_ids)

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

        df = self._dfs["sys_serverless_usage"][cluster_name]
        charged_seconds = df["charged_seconds"].sum()
        if "extra_compute_for_automatic_optimization_seconds" in df.columns:
            charged_seconds += df[
                "charged_extra_compute_for_automatic_optimization_seconds"
            ].sum()
        return charged_seconds / 3600 * Cluster.US_EAST_1_COST_PER_RPU_HOUR

    @property
    def cluster_aware_query_ids(self) -> list[ClusterAwareQueryId]:
        """
        Get the cluster-aware query IDs of the queries in the trace.

        Returns:
            A list of cluster-aware query IDs.
        """
        if len(self._cluster_aware_query_ids) == 0:
            for df in self._dfs["sys_query_history"].values():
                self._cluster_aware_query_ids.extend(
                    list(df["cluster_aware_query_id"].unique())
                )

        return self._cluster_aware_query_ids

    @property
    def seq_nums(self) -> pd.Series:
        """
        Get the sequence numbers of the queries in the trace.

        Returns:
            A pandas Series containing the sequence numbers.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("cluster_aware_query_id")["query_text"].apply(
                lambda x: int(x.split("\\n")[0].split("/")[1])
            )
            series.append(s)

        return pd.concat(series).reindex(self.cluster_aware_query_ids)

    @staticmethod
    def _get_query_text_id(query_text: str) -> QueryTextId:
        """
        Extract the query text ID from the given TPC-DS query text.

        The start of the querytext is assumed to look like:
        --1778822203993/query_1\n-- ext_tpcds1000#028#002\n\n...

        Parameters:
            query_text: The text of the query.

        Returns:
            The :class:`QueryTextId` for the query.

        Raises:
            ValueError: If the query text does not contain a valid TPC-DS
                query text ID following the required format.
        """
        try:
            raw_id = query_text.split("\\n")[1][3:].strip()
            return QueryTextId(raw_id)
        except Exception as e:
            raise ValueError(
                "Query text does not contain a valid TPC-DS query text ID "
                "following the required format."
            ) from e

    @property
    def query_text_ids(self) -> pd.Series:
        """
        Return a Series where the index is the cluster-awarequery IDs and the
        values are :class:`~autoslo.workload_definition.query.QueryTextId`
        objects associated with each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the ``query_ids`` property.
        """
        run_dir = os.path.join(pu.get_runs_path(), self.run_id)
        cache_path = os.path.join(run_dir, "query_text_ids.parquet")

        if os.path.exists(cache_path):
            concatenated = cast(
                pd.Series,
                pd.read_parquet(cache_path).squeeze("columns").map(QueryTextId),
            )
            # Invalidate caches written by older formats:
            if (
                not concatenated.empty
                and concatenated.index.name != "cluster_aware_query_id"
            ):
                os.remove(cache_path)
            else:
                return concatenated

        # Compute query text IDs from the raw query texts.
        series = []
        for cluster_name in self.seen_clusters:
            df = pd.read_parquet(
                os.path.join(
                    run_dir, f"sys_query_history+{cluster_name}.parquet"
                ),
                columns=["query_id", "query_text"],
            )
            df["cluster_aware_query_id"] = df["query_id"].apply(
                lambda raw_id: (self._redshift_to_cluster_aware[str(raw_id)])
            )
            df["query_text_id"] = df["query_text"].map(Trace._get_query_text_id)
            s = df.set_index("cluster_aware_query_id")["query_text_id"]
            series.append(s)

        concatenated = (
            pd.concat(series)
            .reindex(self.cluster_aware_query_ids)
            .map(QueryTextId)
        )
        concatenated.to_frame().to_parquet(cache_path, index=True)
        return concatenated

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
                events.append(
                    (row["start_time"], "start", row["cluster_aware_query_id"])
                )
                events.append(
                    (row["end_time"], "end", row["cluster_aware_query_id"])
                )

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

        return pd.Series(non_overlapping_dict).reindex(
            self.cluster_aware_query_ids
        )

    def arrival_times(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the arrival times (start times in SYS_QUERY_HISTORY) of each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the `cluster_aware_query_ids` property.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("cluster_aware_query_id")["start_time"]
            s = pd.to_datetime(s)
            series.append(s)

        return pd.concat(series).reindex(self.cluster_aware_query_ids)

    def completion_times(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the completion times (end times in SYS_QUERY_HISTORY) of each query.

        The order of the query IDs in the Series matches the order of the query
        IDs provided by the `cluster_aware_query_ids` property.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("cluster_aware_query_id")["end_time"]
            s = pd.to_datetime(s)
            series.append(s)

        return pd.concat(series).reindex(self.cluster_aware_query_ids)

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
            str, list[tuple[ClusterAwareQueryId, QueryTextId]]
        ] = defaultdict(list)
        for (
            cluster_aware_query_id,
            query_text_id,
        ) in self.query_text_ids.items():
            cluster_aware_query_id = cast(
                ClusterAwareQueryId, cluster_aware_query_id
            )
            cached_plan = (
                None if ignore_caching else QueryPlanRegistry.get(query_text_id)
            )
            if cached_plan is not None:
                d[cluster_aware_query_id] = cached_plan
            else:
                cluster_name = cluster_aware_query_id.cluster_name
                query_ids_to_parse_per_cluster[cluster_name].append(
                    (cluster_aware_query_id, query_text_id)
                )
        if all(len(v) == 0 for v in query_ids_to_parse_per_cluster.values()):
            return d

        # Parse the remaining queries.
        new_plans: dict[str, Any] = {}
        for cluster_name, query_ids in query_ids_to_parse_per_cluster.items():
            explain_df = self._dfs["sys_query_explain"][cluster_name]

            for cluster_aware_query_id, query_text_id in query_ids:
                query_df = explain_df[
                    explain_df["cluster_aware_query_id"]
                    == cluster_aware_query_id
                ]
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

                d[cluster_aware_query_id] = verbose_plan_dict
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
            s = ~df.set_index("cluster_aware_query_id")["status"].str.contains(
                "success"
            )
            series.append(s)

        return pd.concat(series).reindex(self.cluster_aware_query_ids)

    def error_messages(self) -> pd.Series:
        """
        Return a Series where the index is the query IDs and the values are
        the error messages associated with each query.
        """
        series = []
        for df in self._dfs["sys_query_history"].values():
            s = df.set_index("cluster_aware_query_id")["error_message"]
            series.append(s)
        return (
            pd.concat(series).reindex(self.cluster_aware_query_ids).str.strip()
        )

    def sys_query_explain_rows_per_query(self) -> dict[str, pd.DataFrame]:
        """
        Return a dictionary mapping query IDs to their corresponding rows in
        SYS_QUERY_EXPLAIN. Ignore any rows corresponding to preempted child
        queries as indicated by the error messages.
        """
        d: dict[str, pd.DataFrame] = {}
        error_messages = self.error_messages()

        for df in self._dfs["sys_query_explain"].values():

            for cluster_aware_query_id, query_df in df.groupby(
                "cluster_aware_query_id"
            ):
                if cluster_aware_query_id not in self.cluster_aware_query_ids:
                    continue

                # If the error message specifies a preempted child query, skip
                # the rows for that child query.
                error_message = error_messages[cluster_aware_query_id].strip()
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
                d[cluster_aware_query_id] = (
                    query_df.sort_values(
                        ["child_query_sequence", "plan_node_id"]
                    )
                    .reset_index(drop=True)
                    .copy()
                )
        return d
