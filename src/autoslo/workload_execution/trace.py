import os
from datetime import datetime, timezone

import pandas as pd

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster import Cluster
from collections import defaultdict

import pyarrow.parquet as pq


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
        ],
        "sys_query_detail": [
            "query_id",
            "step_name",
            "output_bytes",
        ],
    }

    REDSHIFT_BILLING_THRESHOLD_S = 60
    REDSHIFT_BILLING_GRANULARITY_S = 1
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
                df["query_id"] = df["query_id"].apply(
                    lambda x: f"{cluster_name}_{x}"
                )
                self._dfs[table_name][cluster_name] = df

                if table_name == "sys_query_history":
                    min_start_time = df["start_time"].min()
                    if min_start_time < self._original_start:
                        self._original_start = min_start_time

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
        return pd.read_parquet(path, columns=column_list)

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
    def latencies_s(self) -> list[float]:
        """
        Get the latencies of the queries in the trace, in seconds.

        Returns:
            A pandas Series containing the latencies in seconds.
        """
        conversion_factor = pd.Timedelta(
            1, Trace.REDSHIFT_ELAPSED_TIME_UNIT  # type: ignore
        ).total_seconds()

        latencies = []
        for df in self._dfs["sys_query_history"].values():
            elapsed_times_in_s = (
                df["elapsed_time"].astype("float") * conversion_factor
            )
            latencies.extend(elapsed_times_in_s.tolist())

        return latencies

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

    def mbytes_scanned(self) -> pd.Series:
        """
        Return a Series with the total MB scanned per query.
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

    def num_joins(self) -> float:
        """
        Placeholder method to return mean number of joins.
        """
        # Placeholder implementation - need to get from SYS_QUERY_EXPLAIN
        return 0.0

    def num_scans(self) -> float:
        """
        Placeholder method to return mean number of scans.
        """
        # Placeholder implementation - need to get from SYS_QUERY_EXPLAIN
        return 0.0

    def num_aggregations(self) -> float:
        """
        Placeholder method to return mean number of aggregations.
        """
        # Placeholder implementation - need to get from SYS_QUERY_EXPLAIN
        return 0.0
