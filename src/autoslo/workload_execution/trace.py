import os
from datetime import datetime

import pandas as pd

import autoslo.utils.paths as pu


class Trace:
    """
    A query execution trace. This class is used to abstract away the details of
    the trace data as collected from the database engine.
    """

    REQUIRED_COLUMNS = ["start_time", "end_time", "elapsed_time"]
    MICROSECONDS_IN_SECOND = 1_000_000
    REDSHIFT_BILLING_THRESHOLD_S = 60
    REDSHIFT_BILLING_GRANULARITY_S = 1

    def __init__(self, trace_df: pd.DataFrame) -> None:
        """
        Initialize a Trace instance.

        Parameters:
            trace_df: A pandas DataFrame containing the trace data.
        """
        self._validate(trace_df)
        self._trace_df = trace_df.sort_values(
            by="start_time", ascending=True
        ).reset_index(drop=True)
        self._original_start = self._trace_df["start_time"].min()

    @property
    def trace_df(self) -> pd.DataFrame:
        """Get the underlying trace DataFrame."""
        return self._trace_df

    @staticmethod
    def from_run(run_id: str) -> "Trace":
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
        run_dir = os.path.join(pu.get_runs_path(), run_id)
        cached_trace_path = os.path.join(run_dir, "trace.parquet")

        if os.path.exists(cached_trace_path):
            trace_df = pd.read_parquet(
                cached_trace_path, columns=Trace.REQUIRED_COLUMNS
            )
            return Trace(trace_df)

        trace_path = None
        for fname in os.listdir(run_dir):
            if fname.startswith("sys_query_history") and fname.endswith(
                ".parquet"
            ):
                trace_path = os.path.join(run_dir, fname)
                break
        if trace_path is None:
            raise ValueError(
                "No sys_query_history Parquet file found in run directory."
            )
        trace_df = pd.read_parquet(trace_path, columns=Trace.REQUIRED_COLUMNS)

        # Cache the trace for future use.
        trace_df.to_parquet(cached_trace_path, index=False)
        return Trace(trace_df)

    @staticmethod
    def _validate(trace_df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame contains the required columns and that the
        "elapsed_time" column is in microseconds.

        Parameters:
            trace_df: A pandas DataFrame to validate.

        Raises:
            ValueError: If any of the required columns are missing.
            ValueError: If the "elapsed_time" column is not in microseconds.
        """
        # Check for required columns
        missing_columns = [
            col for col in Trace.REQUIRED_COLUMNS if col not in trace_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Trace DataFrame is missing required columns: "
                f"{', '.join(missing_columns)}"
            )

        # Check that elapsed_time is in microseconds.
        manual_diff_timedelta = trace_df["end_time"] - trace_df["start_time"]
        elapsed_time_timedelta = pd.to_timedelta(
            trace_df["elapsed_time"], unit="us"
        )
        if not all(manual_diff_timedelta == elapsed_time_timedelta):
            raise ValueError(
                '"elapsed_time" column does not match the difference between '
                '"start_time" and "end_time".'
            )

    @staticmethod
    def concat(traces: list["Trace"]) -> "Trace":
        """
        Concatenate multiple Trace instances into a single Trace instance.

        Parameters:
            traces: A list of Trace instances to concatenate.

        Returns:
            A new Trace instance containing the concatenated trace data.
        """
        concatenated_df = pd.concat(
            [trace.trace_df for trace in traces]
        ).reset_index(drop=True)
        return Trace(concatenated_df)

    def normalize_start_to(self, new_start: datetime) -> "Trace":
        """
        Normalize start and end times of the trace so that the earliest start
        time is equal to `new_start`.

        Parameters:
            new_start: A pandas Timestamp to which the earliest start time
                will be normalized.

        Returns:
            A new Trace instance with normalized start times.
        """
        earliest_start = self._trace_df["start_time"].min()
        shift = pd.Timestamp(new_start) - earliest_start
        normalized_df = self._trace_df.copy()
        normalized_df["start_time"] = normalized_df["start_time"] + shift
        normalized_df["end_time"] = normalized_df["end_time"] + shift
        self._trace_df = normalized_df
        return self

    def reset_start(self) -> "Trace":
        """
        Reset start and end times of the trace to their original values.

        Returns:
            A new Trace instance with original start times.
        """
        return self.normalize_start_to(self._original_start)

    def latency_s_at(self, quantile: float) -> float:
        """
        Get the query latency at a given quantile, in seconds.

        Parameters:
            quantile: The quantile to compute the latency for (between 0 and 1).

        Returns:
            The latency at the specified quantile.

        Raises:
            ValueError: If the quantile is not between 0 and 1.
        """
        if not (0 <= quantile <= 1):
            raise ValueError("Quantile must be between 0 and 1.")
        return (
            self._trace_df["elapsed_time"].quantile(quantile)
            / self.MICROSECONDS_IN_SECOND
        )

    def num_queries(self) -> int:
        """
        Get the total number of queries in the trace.

        Returns:
            The total number of queries.
        """
        return len(self._trace_df)

    def num_queries_with_latency_over(self, latency_s: float) -> int:
        """
        Get the number of queries with latency over a given threshold.

        Parameters:
            latency_s: The latency threshold in seconds.

        Returns:
            The number of queries with latency over the specified threshold.
        """
        latency_us = latency_s * self.MICROSECONDS_IN_SECOND
        return (self._trace_df["elapsed_time"] > latency_us).sum()

    def _round_up(self, value: float, granularity: float) -> float:
        """
        Round up a value to the nearest multiple of granularity.

        Parameters:
            value: The value to round up.
            granularity: The granularity to round up to.
        """
        return granularity * ((value + granularity - 1) // granularity)

    def billed_s(
        self,
        threshold_s: float = REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = REDSHIFT_BILLING_GRANULARITY_S,
    ) -> float:
        """
        Get the total billed time for the trace, in seconds.

        Parameters:
            threshold_s: The billing threshold in seconds. This is the minimum
                time that will be billed - i.e. all smaller intervals are
                rounded up to this threshold.
            granularity_s: The billing granularity in seconds. This is the time
                interval to which the billed time is rounded up.

        Returns:
            The total billed time for the trace, in seconds.
        """

        total_billed_s = 0.0
        if self._trace_df.empty:
            return total_billed_s

        current_interval_start = self._trace_df.iloc[0]["start_time"]
        current_interval_end = max(
            self._trace_df.iloc[0]["end_time"],
            current_interval_start + pd.Timedelta(seconds=threshold_s),
        )

        for _, row in self._trace_df.iterrows():
            start_time = row["start_time"]
            end_time = row["end_time"]

            if start_time <= current_interval_end:
                current_interval_end = max(current_interval_end, end_time)
            else:
                interval_duration_s = (
                    current_interval_end - current_interval_start
                ).total_seconds()
                billed_duration_s = max(
                    threshold_s,
                    self._round_up(interval_duration_s, granularity_s),
                )
                total_billed_s += billed_duration_s

                current_interval_start = start_time
                current_interval_end = max(
                    end_time,
                    current_interval_start + pd.Timedelta(seconds=threshold_s),
                )

        interval_duration_s = (
            current_interval_end - current_interval_start
        ).total_seconds()
        billed_duration_s = max(
            threshold_s, self._round_up(interval_duration_s, granularity_s)
        )
        total_billed_s += billed_duration_s

        return total_billed_s

    def mbytes_scanned_mean(self) -> float:
        """
        Placeholder method to return mean MB scanned.
        """
        # Placeholder implementation - need to get from SYS_QUERY_DETAIL
        return 0.0

    def num_joins_mean(self) -> float:
        """
        Placeholder method to return mean number of joins.
        """
        # Placeholder implementation - need to get from SYS_QUERY_EXPLAIN
        return 0.0

    def num_scans_mean(self) -> float:
        """
        Placeholder method to return mean number of scans.
        """
        # Placeholder implementation - need to get from SYS_QUERY_EXPLAIN
        return 0.0

    def num_aggregations_mean(self) -> float:
        """
        Placeholder method to return mean number of aggregations.
        """
        # Placeholder implementation - need to get from SYS_QUERY_EXPLAIN
        return 0.0
