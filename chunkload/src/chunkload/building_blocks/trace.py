import pandas as pd


class Trace():
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
            col for col in Trace.REQUIRED_COLUMNS if col not in obj.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Trace DataFrame is missing required columns: "
                f"{', '.join(missing_columns)}"
            )

        # Check that elapsed_time is in microseconds.
        manual_diff_timedelta = obj["end_time"] - obj["start_time"]
        elapsed_time_timedelta = pd.to_timedelta(obj["elapsed_time"], unit="us")
        if not all(manual_diff_timedelta == elapsed_time_timedelta):
            raise ValueError(
                '"elapsed_time" column does not match the difference between '
                '"start_time" and "end_time".'
            )

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
        if not (0 < quantile < 1):
            raise ValueError("Quantile must be between 0 and 1.")
        return (
            self._obj["elapsed_time"].quantile(quantile)
            / self.MICROSECONDS_IN_SECOND
        )

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
        if self._obj.empty:
            return total_billed_s

        current_interval_start = self._obj.iloc[0]["start_time"]
        current_interval_end = max(
            self._obj.iloc[0]["end_time"],
            current_interval_start + pd.Timedelta(seconds=threshold_s),
        )

        for _, row in self._obj.iterrows():
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
