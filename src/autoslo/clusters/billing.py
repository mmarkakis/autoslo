import math
from typing import NamedTuple, Optional

import pandas as pd


class BillingInterval(NamedTuple):
    start: float
    end: float


class Billing:
    REDSHIFT_BILLING_THRESHOLD_S = 60
    REDSHIFT_BILLING_GRANULARITY_S = 1

    @staticmethod
    def _round_up(value: float, granularity: float) -> float:
        """
        Round up a value to the nearest multiple of granularity.

        Parameters:
            value: The value to round up.
            granularity: The granularity to round up to.
        """
        if granularity == 0:
            return value
        return math.ceil(value / granularity) * granularity

    @staticmethod
    def billed_s(
        query_intervals: list[BillingInterval],
        threshold_s: float = REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = REDSHIFT_BILLING_GRANULARITY_S,
    ) -> float:
        """
        Calculate the total billed time given the execution intervals of each
        query, considering the billing threshold and granularity.

        Parameters:
            query_intervals: The execution intervals of the queries.
            threshold_s: The billing threshold in seconds. This is the minimum
                time that will be billed - i.e. all smaller intervals are
                rounded up to this threshold.
            granularity_s: The billing granularity in seconds. Each billed
                interval is rounded up to the nearest multiple of this
                granularity.

        Returns:
            The total billed time in seconds.
        """
        billed_intervals = Billing.billed_intervals(
            query_intervals, threshold_s, granularity_s
        )
        if len(billed_intervals) == 0:
            return 0.0
        total_billed_s = sum(iv.end - iv.start for iv in billed_intervals)
        return float(total_billed_s)

    @staticmethod
    def billed_s_from_df(
        df: pd.DataFrame,
        start_col_name: str = "start",
        end_col_name: str = "end",
        threshold_s: float = REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = REDSHIFT_BILLING_GRANULARITY_S,
    ) -> float:
        """
        Calculate the total billed time given a dataframe with query execution
        intervals, considering the billing threshold and granularity.

        Parameters:
            df: A dataframe containing the query execution intervals. It must
                have columns for the start and end times of the intervals, as
                well as a query_id column.
            start_col_name: The name of the column in the dataframe that
                contains the start times of the intervals.
            end_col_name: The name of the column in the dataframe that contains
                the end times of the intervals.
            threshold_s: The billing threshold in seconds. This is the minimum
                time that will be billed - i.e. all smaller intervals are
                rounded up to this threshold.
            granularity_s: The billing granularity in seconds. Each billed
                interval is rounded up to the nearest multiple of this
                granularity.

        Returns:
            The total billed time in seconds.
        """

        query_intervals = []

        # Convert start/end columns to Unix timestamps if they are datetimes.
        if pd.api.types.is_datetime64_any_dtype(df[start_col_name]):
            df[start_col_name] = (
                df[start_col_name] - pd.Timestamp("1970-01-01")
            ) / pd.Timedelta("1s")
        if pd.api.types.is_datetime64_any_dtype(df[end_col_name]):
            df[end_col_name] = (
                df[end_col_name] - pd.Timestamp("1970-01-01")
            ) / pd.Timedelta("1s")

        for s, e in zip(df[start_col_name], df[end_col_name]):
            interval = BillingInterval(s, e)
            query_intervals.append(interval)

        return Billing.billed_s(query_intervals, threshold_s, granularity_s)

    @staticmethod
    def billed_intervals(
        query_intervals: list[BillingInterval],
        threshold_s: float = REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = REDSHIFT_BILLING_GRANULARITY_S,
    ) -> list[BillingInterval]:
        """
        Calculate the billed intervals given the execution intervals of each
        query, considering the billing threshold and granularity.

        Parameters:
            query_intervals: The execution intervals of the queries.
            threshold_s: The billing threshold in seconds. This is the minimum
                time that will be billed - i.e. all smaller intervals are
                rounded up to this threshold.
            granularity_s: The billing granularity in seconds. Each billed
                interval is rounded up to the nearest multiple of this
                granularity.

        Returns:
            The list of billed intervals.
        """

        billed_intervals: list[BillingInterval] = []
        if not query_intervals or len(query_intervals) == 0:
            return billed_intervals

        ivs = sorted(query_intervals, key=lambda iv: (iv.start, iv.end))

        current_start = ivs[0].start
        current_end = max(ivs[0].end, current_start + threshold_s)

        for iv in ivs[1:]:
            start_time, end_time = iv.start, iv.end

            if start_time <= current_end:
                current_end = max(current_end, end_time)
            else:
                duration_s = current_end - current_start
                billed_duration_s = max(
                    threshold_s,
                    Billing._round_up(duration_s, granularity_s),
                )
                billed_interval = BillingInterval(
                    current_start,
                    current_start + billed_duration_s,
                )

                billed_intervals.append(billed_interval)
                current_start = start_time
                current_end = max(end_time, current_start + threshold_s)

        duration_s = current_end - current_start
        billed_duration_s = max(
            threshold_s, Billing._round_up(duration_s, granularity_s)
        )
        billed_interval = BillingInterval(
            current_start,
            current_start + billed_duration_s,
        )
        billed_intervals.append(billed_interval)

        return billed_intervals


class BillingAccumulator:
    """Incrementally maintains the total billed seconds for a sequence of
    chronologically ordered, non-overlapping raw billing intervals.

    All threshold extension and merge logic lives here; callers never touch
    those details.  Every public method is O(1).
    """

    def __init__(
        self,
        threshold_s: float = Billing.REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = Billing.REDSHIFT_BILLING_GRANULARITY_S,
    ) -> None:
        self._threshold_s = threshold_s
        self._granularity_s = granularity_s
        self._closed_billed_s: float = 0.0
        self._open_start: Optional[float] = None
        self._open_raw_end: float = 0.0
        # max(_open_raw_end, _open_start + threshold_s) — used for merge detection
        self._open_threshold_end: float = 0.0

    def add_interval(self, start: float, end: float) -> None:
        """Record a new closed billing interval.

        Must be called in chronological order: *start* must be >= the end
        of every previously added interval.
        """
        if self._open_start is None:
            self._open_start = start
            self._open_raw_end = end
            self._open_threshold_end = max(end, start + self._threshold_s)
        elif start <= self._open_threshold_end:
            # Merges into the current open group.
            self._open_raw_end = max(self._open_raw_end, end)
            self._open_threshold_end = max(
                self._open_raw_end, self._open_start + self._threshold_s
            )
        else:
            # Closes the open group; starts a new one.
            self._closed_billed_s += self._billed_duration(
                self._open_start, self._open_raw_end
            )
            self._open_start = start
            self._open_raw_end = end
            self._open_threshold_end = max(end, start + self._threshold_s)

    def billed_s(self) -> float:
        """Total billed seconds for all intervals recorded so far."""
        if self._open_start is None:
            return self._closed_billed_s
        return self._closed_billed_s + self._billed_duration(
            self._open_start, self._open_raw_end
        )

    def billed_s_with_window(
        self, window_start: float, window_end: float
    ) -> float:
        """Total billed seconds if a hypothetical additional interval
        [window_start, window_end] is included.

        Does not mutate the accumulator.
        """
        if self._open_start is None:
            return self._billed_duration(window_start, window_end)
        if window_start <= self._open_threshold_end:
            # The window merges with the open group.
            merged_raw_end = max(self._open_raw_end, window_end)
            return self._closed_billed_s + self._billed_duration(
                self._open_start, merged_raw_end
            )
        # The window is a standalone interval.
        return (
            self._closed_billed_s
            + self._billed_duration(self._open_start, self._open_raw_end)
            + self._billed_duration(window_start, window_end)
        )

    def copy(self) -> "BillingAccumulator":
        """Return an independent copy of this accumulator."""
        c = BillingAccumulator(self._threshold_s, self._granularity_s)
        c._closed_billed_s = self._closed_billed_s
        c._open_start = self._open_start
        c._open_raw_end = self._open_raw_end
        c._open_threshold_end = self._open_threshold_end
        return c

    def _billed_duration(self, start: float, raw_end: float) -> float:
        return max(
            self._threshold_s,
            Billing._round_up(raw_end - start, self._granularity_s),
        )
