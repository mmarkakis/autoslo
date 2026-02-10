from intervaltree import Interval  # type: ignore[import]


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
        return granularity * ((value + granularity - 1) // granularity)

    @staticmethod
    def billed_s(
        query_intervals: list[Interval],
        threshold_s: float = REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = REDSHIFT_BILLING_GRANULARITY_S,
    ) -> float:
        """
        Calculate the total billed time given the execution intervals of each
        query, considering the billing threshold and granularity.

        Parameters:
            query_intervals: Tthe execution intervals of the queries.
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
        total_billed_s = sum(iv.end - iv.begin for iv in billed_intervals)
        return float(total_billed_s)
    
    @staticmethod
    def _query_id(interval: Interval) -> str:
        if interval.data and "query_id" in interval.data:
            return interval.data["query_id"]
        return "unknown"

    @staticmethod
    def billed_intervals(
        query_intervals: list[Interval],
        threshold_s: float = REDSHIFT_BILLING_THRESHOLD_S,
        granularity_s: float = REDSHIFT_BILLING_GRANULARITY_S,
    ) -> list[Interval]:
        """
        Calculate the billed intervals given the execution intervals of each
        query, considering the billing threshold and granularity.

        Parameters:
            query_intervals: Tthe execution intervals of the queries.
            threshold_s: The billing threshold in seconds. This is the minimum
                time that will be billed - i.e. all smaller intervals are
                rounded up to this threshold.
            granularity_s: The billing granularity in seconds. Each billed
                interval is rounded up to the nearest multiple of this
                granularity.

        Returns:
            The list of billed intervals.
        """

        billed_intervals: list[Interval] = []
        if not query_intervals or len(query_intervals) == 0:
            return billed_intervals

        ivs = sorted(query_intervals, key=lambda iv: (iv.begin, iv.end))

        current_start = ivs[0].begin
        current_end = max(ivs[0].end, current_start + threshold_s)
        current_query_ids = {Billing._query_id(ivs[0])}

        for iv in ivs[1:]:
            start_time, end_time = iv.begin, iv.end

            if start_time <= current_end:
                current_end = max(current_end, end_time)
                current_query_ids.add(Billing._query_id(iv))
            else:
                duration_s = current_end - current_start
                billed_duration_s = max(
                    threshold_s,
                    Billing._round_up(duration_s, granularity_s),
                )
                billed_interval = Interval(
                    begin=current_start,
                    end=current_start + billed_duration_s,
                    data={"query_ids": current_query_ids},
                )
                billed_intervals.append(billed_interval)
                current_start = start_time
                current_end = max(end_time, current_start + threshold_s)
                current_query_ids = {Billing._query_id(iv)}

        duration_s = current_end - current_start
        billed_duration_s = max(
            threshold_s, Billing._round_up(duration_s, granularity_s)
        )
        billed_interval = Interval(
            begin=current_start,
            end=current_start + billed_duration_s,
            data={"query_ids": current_query_ids},
        )
        billed_intervals.append(billed_interval)


        return billed_intervals
