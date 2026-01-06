from intervaltree import Interval # type: ignore[import]


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
        total_billed_s = 0.0
        if not query_intervals or len(query_intervals) == 0:
            return total_billed_s

        ivs = sorted(query_intervals, key=lambda iv: (iv.begin, iv.end))

        current_start = ivs[0].begin
        current_end = max(ivs[0].end, current_start + threshold_s)

        for iv in ivs[1:]:
            start_time, end_time = iv.begin, iv.end

            if start_time <= current_end:
                current_end = max(current_end, end_time)
            else:
                duration_s = current_end - current_start
                billed_s = max(
                    threshold_s,
                    Billing._round_up(duration_s, granularity_s),
                )
                total_billed_s += billed_s
                current_start = start_time
                current_end = max(end_time, current_start + threshold_s)

        duration_s = current_end - current_start
        billed_s = max(
            threshold_s, Billing._round_up(duration_s, granularity_s)
        )
        total_billed_s += billed_s

        return total_billed_s
