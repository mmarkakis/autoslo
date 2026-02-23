from datetime import datetime
import numpy as np

from autoslo.workload_definition.query import Query


class AdHocTemplateDetector:

    def __init__(self, queries_per_template: dict[int, list[Query]]):

        self._queries_per_template = queries_per_template

    def detect(self) -> dict[int, dict]:
        """
        We want to classify between normal and ad hoc templates.
        A normal template appears often, on many unique days, and has a stable
        daily/weekly shape.

        An ad hoc template appears rarely, on few unique days, and does not have
        a stable daily/weekly shape.

        """

        # First, for each template we need to determine:
        # - the number of samples, and the fraction of total.
        # - the number of unique days it appears on, and the fraction of total days.
        # - the stability of its daily/weekly shape. We can use a statistical test
        #   for this, e.g. the Kolmogorov-Smirnov test, to compare the distribution of arrival times across different days.
        #   We can also use a clustering approach, e.g. k-means, to cluster the arrival times across different days and see
        #   if they form distinct clusters (indicating a stable shape) or not (indicating an ad hoc shape).

        total_num_queries = sum(
            len(queries) for queries in self._queries_per_template.values()
        )
        total_num_days = len(
            set(
                query.abs_start_time.date()
                for queries in self._queries_per_template.values()
                for query in queries
            )
        )

        template_data: dict[int, dict] = {}

        for template_id, queries in self._queries_per_template.items():
            # Sort the queries by arrival time to make the subsequent analysis easier.
            queries.sort(key=lambda x: x.rel_start_time_s)

            # Look at number of queries.
            num_queries = len(queries)
            fraction_of_total = num_queries / total_num_queries

            # Look at number of unique days.
            unique_days = set(
                query.abs_start_time.date()
                for query in queries
            )
            num_unique_days = len(unique_days)
            fraction_of_total_days = num_unique_days / total_num_days

            # Look at the daily seasonality.
            avg_weekday_correlation, avg_weekend_correlation = (
                self._compute_daily_seasonality(queries)
            )
            has_weekday_seasonality = bool(avg_weekday_correlation > 0.7)
            has_weekend_seasonality = bool(avg_weekend_correlation > 0.7)

            avg_weekly_correlation = self._compute_weekly_seasonality(queries)
            has_weekly_seasonality = bool(avg_weekly_correlation > 0.7)

            is_normal = (
                num_queries >= 10
                and fraction_of_total >= 0.01
                and num_unique_days >= 5
                and fraction_of_total_days >= 0.1
                and (
                    has_weekday_seasonality
                    or has_weekend_seasonality
                    or has_weekly_seasonality
                )
            )

            template_data[template_id] = {
                "is_normal": is_normal,
                "num_queries": num_queries,
                "fraction_of_total": fraction_of_total,
                "num_unique_days": num_unique_days,
                "fraction_of_total_days": fraction_of_total_days,
                "avg_weekday_correlation": avg_weekday_correlation,
                "avg_weekend_correlation": avg_weekend_correlation,
                "avg_weekly_correlation": avg_weekly_correlation,
                "has_weekday_seasonality": has_weekday_seasonality,
                "has_weekend_seasonality": has_weekend_seasonality,
                "has_weekly_seasonality": has_weekly_seasonality,
            }

        # Sort the templates by template ID for consistency.
        template_data = dict(sorted(template_data.items(), key=lambda x: x[0]))

        return template_data

    def _compute_daily_seasonality(self, queries):

        # Look at daily seasonality. We can use the Kolmogorov-Smirnov test
        # to compare the distribution of arrival times across different days,
        # taking weekday/weekend into account.
        weekday_distributions = []
        weekend_distributions = []
        previous_day = queries[0].abs_start_time.date()
        previous_day_is_weekday = queries[0].abs_start_time.weekday() < 5

        daily_counts = np.zeros(24)
        for query in queries:

            current_day = query.abs_start_time.date()

            if current_day != previous_day:
                daily_counts /= np.sum(
                    daily_counts
                )  # Normalize to get a distribution.
                if previous_day_is_weekday:
                    weekday_distributions.append(daily_counts)
                else:
                    weekend_distributions.append(daily_counts)
                daily_counts = np.zeros(24)
                previous_day = current_day
                previous_day_is_weekday = query.abs_start_time.weekday() < 5

            daily_counts[query.abs_start_time.hour] += 1

        # Store the last day's distribution as well.
        daily_counts /= np.sum(daily_counts)  # Normalize to get a distribution.
        if previous_day_is_weekday:
            weekday_distributions.append(daily_counts)
        else:
            weekend_distributions.append(daily_counts)

        # For the weekdays, compute the pairwise correlation between the distributions.
        # If the average correlation is above a certain threshold, we can say it has weekday seasonality.
        weekday_correlations = []
        for i in range(len(weekday_distributions)):
            for j in range(i + 1, len(weekday_distributions)):
                corr = np.corrcoef(
                    weekday_distributions[i], weekday_distributions[j]
                )[0, 1]
                weekday_correlations.append(corr)
        avg_weekday_correlation = (
            np.mean(weekday_correlations)
            if len(weekday_correlations) > 0
            else 0
        )
        has_weekday_seasonality = avg_weekday_correlation > 0.7

        # For the weekends, compute the pairwise correlation between the distributions.
        # If the average correlation is above a certain threshold, we can say it has weekend seasonality.
        weekend_correlations = []
        for i in range(len(weekend_distributions)):
            for j in range(i + 1, len(weekend_distributions)):
                corr = np.corrcoef(
                    weekend_distributions[i], weekend_distributions[j]
                )[0, 1]
                weekend_correlations.append(corr)
        avg_weekend_correlation = (
            np.mean(weekend_correlations)
            if len(weekend_correlations) > 0
            else 0
        )
        has_weekend_seasonality = avg_weekend_correlation > 0.7

        return avg_weekday_correlation, avg_weekend_correlation

    def _compute_weekly_seasonality(self, queries):

        week_distributions = []
        previous_week = queries[0].abs_start_time.isocalendar()[1]

        weekly_counts = np.zeros(7)
        for query in queries:

            current_week = query.abs_start_time.isocalendar()[1]

            if current_week != previous_week:
                weekly_counts /= np.sum(
                    weekly_counts
                )  # Normalize to get a distribution.
                week_distributions.append(weekly_counts)
                weekly_counts = np.zeros(7)
                previous_week = current_week

            weekly_counts[query.abs_start_time.weekday()] += 1

        # Store the last week's distribution as well.
        weekly_counts /= np.sum(
            weekly_counts
        )  # Normalize to get a distribution.
        week_distributions.append(weekly_counts)

        # Compute the pairwise correlation between the distributions.
        # If the average correlation is above a certain threshold, we can say it has weekly seasonality.
        week_correlations = []
        for i in range(len(week_distributions)):
            for j in range(i + 1, len(week_distributions)):
                corr = np.corrcoef(
                    week_distributions[i], week_distributions[j]
                )[0, 1]
                week_correlations.append(corr)
        avg_week_correlation = (
            np.mean(week_correlations) if len(week_correlations) > 0 else 0
        )
        has_weekly_seasonality = avg_week_correlation > 0.7

        return avg_week_correlation
