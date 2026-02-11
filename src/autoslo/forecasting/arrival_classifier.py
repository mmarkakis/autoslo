from collections import defaultdict

import rich

from autoslo.forecasting.adhoc_template_detector import AdHocTemplateDetector
from autoslo.forecasting.windowed_template_detector import (
    WindowedTemplateDetector,
)
from autoslo.workload_definition.query import Query


class ArrivalClassifier:

    def __init__(self, queries: list[Query]):
        self._queries = queries
        self._queries.sort(key=lambda x: x.start_time_s)
        self._reference_time = self._queries[0].start_time_s

        self._queries_per_template: dict[int, list[Query]] = defaultdict(list)
        for query in self._queries:
            template_id = Query.template_id(query.tpcds_temp_and_q_idx)
            self._queries_per_template[template_id].append(query)
        self._template_classification: dict[int, str] = {
            template_id: "unclassified"
            for template_id in self._queries_per_template.keys()
        }
        # Store detailed detection results for each template
        self._template_details: dict[int, dict] = {}

    def _determine_windowed_templates(self) -> None:

        distinct_templates = sorted(list(self._queries_per_template.keys()))

        # Pretty print the templates we're analyzing using rich.
        rich.print(f"Analyzing {len(distinct_templates)} distinct templates")

        for template_id in distinct_templates:
            detector = WindowedTemplateDetector(
                queries=self._queries_per_template[template_id]
            )
            result = detector.detect()
            
            # Store detailed results
            self._template_details[template_id] = result

            if result["is_windowed"]:
                self._template_classification[template_id] = "windowed"

    def _determine_adhoc_templates(self) -> None:
        queries_per_undetermined_template = {
            template_id: queries
            for template_id, queries in self._queries_per_template.items()
            if self._template_classification[template_id] == "unclassified"
        }

        detector = AdHocTemplateDetector(queries_per_undetermined_template)
        results = detector.detect()

        for template, result in results.items():
            # Store or update detailed results
            if template in self._template_details:
                self._template_details[template].update(result)
            else:
                self._template_details[template] = result
            
            if result["is_normal"]:
                self._template_classification[template] = "normal"
            else:
                self._template_classification[template] = "ad-hoc"

    def _print_classification_summary(self) -> None:
        classification_counts: dict[str, int] = defaultdict(int)
        for classification in self._template_classification.values():
            classification_counts[classification] += 1

        # Print a nicely formatted table. For each class, it should include
        # - the number of templates in that class
        # - the fraction of total templates in that class
        # - the number of queries in that class
        # - the fraction of total queries in that class
        total_templates = len(self._template_classification)
        total_queries = len(self._queries)
        rich.print(
            f"{'Classification':<15} {'# Templates':<15} {'% Templates':<15} "
            f"{'# Queries':<15} {'% Queries':<15}"
        )
        for classification, count in classification_counts.items():
            templates_in_class = [
                template_id
                for template_id, cls in self._template_classification.items()
                if cls == classification
            ]
            num_templates_in_class = len(templates_in_class)
            fraction_templates_in_class = (
                num_templates_in_class / total_templates
            )
            num_queries_in_class = sum(
                len(self._queries_per_template[template_id])
                for template_id in templates_in_class
            )
            fraction_queries_in_class = num_queries_in_class / total_queries
            rich.print(
                f"{classification:<15} {num_templates_in_class:<15} "
                f"{fraction_templates_in_class:<15.2%} "
                f"{num_queries_in_class:<15} {fraction_queries_in_class:<15.2%}"
            )

    def classify_arrivals(self) -> None:

        # First, determine which templates are windowed and filter out the
        # corresponding arrivals.
        self._determine_windowed_templates()
        self._determine_adhoc_templates()
        self._print_classification_summary()
