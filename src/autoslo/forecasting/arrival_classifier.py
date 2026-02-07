from dataclasses import dataclass
from datetime import datetime
import numpy as np
from autoslo.forecasting.windowed_template_detector import (
    WindowedTemplateDetector,
)
import rich


@dataclass
class QueryArrival:
    template_id: str
    arrival_time: datetime


class ArrivalClassifier:

    def __init__(self, arrivals: list[QueryArrival]):
        self._arrivals = arrivals
        self._arrivals.sort(key=lambda x: x.arrival_time)
        self._reference_time = self._arrivals[0].arrival_time

        self._windowed_templates_info: dict[str, dict] = {}
        self._windowed_arrivals: list[QueryArrival] = []

    def _determine_windowed_templates(self) -> None:

        distinct_templates = sorted(
            list(set(arrival.template_id for arrival in self._arrivals))
        )

        # Pretty print the templates we're analyzing using rich.
        rich.print(f"Analyzing {len(distinct_templates)} distinct templates")

        for template_id in distinct_templates:
            template_timestamps = np.array(
                [
                    (
                        arrival.arrival_time - self._reference_time
                    ).total_seconds()
                    for arrival in self._arrivals
                    if arrival.template_id == template_id
                ]
            )

            start_time = datetime.now()
            detector = WindowedTemplateDetector(template_timestamps)
            result = detector.detect()
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            print(
                f"Template {template_id}: "
                f"Elapsed Time: {elapsed:.2f} s, "
                f"Result: {result}"
            )

            if result["is_windowed"]:
                self._windowed_templates_info[template_id] = result

    def _filter_out_windowed_arrivals(self) -> None:
        remaining_arrivals: list[QueryArrival] = []
        for arrival in self._arrivals:
            template_id = arrival.template_id
            if template_id in self._windowed_templates_info:
                self._windowed_arrivals.append(arrival)
            else:
                remaining_arrivals.append(arrival)
        self._arrivals = remaining_arrivals

    def classify_arrivals(self) -> None:

        # First, determine which templates are windowed and filter out the
        # corresponding arrivals.
        self._determine_windowed_templates()
        self._filter_out_windowed_arrivals()

        
