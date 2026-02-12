from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from autoslo.workload_definition.query import Query


class ArrivalProcess:
    def __init__(self, name: str, template_list: list[str], *args, **kwargs):
        self._name = name
        self.template_list = template_list

    @property
    def num_templates(self) -> int:
        return len(self.template_list)

    @property
    def templates(self) -> list[str]:
        return self.template_list

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        start_time: datetime,
        end_time: datetime,
        seed: int = 42,
        active_window_start: datetime | None = None,
        active_window_end: datetime | None = None,
        *args,
        **kwargs,
    ) -> list[Query]:
        raise NotImplementedError("Subclasses must implement this method")


class PoissonProcess(ArrivalProcess):

    def __init__(
        self,
        rate: float,
        name: str,
        template_list: list[str],
        *args,
        **kwargs,
    ):
        super().__init__(name, template_list)
        self.rate = rate

    def generate(
        self,
        start_time: datetime,
        end_time: datetime,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> list[Query]:
        current_time = start_time
        arrivals = []
        np.random.seed(seed)

        samples_needed = int(
            (end_time - start_time).total_seconds() * self.rate * 1.5
        )

        template_ids = np.random.choice(self.template_list, size=samples_needed)
        inter_arrival_times = np.random.exponential(
            1 / self.rate, size=samples_needed
        )

        current_time_s = start_time.timestamp()
        end_time_s = end_time.timestamp()
        query_id = 0
        for template_id, inter_arrival_time in zip(
            template_ids, inter_arrival_times
        ):
            if current_time_s >= end_time_s:
                break
            arrivals.append(
                Query(
                    query_id=str(query_id),
                    tpcds_temp_and_q_idx=template_id,
                    start_time_s=current_time_s,
                )
            )
            current_time_s += inter_arrival_time
            query_id += 1

        return arrivals


class SinusoidalPoissonProcess(ArrivalProcess):

    def __init__(
        self,
        max_rate: float,
        name: str,
        template_list: list[str],
        *args,
        **kwargs,
    ):
        super().__init__(name, template_list)
        self.max_rate = max_rate

    def generate(
        self,
        start_time: datetime,
        end_time: datetime,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> list[Query]:
        # Split the interval between start_time and end_time into 1-minute segments.
        minute_starts = []
        current = start_time.replace(second=0, microsecond=0)
        end_rounded = end_time.replace(second=0, microsecond=0)
        while current <= end_rounded:
            minute_starts.append(current)
            current += timedelta(minutes=1)
        mid_minute_idx = len(minute_starts) // 2

        samples_needed = int(
            (end_time - start_time).total_seconds() * self.max_rate * 1.5
        )
        template_ids = np.random.choice(self.template_list, size=samples_needed)

        # Have the number of arrivals in each segment vary sinusoidally between 0 and max_rate,
        # with 0 at the start and end of the interval, and max_rate at the midpoint.
        # Within each segment, generate arrivals according to a Poisson process with an appropriate rate.
        arrivals = []
        np.random.seed(seed)
        template_idx = 0
        for i, minute_start in enumerate(minute_starts):
            factor = abs(i - mid_minute_idx) / mid_minute_idx
            segment_rate = np.sin((1 - factor) * (np.pi / 2)) * self.max_rate
            num_arrivals = np.random.poisson(segment_rate * 60)  # per minute
            inter_arrival_times = np.random.exponential(
                1 / segment_rate, size=num_arrivals
            )
            for inter_arrival_time in inter_arrival_times:
                arrival_time = minute_start + timedelta(
                    seconds=inter_arrival_time
                )
                if start_time <= arrival_time < end_time:
                    template_id = template_ids[template_idx]
                    template_idx += 1
                    arrivals.append(
                        Query(
                            query_id=str(template_idx),
                            tpcds_temp_and_q_idx=template_id,
                            start_time_s=arrival_time.timestamp(),
                        )
                    )

        return arrivals


class NoisyPeriodicProcess(ArrivalProcess):

    def __init__(
        self,
        period_s: float,
        window_width_s: float,
        queries_per_window: int,
        name: str,
        template_list: list[str],
        *args,
        **kwargs,
    ):
        super().__init__(name, template_list)
        self.period_s = period_s
        self.window_width_s = window_width_s
        self.queries_per_window = queries_per_window

    def generate(
        self,
        start_time: datetime,
        end_time: datetime,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> list[Query]:
        window_starts = []
        current = start_time
        while current < end_time:
            window_starts.append(current)
            current += timedelta(seconds=self.period_s)

        arrivals = []
        np.random.seed(seed)
        query_id = 0
        for window_start in window_starts:
            eff_window_width_s = min(
                self.window_width_s, (end_time - window_start).total_seconds()
            )
            offsets = np.random.uniform(
                0, eff_window_width_s, size=self.queries_per_window
            )
            templates = np.random.choice(
                self.template_list, size=self.queries_per_window
            )
            for offset, template_id in zip(offsets, templates):
                arrival_time = window_start + timedelta(seconds=offset)
                arrivals.append(
                    Query(
                        query_id=str(query_id),
                        tpcds_temp_and_q_idx=template_id,
                        start_time_s=arrival_time.timestamp(),
                    )
                )
                query_id += 1

        return arrivals
