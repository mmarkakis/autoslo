from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass
class QueryArrival:
    template_id: str
    arrival_time: datetime


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
        *args,
        **kwargs,
    ) -> list[QueryArrival]:
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
    ) -> list[QueryArrival]:
        current_time = start_time
        arrivals = []
        np.random.seed(seed)
        while current_time < end_time:
            inter_arrival_time = np.random.exponential(1 / self.rate)
            current_time += timedelta(seconds=inter_arrival_time)

            if current_time < end_time:
                template_id = str(np.random.choice(self.template_list))
                arrivals.append(
                    QueryArrival(
                        template_id=template_id, arrival_time=current_time
                    )
                )
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
    ) -> list[QueryArrival]:
        # Split the interval between start_time and end_time into 1-minute segments.
        minute_starts = []
        current = start_time.replace(second=0, microsecond=0)
        end_rounded = end_time.replace(second=0, microsecond=0)
        while current <= end_rounded:
            minute_starts.append(current)
            current += timedelta(minutes=1)
        mid_minute_idx = len(minute_starts) // 2

        # Have the number of arrivals in each segment vary sinusoidally between 0 and max_rate,
        # with 0 at the start and end of the interval, and max_rate at the midpoint.
        # Within each segment, generate arrivals according to a Poisson process with an appropriate rate.
        arrivals = []
        np.random.seed(seed)
        for i, minute_start in enumerate(minute_starts):
            factor = abs(i - mid_minute_idx) / mid_minute_idx
            segment_rate = np.sin((1 - factor) * (np.pi / 2)) * self.max_rate
            num_arrivals = np.random.poisson(segment_rate * 60)  # per minute
            for _ in range(num_arrivals):
                inter_arrival_time = np.random.exponential(1 / segment_rate)
                arrival_time = minute_start + timedelta(
                    seconds=inter_arrival_time
                )
                if start_time <= arrival_time < end_time:
                    template_id = str(np.random.choice(self.template_list))
                    arrivals.append(
                        QueryArrival(
                            template_id=template_id, arrival_time=arrival_time
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
    ) -> list[QueryArrival]:
        window_starts = []
        current = start_time
        while current < end_time:
            window_starts.append(current)
            current += timedelta(seconds=self.period_s)

        arrivals = []
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
                    QueryArrival(
                        template_id=template_id, arrival_time=arrival_time
                    )
                )

        return arrivals
