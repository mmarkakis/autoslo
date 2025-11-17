from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_definition.day import Day
from autoslo.workload_definition.composite import Composite
import os
import numpy as np


def save_def_and_traces(workload: Composite):
    """Helper function to save composite workload definition and traces."""
    workload.save()


weekend_day = Day([Chunk(H=50, T=120)])

# weekly_set
weekday = Day([Chunk(H=10, T=60)])
weekly_set = Composite("weekly_set", ([weekday] * 5 + [weekend_day] * 2) * 4)
save_def_and_traces(weekly_set)

# weekly_peak
monday = Day([Chunk(H=10, T=60)])
tuesday = Day([Chunk(H=10, T=30)])
wednesday = Day([Chunk(H=10, T=10)])
thursday = tuesday
friday = monday

weekly_peak = Composite(
    "weekly_peak",
    ([monday, tuesday, wednesday, thursday, friday] + [weekend_day] * 2) * 4,
)
save_def_and_traces(weekly_peak)

# weekly_random
light_weekday = Day([Chunk(H=10, T=60)])
medium_weekday = Day([Chunk(H=10, T=30)])
heavy_weekday = Day([Chunk(H=10, T=10)])

np.random.seed(42)
weekdays = np.random.choice(
    [light_weekday, medium_weekday, heavy_weekday], size=20
)
days = []
for i in range(len(weekdays) // 5):
    days.extend(weekdays[i * 5 : (i + 1) * 5])
    days.extend([weekend_day] * 2)
weekly_random = Composite("weekly_random", days)
save_def_and_traces(weekly_random)


##############################

# Growth T base
first_week = [Day([Chunk(H=10, T=120)])] * 5 + [weekend_day] * 2
second_week = [Day([Chunk(H=10, T=60)])] * 5 + [weekend_day] * 2
third_week = [Day([Chunk(H=10, T=30)])] * 5 + [weekend_day] * 2
fourth_week = [Day([Chunk(H=10, T=10)])] * 5 + [weekend_day] * 2
growth_t_base = Composite(
    "growth_t_base", first_week + second_week + third_week + fourth_week
)
save_def_and_traces(growth_t_base)


# Growth T added
first_week = [Day([Chunk(H=10, T=120)])] * 5 + [weekend_day] * 2
second_week = [Day([Chunk(H=10, T=120), Chunk(H=10, T=60)])] * 5 + [
    weekend_day
] * 2
third_week = [
    Day(
        [
            Chunk(H=10, T=120),
            Chunk(H=10, T=60),
            Chunk(H=10, T=30),
        ]
    )
] * 5 + [weekend_day] * 2
fourth_week = [
    Day(
        [
            Chunk(H=10, T=120),
            Chunk(H=10, T=60),
            Chunk(H=10, T=30),
            Chunk(H=10, T=10),
        ]
    )
] * 5 + [weekend_day] * 2
growth_t_added = Composite(
    "growth_t_added", first_week + second_week + third_week + fourth_week
)
save_def_and_traces(growth_t_added)


# Growth T noisy
np.random.seed(12)
first_week = [Day([Chunk(10, 120)])] * 5 + [weekend_day] * 2
second_week_t = [60] * 5 + [120] * 5
np.random.shuffle(second_week_t)
second_week = [
    Day(
        [
            Chunk(10, second_week_t[2 * i]),
            Chunk(10, second_week_t[2 * i + 1]),
        ]
    )
    for i in range(5)
] + [weekend_day] * 2
third_week_t = [30] * 5 + [60] * 5 + [120] * 5
np.random.shuffle(third_week_t)
third_week = [
    Day(
        [
            Chunk(10, third_week_t[3 * i]),
            Chunk(10, third_week_t[3 * i + 1]),
            Chunk(10, third_week_t[3 * i + 2]),
        ]
    )
    for i in range(5)
] + [weekend_day] * 2
fourth_week_t = [10] * 5 + [30] * 5 + [60] * 5 + [120] * 5
np.random.shuffle(fourth_week_t)
fourth_week = [
    Day(
        [
            Chunk(10, fourth_week_t[4 * i]),
            Chunk(10, fourth_week_t[4 * i + 1]),
            Chunk(10, fourth_week_t[4 * i + 2]),
            Chunk(10, fourth_week_t[4 * i + 3]),
        ]
    )
    for i in range(5)
] + [weekend_day] * 2
growth_t_noisy = Composite(
    "growth_t_noisy", first_week + second_week + third_week + fourth_week
)
save_def_and_traces(growth_t_noisy)

##############################

# Growth H base
first_week = [Day([Chunk(0, 30)])] * 5 + [weekend_day] * 2
second_week = [Day([Chunk(10, 30)])] * 5 + [weekend_day] * 2
third_week = [Day([Chunk(25, 30)])] * 5 + [weekend_day] * 2
fourth_week = [Day([Chunk(50, 30)])] * 5 + [weekend_day] * 2
growth_h_base = Composite(
    "growth_h_base", first_week + second_week + third_week + fourth_week
)
save_def_and_traces(growth_h_base)


# Growth Η added
first_week = [Day([Chunk(0, 30)])] * 5 + [weekend_day] * 2
second_week = [Day([Chunk(0, 30), Chunk(10, 30)])] * 5 + [weekend_day] * 2
third_week = [Day([Chunk(0, 30), Chunk(10, 30), Chunk(25, 30)])] * 5 + [
    weekend_day
] * 2
fourth_week = [
    Day([Chunk(0, 30), Chunk(10, 30), Chunk(25, 30), Chunk(50, 30)])
] * 5 + [weekend_day] * 2
growth_h_added = Composite(
    "growth_h_added", first_week + second_week + third_week + fourth_week
)
save_def_and_traces(growth_h_added)


# Growth H noisy
np.random.seed(12)
first_week = [Day([Chunk(0, 30)])] * 5 + [weekend_day] * 2
second_week_h = [0] * 5 + [10] * 5
np.random.shuffle(second_week_h)
second_week = [
    Day(
        [
            Chunk(second_week_h[2 * i], 30),
            Chunk(second_week_h[2 * i + 1], 30),
        ]
    )
    for i in range(5)
] + [weekend_day] * 2
third_week_h = [0] * 5 + [10] * 5 + [25] * 5
np.random.shuffle(third_week_h)
third_week = [
    Day(
        [
            Chunk(third_week_h[3 * i], 30),
            Chunk(third_week_h[3 * i + 1], 30),
            Chunk(third_week_h[3 * i + 2], 30),
        ]
    )
    for i in range(5)
] + [weekend_day] * 2
fourth_week_h = [0] * 5 + [10] * 5 + [25] * 5 + [50] * 5
np.random.shuffle(fourth_week_h)
fourth_week = [
    Day(
        [
            Chunk(fourth_week_h[4 * i], 30),
            Chunk(fourth_week_h[4 * i + 1], 30),
            Chunk(fourth_week_h[4 * i + 2], 30),
            Chunk(fourth_week_h[4 * i + 3], 30),
        ]
    )
    for i in range(5)
] + [weekend_day] * 2
growth_h_noisy = Composite(
    "growth_h_noisy", first_week + second_week + third_week + fourth_week
)
save_def_and_traces(growth_h_noisy)
