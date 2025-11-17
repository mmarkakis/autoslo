import autoslo.utils.colors as cu
from autoslo.strategies_total.ts_replay_past import (
    TSReplayPast1Cost,
    TSReplayPast1Perf,
    TSReplayPast7Cost,
    TSReplayPast7Perf,
    TSReplayPast14Cost,
    TSReplayPast14Perf,
)

# Maps strategy names to their corresponding TotalStrategy classes.
STRATEGIES: dict[str, dict[str, object]] = {
    "training_period": {
        "class": None,  # Placeholder
        "color": cu.Palette.gray,
        "marker": "o",
    },
    "TSReplayPast1Cost": {
        "class": TSReplayPast1Cost,
        "color": cu.Palette.light_yellow,
        "marker": "o",
    },
    "TSReplayPast1Perf": {
        "class": TSReplayPast1Perf,
        "color": cu.Palette.dark_yellow,
        "marker": "^",
    },
    "TSReplayPast7Cost": {
        "class": TSReplayPast7Cost,
        "color": cu.Palette.light_orange,
        "marker": "o",
    },
    "TSReplayPast7Perf": {
        "class": TSReplayPast7Perf,
        "color": cu.Palette.dark_orange,
        "marker": "^",
    },
    "TSReplayPast14Cost": {
        "class": TSReplayPast14Cost,
        "color": cu.Palette.light_red,
        "marker": "o",
    },
    "TSReplayPast14Perf": {
        "class": TSReplayPast14Perf,
        "color": cu.Palette.dark_red,
        "marker": "^",
    },
}
