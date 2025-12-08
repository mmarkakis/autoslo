import autoslo.utils.colors as cu
from autoslo.strategies.strat_histuniform_fullperf_single import (
    StratHistUniformFullPerfSingle1,
    StratHistUniformFullPerfSingle7,
    StratHistUniformFullPerfSingle14,
)
from autoslo.strategies.strat_histuniform_obsperf_single import (
    StratHistUniformObsPerfSingle1,
    StratHistUniformObsPerfSingle7,
    StratHistUniformObsPerfSingle14,
)
from autoslo.strategies.strat_exactfuture_predperf_single import (
    StratExactFuturePredPerfSingle,
)
from autoslo.strategies.strat_exactfuture_fullperf_single import (
    StratExactFutureFullPerfSingle,
)

# Maps strategy names to their corresponding TotalStrategy classes.
STRATEGIES: dict[str, dict[str, object]] = {
    "training_period": {
        "class": None,  # Placeholder
        "color": cu.Palette.gray,
        "marker": "o",
    },
    "StratHistUniformFullPerfSingle1": {
        "class": StratHistUniformFullPerfSingle1,
        "color": cu.Palette.light_yellow,
        "marker": "o",
    },
    "StratHistUniformFullPerfSingle7": {
        "class": StratHistUniformFullPerfSingle7,
        "color": cu.Palette.light_orange,
        "marker": "o",
    },
    "StratHistUniformFullPerfSingle14": {
        "class": StratHistUniformFullPerfSingle14,
        "color": cu.Palette.light_red,
        "marker": "o",
    },
    "StratHistUniformObsPerfSingle1": {
        "class": StratHistUniformObsPerfSingle1,
        "color": cu.Palette.dark_yellow,
        "marker": "^",
    },
    "StratHistUniformObsPerfSingle7": {
        "class": StratHistUniformObsPerfSingle7,
        "color": cu.Palette.dark_orange,
        "marker": "^",
    },
    "StratHistUniformObsPerfSingle14": {
        "class": StratHistUniformObsPerfSingle14,
        "color": cu.Palette.dark_red,
        "marker": "^",
    },
    "StratExactFutureFullPerfSingle": {
        "class": StratExactFutureFullPerfSingle,
        "color": cu.Palette.light_blue,
        "marker": "o",
    },
    "StratExactFuturePredPerfSingle": {
        "class": StratExactFuturePredPerfSingle,
        "color": cu.Palette.light_green,
        "marker": "^",
    },
}
