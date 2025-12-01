import autoslo.utils.colors as cu
from autoslo.strategies.strat_historical_single import (
    StratHistoricalSingle1,
    StratHistoricalSingle7,
    StratHistoricalSingle14,
)
from autoslo.strategies.strat_model_noforecast_single import (
    StratModelNoForecastSingle,
)
from autoslo.strategies.strat_oracle_single import StratOracleSingle

# Maps strategy names to their corresponding TotalStrategy classes.
STRATEGIES: dict[str, dict[str, object]] = {
    "training_period": {
        "class": None,  # Placeholder
        "color": cu.Palette.gray,
        "marker": "o",
    },
    "StratHistoricalSingle1": {
        "class": StratHistoricalSingle1,
        "color": cu.Palette.light_yellow,
        "marker": "o",
    },
    "StratHistoricalSingle7": {
        "class": StratHistoricalSingle7,
        "color": cu.Palette.light_orange,
        "marker": "o",
    },
    "StratHistoricalSingle14": {
        "class": StratHistoricalSingle14,
        "color": cu.Palette.light_red,
        "marker": "o",
    },
    "StratOracleSingle": {
        "class": StratOracleSingle,
        "color": cu.Palette.light_blue,
        "marker": "s",
    },
    "StratModelNoForecastSingle": {
        "class": StratModelNoForecastSingle,
        "color": cu.Palette.light_green,
        "marker": "^",
    },
}
