from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt
from cycler import cycler


@dataclass(frozen=True)
class Palette:
    white: str = "#FFFFFF"
    light_green: str = "#8FD694"
    dark_green: str = "#3A9D5D"
    light_blue: str = "#8DB8FF"
    dark_blue: str = "#3466FF"
    light_yellow: str = "#F7E17D"
    dark_yellow: str = "#E3C029"
    light_orange: str = "#F4B383"
    dark_orange: str = "#E07022"
    light_red: str = "#E78C84"
    dark_red: str = "#C9302C"
    gray: str = "#4A4A4A"
    black: str = "#000000"

    def semantic_colors(self) -> Mapping[str, str]:
        return {
            "success": self.dark_green,
            "info": self.dark_blue,
            "warning": self.dark_yellow,
            "error": self.dark_red,
        }


def set_plot_colors(dark: bool = False) -> None:
    """
    Set matplotlib plot colors based on the provided palette.

    Parameters:
        dark: If True, use darker colors; otherwise, use light colors.
    
    """

    PALETTE = Palette()

    if dark:
        cycle = [
            PALETTE.dark_blue,
            PALETTE.dark_orange,
            PALETTE.dark_green,
            PALETTE.dark_red,
            PALETTE.gray,
        ]
    else:
        cycle = [
            PALETTE.light_blue,
            PALETTE.light_orange,
            PALETTE.light_green,
            PALETTE.light_red,
            PALETTE.gray,
        ]

    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE.white,
            "axes.facecolor": PALETTE.light_green,
            "text.color": PALETTE.black,
            "axes.labelcolor": PALETTE.black,
            "xtick.color": PALETTE.black,
            "ytick.color": PALETTE.black,
            "grid.color": PALETTE.gray,
            "axes.prop_cycle": cycler("color", cycle),
        }
    )
