from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt
from cycler import cycler


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
    light_gray: str = "#D3D3D3"
    light_purple: str = "#D8BFD8"
    dark_purple: str = "#800080"
    gray: str = "#4A4A4A"
    black: str = "#000000"

    @staticmethod
    def semantic_colors() -> Mapping[str, str]:
        return {
            "success": Palette.dark_green,
            "info": Palette.dark_blue,
            "warning": Palette.dark_yellow,
            "error": Palette.dark_red,
        }

    # Expose as a colormap for seaborn/matplotlib
    @staticmethod
    def as_colormap() -> Mapping[str, str]:
        return {
            "light_green": Palette.light_green,
            "dark_green": Palette.dark_green,
            "light_blue": Palette.light_blue,
            "dark_blue": Palette.dark_blue,
            "light_yellow": Palette.light_yellow,
            "dark_yellow": Palette.dark_yellow,
            "light_orange": Palette.light_orange,
            "dark_orange": Palette.dark_orange,
            "light_red": Palette.light_red,
            "dark_red": Palette.dark_red,
            "light_gray": Palette.light_gray,
            "gray": Palette.gray,
            "black": Palette.black,
            "light_purple": Palette.light_purple,
            "dark_purple": Palette.dark_purple,
        }
    
    @staticmethod
    def as_list() -> list[str]:
        return [
            Palette.dark_blue,
            Palette.dark_orange,
            Palette.dark_green,
            Palette.dark_red,
            Palette.light_blue,
            Palette.light_orange,
            Palette.light_green,
            Palette.light_red,
            Palette.gray,
        ]


def set_plot_colors(dark: bool = False) -> None:
    """
    Set matplotlib plot colors based on the provided palette.

    Parameters:
        dark: If True, use darker colors; otherwise, use light colors.

    """

    if dark:
        cycle = [
            Palette.dark_blue,
            Palette.dark_orange,
            Palette.dark_green,
            Palette.dark_red,
            Palette.gray,
        ]
    else:
        cycle = [
            Palette.light_blue,
            Palette.light_orange,
            Palette.light_green,
            Palette.light_red,
            Palette.gray,
        ]

    plt.rcParams.update(
        {
            "figure.facecolor": Palette.white,
            "axes.facecolor": Palette.light_green,
            "text.color": Palette.black,
            "axes.labelcolor": Palette.black,
            "xtick.color": Palette.black,
            "ytick.color": Palette.black,
            "grid.color": Palette.gray,
            "axes.prop_cycle": cycler("color", cycle),
        }
    )
