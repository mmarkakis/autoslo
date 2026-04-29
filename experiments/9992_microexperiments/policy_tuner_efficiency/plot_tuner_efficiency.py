import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle

import autoslo.filesystem.path_utils as pu
from autoslo.visualizations.colors import Palette

# Read in the names of the experiments to plot from the config.
exp_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(exp_dir, "timed_tuner_runs.yml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

experiment_names = config["experiment_names"]

all_dfs = []

for experiment_name in experiment_names:
    experiment_path = os.path.join(
        pu.get_data_path(), "tuner_runs", experiment_name
    )
    for scenario_name in os.listdir(experiment_path):
        scenario_path = os.path.join(experiment_path, scenario_name)
        if not os.path.isdir(scenario_path):
            continue

        timing_report = pd.read_csv(
            os.path.join(scenario_path, "timing_report.csv")
        )
        timing_report["experiment"] = experiment_name
        timing_report["scenario"] = scenario_name
        all_dfs.append(timing_report)

summary = pd.concat(all_dfs, ignore_index=True)


# Plot a violin plot per phase key, showing elapsed time.
# Color last column differently.
plt.figure(figsize=(12, 6))
sns.barplot(
    data=summary, y="phase_key", x="elapsed_s", color=Palette.light_green
)
plt.gca().patches[-1].set_color(Palette.dark_green)
plt.xlabel("Elapsed Time (seconds)", fontsize=14)
plotted_pts = summary.groupby("phase_key")["elapsed_s"].mean()
plt.xlim(0.1 * plotted_pts.min(), 10 * plotted_pts.max())
plt.ylabel("")
plt.yticks(fontsize=14)
plt.xscale("log")
plt.xticks(fontsize=14)

# Add an annotation of the mean above each bar
for p in plt.gca().patches:
    assert isinstance(p, Rectangle)  # Make mypy happy
    width = p.get_width()
    plt.gca().annotate(
        f"{width:.2f}",
        (width * 1.5, p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        fontsize=14,
    )

fig_path = os.path.join(exp_dir, "tuner_efficiency.png")
plt.savefig(fig_path, bbox_inches="tight", dpi=300)
