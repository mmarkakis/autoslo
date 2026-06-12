from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from autoslo.filesystem.structured_log import StructuredLog
from autoslo.visualizations.colors import Palette
from autoslo.visualizations.prediction_error_cdf import (
    add_monospace_summary_box,
    build_direction_summary_lines,
    build_percentile_summary_lines,
    plot_grouped_cdf,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize per-RPU prediction errors from a structured log and "
            "save the figure to the run directory."
        )
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID (e.g. 1780875805584).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log = StructuredLog.load(args.run_id)
    results_df = log.prediction_accuracy_df()
    if results_df.empty:
        raise ValueError(
            "No complete query records with predictions were found."
        )

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    plot_grouped_cdf(
        axes[0],
        results_df,
        value_col="factor_error",
        group_col="rpu",
        palette=dict(Palette.rpu_to_color()),
        title="Factor Error by RPU",
        xlabel="Factor error: predicted / actual",
        ylabel="Cumulative probability",
        log_x=True,
        legend_fontsize=8,
    )
    plot_grouped_cdf(
        axes[1],
        results_df,
        value_col="q_error",
        group_col="rpu",
        palette=dict(Palette.rpu_to_color()),
        title="Q-Error by RPU",
        xlabel="Q-error: max(actual / predicted, predicted / actual)",
        ylabel="Cumulative probability",
        log_x=True,
        legend_fontsize=8,
    )

    q_summary = build_percentile_summary_lines(
        results_df,
        group_col="rpu",
        value_col="q_error",
        quantiles=(0.50, 0.90, 0.95),
        group_header="RPU",
    )
    add_monospace_summary_box(axes[1], q_summary, fontsize=9)

    direction_summary = build_direction_summary_lines(
        results_df,
        group_col="rpu",
        actual_col="actual_latency",
        predicted_col="predicted_latency",
        group_header="RPU",
    )
    add_monospace_summary_box(axes[0], direction_summary, fontsize=9)

    fig.suptitle(f"Prediction Errors by RPU — Run {args.run_id}", fontsize=15)

    output_path = (
        Path("data") / "runs" / args.run_id / "prediction_errors_per_rpu.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
