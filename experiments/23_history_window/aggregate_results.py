"""Aggregate results from the history-window experiment.

Reads holdout and final summaries from each scenario, produces a
comparison table, CSV, and bar chart.

Usage
-----
::

    python experiments/23_history_window/aggregate_results.py \\
        --run-dir data/tuner_runs/history_exp
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

SCENARIOS = ["prev_day", "prev_week", "prev_month"]
LABELS = {
    "prev_day": "1 day",
    "prev_week": "1 week",
    "prev_month": "1 month",
}


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate history-window experiment results.",
    )
    parser.add_argument(
        "--run-dir",
        default="data/tuner_runs/history_exp",
        help="Root directory containing prev_day/, prev_week/, prev_month/.",
    )
    args = parser.parse_args()
    run_root = Path(args.run_dir)
    console = Console()

    # ---- Collect data ---------------------------------------------------
    rows: list[dict] = []
    for scenario in SCENARIOS:
        scenario_dir = run_root / scenario
        holdout = _load_yaml(scenario_dir / "holdout" / "summary.yml")
        final = _load_yaml(scenario_dir / "final" / "summary.yml")
        reservoir_meta = _load_yaml(scenario_dir / "reservoir" / "reservoir_meta.yml")

        row = {
            "scenario": scenario,
            "label": LABELS[scenario],
            "reservoir_arrivals": (
                reservoir_meta.get("num_arrivals") if reservoir_meta else None
            ),
        }

        if holdout:
            row["holdout_baseline_violation"] = holdout.get("baseline_violation")
            row["holdout_baseline_cost"] = holdout.get("baseline_cost")
            row["holdout_tuned_violation"] = holdout.get("tuned_violation")
            row["holdout_tuned_cost"] = holdout.get("tuned_cost")
            row["holdout_queries"] = holdout.get("num_holdout_queries")
        if final:
            row["final_train_violation"] = final.get("train_violation_agg")
            row["final_train_cost"] = final.get("train_cost_agg")
            row["final_val_violation"] = final.get("val_violation_agg")
            row["final_val_cost"] = final.get("val_cost_agg")

        rows.append(row)

    # ---- Rich table -----------------------------------------------------
    table = Table(
        title="History-Window Experiment — Holdout Results",
        show_lines=True,
    )
    table.add_column("History", justify="left")
    table.add_column("Reservoir", justify="right")
    table.add_column("Holdout\nBaseline Viol.", justify="right")
    table.add_column("Holdout\nTuned Viol.", justify="right")
    table.add_column("Δ Viol.", justify="right")
    table.add_column("Holdout\nBaseline Cost", justify="right")
    table.add_column("Holdout\nTuned Cost", justify="right")
    table.add_column("Δ Cost", justify="right")

    for r in rows:
        bv = r.get("holdout_baseline_violation")
        tv = r.get("holdout_tuned_violation")
        bc = r.get("holdout_baseline_cost")
        tc = r.get("holdout_tuned_cost")

        dv_str = ""
        if bv is not None and tv is not None:
            dv = tv - bv
            sign = "+" if dv >= 0 else ""
            style = "green" if dv <= 0 else "red"
            dv_str = f"[{style}]{sign}{dv:.4f}[/{style}]"

        dc_str = ""
        if bc is not None and tc is not None:
            dc = tc - bc
            sign = "+" if dc >= 0 else ""
            style = "green" if dc <= 0 else "red"
            dc_str = f"[{style}]{sign}{dc:.2f}[/{style}]"

        table.add_row(
            r["label"],
            f"{r.get('reservoir_arrivals', '?'):,}" if r.get("reservoir_arrivals") else "?",
            f"{bv:.4f}" if bv is not None else "—",
            f"{tv:.4f}" if tv is not None else "—",
            dv_str or "—",
            f"{bc:.2f}" if bc is not None else "—",
            f"{tc:.2f}" if tc is not None else "—",
            dc_str or "—",
        )

    console.print(table)

    # ---- CSV ------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "comparison.csv"
    fieldnames = [
        "scenario", "label", "reservoir_arrivals", "holdout_queries",
        "holdout_baseline_violation", "holdout_tuned_violation",
        "holdout_baseline_cost", "holdout_tuned_cost",
        "final_train_violation", "final_val_violation",
        "final_train_cost", "final_val_cost",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\nCSV written to: {csv_path}")

    # ---- Bar chart (optional, needs matplotlib) -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping plot.[/yellow]")
        return

    labels = [r["label"] for r in rows]
    has_holdout = all(
        r.get("holdout_tuned_violation") is not None for r in rows
    )
    if not has_holdout:
        console.print("[yellow]Holdout data missing for some scenarios; "
                       "skipping plot.[/yellow]")
        return

    baseline_viol = [r["holdout_baseline_violation"] for r in rows]
    tuned_viol = [r["holdout_tuned_violation"] for r in rows]
    baseline_cost = [r["holdout_baseline_cost"] for r in rows]
    tuned_cost = [r["holdout_tuned_cost"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(labels))
    width = 0.35

    ax1.bar([i - width / 2 for i in x], baseline_viol, width, label="Baseline")
    ax1.bar([i + width / 2 for i in x], tuned_viol, width, label="Tuned")
    ax1.set_ylabel("Violation Rate")
    ax1.set_title("Holdout Violation Rate")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.legend()

    ax2.bar([i - width / 2 for i in x], baseline_cost, width, label="Baseline")
    ax2.bar([i + width / 2 for i in x], tuned_cost, width, label="Tuned")
    ax2.set_ylabel("Cost ($)")
    ax2.set_title("Holdout Cost")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.legend()

    fig.suptitle("History-Window Experiment: Holdout Performance", fontsize=14)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "holdout_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    console.print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
