"""Aggregate results from the history-window experiment.

Reads holdout summaries, final (train/val) summaries, and config diffs
from each scenario produced by :class:`PolicyTuner`, then generates
Rich comparison tables, CSVs, and scatter plots.

Usage
-----
::

    python experiments/23_history_window/aggregate_results.py \\
        --run-dir data/tuner_runs/history_window_exp
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

# Scenario names map to run_ids like "tuner_prev_day" under the run root.
SCENARIOS = ["prev_day", "prev_week", "prev_month"]
LABELS = {
    "prev_day": "1 day",
    "prev_week": "1 week",
    "prev_month": "1 month",
}

# Three violation metrics emitted by the tuner.
_VIOLATION_SUFFIXES = [
    "violation_rate",
    "violation_amount_s",
    "violation_relative_mean",
]

_CHECKPOINT_KEY = "autoscaling_config.capacity_checkpoints"


# ---------------------------------------------------------------------------
# YAML / config helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict to dot-path keys."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items


def _load_config_pair(
    scenario_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load initial and final configs, return as flat dicts."""
    init_cfg = _load_yaml(scenario_dir / "initial_config.yml")
    final_cfg = _load_yaml(scenario_dir / "final_config.yml")
    if init_cfg is None or final_cfg is None:
        return None
    return _flatten_dict(init_cfg), _flatten_dict(final_cfg)


def _diff_configs(
    base_flat: dict[str, Any], tuned_flat: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Return (key, base_val, tuned_val) for keys that differ.

    Excludes ``capacity_checkpoints`` (shown in its own table).
    """
    diffs: list[tuple[str, Any, Any]] = []
    all_keys = sorted(set(base_flat) | set(tuned_flat))
    for k in all_keys:
        if k == _CHECKPOINT_KEY:
            continue
        bv = base_flat.get(k)
        tv = tuned_flat.get(k)
        if bv != tv:
            diffs.append((k, bv, tv))
    return diffs


def _extract_checkpoints(tuned_flat: dict[str, Any]) -> list[dict]:
    """Return the capacity_checkpoints list from the flat config."""
    raw = tuned_flat.get(_CHECKPOINT_KEY)
    if isinstance(raw, list):
        return raw
    return []


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------

_METRIC_COLUMNS = [
    ("Viol. Rate", "violation_rate", ".4f"),
    ("Viol. Amt (s)", "violation_amount_s", ".4f"),
    ("Viol. Rel.", "violation_relative_mean", ".4f"),
    ("Cost ($)", "cost", ".2f"),
]


def _fmt(val: Any, spec: str) -> str:
    """Format a numeric value or return '—'."""
    if val is None:
        return "—"
    return f"{val:{spec}}"


def _delta_cell(
    before: float | None, after: float | None, fmt_spec: str = ".2f"
) -> str:
    """Return a Rich-styled delta string (green if improved, red if worse)."""
    if before is None or after is None:
        return "—"
    delta = after - before
    sign = "+" if delta >= 0 else ""
    style = "green" if delta <= 0 else "red"
    return f"[{style}]{sign}{delta:{fmt_spec}}[/{style}]"


def _build_comparison_table(
    title: str,
    rows: list[dict],
    label_key: str,
    initial_prefix: str,
    final_prefix: str,
) -> Table:
    """Build a Rich table comparing initial vs final for each scenario.

    Each row has: label, then for each of the 4 metrics (3 violations + cost):
    initial value, final value, delta.
    """
    table = Table(title=title, show_lines=True)
    table.add_column("History", justify="left")
    for col_label, _, _ in _METRIC_COLUMNS:
        table.add_column(f"Initial\n{col_label}", justify="right")
        table.add_column(f"Final\n{col_label}", justify="right")
        table.add_column(f"Δ {col_label}", justify="right")

    for r in rows:
        cells: list[str] = [r[label_key]]
        for _, suffix, spec in _METRIC_COLUMNS:
            iv = r.get(f"{initial_prefix}_{suffix}")
            fv = r.get(f"{final_prefix}_{suffix}")
            cells.append(_fmt(iv, spec))
            cells.append(_fmt(fv, spec))
            cells.append(_delta_cell(iv, fv, spec))
        table.add_row(*cells)

    return table


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _collect_scenario_data(
    scenario_dir: Path, scenario: str
) -> dict[str, Any]:
    """Collect all metrics for a single scenario into a flat dict."""
    row: dict[str, Any] = {
        "scenario": scenario,
        "label": LABELS[scenario],
    }

    # --- Holdout data (Phase 8: target-period evaluation) -----------------
    holdout = _load_yaml(scenario_dir / "holdout" / "summary.yml")
    if holdout:
        row["slo_metric"] = holdout.get("slo_metric")
        for prefix in ("initial", "final"):
            for suffix in _VIOLATION_SUFFIXES:
                row[f"holdout_{prefix}_{suffix}"] = holdout.get(
                    f"{prefix}_{suffix}"
                )
            row[f"holdout_{prefix}_cost"] = holdout.get(f"{prefix}_cost")
        if holdout.get("static_baselines"):
            row["static_baselines"] = holdout["static_baselines"]

    # --- Final train/val summaries (Phase 7) ------------------------------
    for split in ("train", "val"):
        summary = _load_yaml(
            scenario_dir / "final" / f"{split}_summary.yml"
        )
        if summary:
            for suffix in _VIOLATION_SUFFIXES:
                row[f"final_{split}_{suffix}"] = summary.get(suffix)
            row[f"final_{split}_cost"] = summary.get("cost")

    # --- Baseline train/val summaries (Phase 3) ---------------------------
    for split in ("train", "val"):
        summary = _load_yaml(
            scenario_dir / "baseline" / f"{split}_summary.yml"
        )
        if summary:
            for suffix in _VIOLATION_SUFFIXES:
                row[f"baseline_{split}_{suffix}"] = summary.get(suffix)
            row[f"baseline_{split}_cost"] = summary.get("cost")

    # --- Config diff info -------------------------------------------------
    pair = _load_config_pair(scenario_dir)
    if pair:
        base_flat, tuned_flat = pair
        cps = _extract_checkpoints(tuned_flat)
        row["num_checkpoints"] = len(cps)
        row["checkpoints"] = cps
        row["config_diffs"] = _diff_configs(base_flat, tuned_flat)
    else:
        row["num_checkpoints"] = None
        row["checkpoints"] = []
        row["config_diffs"] = []

    return row


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _print_holdout_table(console: Console, rows: list[dict]) -> None:
    """Print the holdout (target-day) comparison table."""
    holdout_rows = [
        {
            "label": r["label"],
            **{
                f"initial_{s}": r.get(f"holdout_initial_{s}")
                for s in [*_VIOLATION_SUFFIXES, "cost"]
            },
            **{
                f"final_{s}": r.get(f"holdout_final_{s}")
                for s in [*_VIOLATION_SUFFIXES, "cost"]
            },
        }
        for r in rows
    ]
    table = _build_comparison_table(
        title="Holdout Results (Target-Day Evaluation)",
        rows=holdout_rows,
        label_key="label",
        initial_prefix="initial",
        final_prefix="final",
    )
    console.print(table)


def _print_train_val_table(
    console: Console, rows: list[dict], phase: str, title: str
) -> None:
    """Print a train/val comparison table for a given phase.

    *phase* is e.g. ``"baseline"`` or ``"final"``.
    """
    for split in ("train", "val"):
        split_rows = [
            {
                "label": r["label"],
                **{
                    f"initial_{s}": r.get(f"baseline_{split}_{s}")
                    for s in [*_VIOLATION_SUFFIXES, "cost"]
                },
                **{
                    f"final_{s}": r.get(f"{phase}_{split}_{s}")
                    for s in [*_VIOLATION_SUFFIXES, "cost"]
                },
            }
            for r in rows
        ]
        table = _build_comparison_table(
            title=f"{title} — {split.title()} Set",
            rows=split_rows,
            label_key="label",
            initial_prefix="initial",
            final_prefix="final",
        )
        console.print()
        console.print(table)


def _print_checkpoint_table(console: Console, rows: list[dict]) -> None:
    """Print the capacity checkpoints table."""
    if not any(r.get("checkpoints") for r in rows):
        return
    table = Table(
        title="Capacity Checkpoints Added by Tuner",
        show_lines=True,
    )
    table.add_column("History", justify="left")
    table.add_column("# Checkpts", justify="right")
    table.add_column("Times (s)", justify="left")
    table.add_column("RPU Sizes", justify="left")
    for r in rows:
        cps = r.get("checkpoints", [])
        if not cps:
            table.add_row(r["label"], "0", "—", "—")
        else:
            times = ", ".join(f"{cp.get('time_s', '?'):.0f}" for cp in cps)
            rpus = ", ".join(str(cp.get("min_rpus", "?")) for cp in cps)
            table.add_row(r["label"], str(len(cps)), times, rpus)
    console.print()
    console.print(table)


def _print_param_diff_table(console: Console, rows: list[dict]) -> None:
    """Print the parameter changes table."""
    param_diff_rows: list[tuple[str, str, Any, Any]] = []
    for r in rows:
        for key, bv, tv in r.get("config_diffs", []):
            param_diff_rows.append((r["label"], key, bv, tv))

    if not param_diff_rows:
        console.print(
            "\n[yellow]No config diffs found (initial_config.yml / "
            "final_config.yml may be missing).[/yellow]"
        )
        return

    table = Table(
        title="Parameter Changes (Initial → Final)",
        show_lines=True,
    )
    table.add_column("History", justify="left")
    table.add_column("Parameter", justify="left")
    table.add_column("Initial", justify="right")
    table.add_column("Final", justify="right")
    prev_label = None
    for label, key, bv, tv in param_diff_rows:
        display_label = label if label != prev_label else ""
        prev_label = label
        table.add_row(
            display_label,
            key,
            str(bv) if bv is not None else "—",
            str(tv) if tv is not None else "—",
        )
    console.print()
    console.print(table)

    # CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    param_csv_path = RESULTS_DIR / "param_changes.csv"
    with open(param_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "parameter", "initial", "final"])
        writer.writerows(param_diff_rows)
    console.print(f"Param changes written to: {param_csv_path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict]) -> Path:
    """Write the main comparison CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "comparison.csv"

    # Build column list from all prefixes x suffixes.
    phases = [
        ("holdout_initial", "holdout_final"),
        ("baseline_train", "final_train"),
        ("baseline_val", "final_val"),
    ]
    metric_keys: list[str] = []
    for initial_pfx, final_pfx in phases:
        for suffix in [*_VIOLATION_SUFFIXES, "cost"]:
            metric_keys.append(f"{initial_pfx}_{suffix}")
            metric_keys.append(f"{final_pfx}_{suffix}")

    fieldnames = [
        "scenario",
        "label",
        "slo_metric",
        *metric_keys,
        "num_checkpoints",
        "checkpoint_details",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out_row = {k: r.get(k) for k in fieldnames}
            cps = r.get("checkpoints", [])
            out_row["checkpoint_details"] = json.dumps(cps) if cps else ""
            writer.writerow(out_row)
    return csv_path


def _write_checkpoints_csv(rows: list[dict]) -> Path | None:
    """Write a per-checkpoint CSV."""
    if not any(r.get("checkpoints") for r in rows):
        return None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "checkpoints.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "checkpoint_idx", "time_s", "min_rpus"])
        for r in rows:
            for i, cp in enumerate(r.get("checkpoints", [])):
                writer.writerow(
                    [
                        r["label"],
                        i,
                        cp.get("time_s", ""),
                        json.dumps(cp.get("min_rpus", [])),
                    ]
                )
    return csv_path


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------


def _generate_scatter_plots(
    console: Console, rows: list[dict], run_root: Path
) -> None:
    """Generate cost-vs-violation scatter plots for each metric."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from autoslo.utils.colors import Palette
        from autoslo.utils.plotting import (
            ScatterPoint,
            cost_vs_compliance_scatter,
        )
    except ImportError:
        console.print(
            "[yellow]matplotlib not installed; skipping plot.[/yellow]"
        )
        return

    has_holdout = all(
        r.get("holdout_final_violation_rate") is not None for r in rows
    )
    if not has_holdout:
        console.print(
            "[yellow]Holdout data missing for some scenarios; "
            "skipping plot.[/yellow]"
        )
        return

    # Read SLO config from the first scenario's initial config.
    first_scenario_dir = run_root / f"tuner_{SCENARIOS[0]}"
    base_cfg = _load_yaml(first_scenario_dir / "initial_config.yml")
    slo_section = (base_cfg or {}).get("slo_config", {})
    slo_metric = slo_section.get("slo_metric", "binary")
    slo_threshold = slo_section.get("slo_threshold")
    slo_s = slo_section.get("slo_s")
    slo_dict_filename = slo_section.get("slo_dict_filename")

    slo_info = f"SLO: {slo_s}s" if (slo_s and not slo_dict_filename) else ""
    title_suffix = f" ({slo_info})" if slo_info else ""

    scenario_colors = {
        "prev_day": Palette.light_blue,
        "prev_week": Palette.dark_blue,
        "prev_month": Palette.dark_green,
    }

    # Collect static baselines from the first scenario that has them.
    static_ref: list[dict] | None = None
    for r in rows:
        if r.get("static_baselines"):
            static_ref = r["static_baselines"]
            break

    # Define the three violation metrics to plot.
    _METRIC_SPECS: list[tuple[str, str, str]] = [
        ("violation_rate", "SLO Violation Rate", "binary"),
        ("violation_amount_s", "Total SLO Violation Amount (s)", "absolute_s"),
        ("violation_relative_mean", "Mean Relative SLO Violation", "relative"),
    ]

    static_colors = [
        Palette.light_orange,
        Palette.dark_orange,
        Palette.light_red,
        Palette.dark_red,
    ]

    for metric_suffix, xlabel, metric_name in _METRIC_SPECS:
        initial_key = f"holdout_initial_{metric_suffix}"
        final_key = f"holdout_final_{metric_suffix}"

        points: list[ScatterPoint] = []

        # Initial (baseline) point — average across scenarios.
        initial_xs = [
            r[initial_key] for r in rows if r.get(initial_key) is not None
        ]
        initial_costs = [
            r["holdout_initial_cost"]
            for r in rows
            if r.get("holdout_initial_cost") is not None
        ]
        if initial_xs and initial_costs:
            points.append(
                ScatterPoint(
                    label="Initial (avg)",
                    x=sum(initial_xs) / len(initial_xs),
                    y=sum(initial_costs) / len(initial_costs),
                    color=Palette.gray,
                    marker="x",
                )
            )

        # Final (tuned) points per scenario.
        for r in rows:
            xv = r.get(final_key)
            yv = r.get("holdout_final_cost")
            if xv is not None and yv is not None:
                points.append(
                    ScatterPoint(
                        label=r["label"],
                        x=xv,
                        y=yv,
                        color=scenario_colors.get(
                            r["scenario"], Palette.dark_orange
                        ),
                    )
                )

        # Static baselines.
        if static_ref:
            for i, entry in enumerate(static_ref):
                xv = entry.get(metric_suffix, entry.get("violation", 0.0))
                points.append(
                    ScatterPoint(
                        label=entry.get("label", f"Static {i}"),
                        x=xv,
                        y=entry.get("cost", 0.0),
                        color=static_colors[i % len(static_colors)],
                        marker="s",
                    )
                )

        # Only shade the feasibility region on the plot matching the
        # configured slo_metric.
        if slo_threshold is not None and metric_name == slo_metric:
            viol_threshold = float(slo_threshold)
            if metric_name == "absolute_s":
                threshold_label = f"Target (≤{viol_threshold}s)"
            else:
                threshold_label = f"Target (≤{viol_threshold})"
        else:
            viol_threshold = None
            threshold_label = None

        fig, ax = cost_vs_compliance_scatter(
            points,
            xlabel=xlabel,
            ylabel="Cost ($)",
            title=f"History-Window Experiment: Cost vs {xlabel}{title_suffix}",
            x_threshold=viol_threshold,
            x_threshold_label=threshold_label,
            xscale="log" if metric_name == "absolute_s" else "linear",
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_name = f"holdout_{metric_suffix}.png"
        plot_path = RESULTS_DIR / plot_name
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        console.print(f"Plot written to: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate history-window experiment results.",
    )
    parser.add_argument(
        "--run-dir",
        default="data/tuner_runs/history_window_exp",
        help=(
            "Root directory containing tuner_prev_day/, tuner_prev_week/, "
            "tuner_prev_month/."
        ),
    )
    args = parser.parse_args()
    run_root = Path(args.run_dir)
    console = Console()

    # ---- Collect data ---------------------------------------------------
    rows: list[dict] = []
    for scenario in SCENARIOS:
        scenario_dir = run_root / f"tuner_{scenario}"
        row = _collect_scenario_data(scenario_dir, scenario)
        rows.append(row)

    # ---- Tables ---------------------------------------------------------
    # 1. Holdout comparison (target-day)
    console.print()
    _print_holdout_table(console, rows)

    # 2. Final tuned config: train & val vs baseline
    _print_train_val_table(
        console, rows, phase="final", title="Final vs Baseline"
    )

    # 3. Capacity checkpoints
    _print_checkpoint_table(console, rows)

    # 4. Parameter diffs
    _print_param_diff_table(console, rows)

    # ---- CSV export -----------------------------------------------------
    csv_path = _write_csv(rows)
    console.print(f"\nCSV written to: {csv_path}")

    ckpt_csv = _write_checkpoints_csv(rows)
    if ckpt_csv:
        console.print(f"Checkpoints written to: {ckpt_csv}")

    # ---- Scatter plots --------------------------------------------------
    _generate_scatter_plots(console, rows, run_root)


if __name__ == "__main__":
    main()
