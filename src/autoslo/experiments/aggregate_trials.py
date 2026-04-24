"""Aggregate results for any template-driven experiment.

Reads a ``trial_spec.yml``, discovers the tuner output directory for each
trial, and produces Rich comparison tables and CSV exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

import autoslo.utils.paths as pu
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.utils.plotting import (
    ScatterPoint,
    cost_vs_compliance_scatter,
    plot_legend_to,
)
from autoslo.utils.yaml_helpers import load_yaml


def consolidate_summaries(
    experiment_definition_dir: Path | str,
) -> pd.DataFrame:
    """
    Consolidate multiple trial summaries into a dataframe.
    """
    # Read trial spec from the experiment dir and construct output directory
    # of the tuner run.
    if isinstance(experiment_definition_dir, str):
        experiment_definition_dir = Path(experiment_definition_dir)
    with open(experiment_definition_dir / "trial_spec.yml") as f:
        trial_spec = yaml.safe_load(f)
    experiment_name = trial_spec["experiment_name"]
    experiment_dir = os.path.join(
        pu.AUTOSLO_ROOT, "data", "tuner_runs", experiment_name
    )

    rows = []
    # Iterate over subdirectories and read their summary.yml files.
    for trial in trial_spec["trials"]:
        trial_id = trial["trial_id"]
        summary_path = os.path.join(
            experiment_dir, f"tuner_{trial_id}", "09_holdout", "summary.yml"
        )
        if not os.path.exists(summary_path):
            raise FileNotFoundError(
                f"Warning: summary.yml not found for trial {trial_id} at {summary_path}"
            )
        with open(summary_path) as f:
            summary = yaml.safe_load(f)
        # Add rows for initial and final
        for point in ["initial", "final"]:
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "trial": trial_id,
                    "label": point,
                    "cost": summary[f"{point}_cost"],
                    "violation_rate": summary[f"{point}_violation_rate"],
                    "violation_amount_s": summary[
                        f"{point}_violation_amount_s"
                    ],
                    "violation_relative_mean": summary[
                        f"{point}_violation_relative_mean"
                    ],
                }
            )
        # Add rows for static baselines.
        for baseline in summary.get("static_baselines", []):
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "trial": trial_id,
                    "label": baseline["label"],
                    "cost": baseline["cost"],
                    "violation_rate": baseline["violation_rate"],
                    "violation_amount_s": baseline["violation_amount_s"],
                    "violation_relative_mean": baseline[
                        "violation_relative_mean"
                    ],
                }
            )

    return pd.DataFrame(rows)


def plot_experiment(experiment_definition_dir: Path | str) -> None:
    if isinstance(experiment_definition_dir, str):
        experiment_definition_dir = Path(experiment_definition_dir)
    summary = consolidate_summaries(experiment_definition_dir)
    with open(experiment_definition_dir / "trial_spec.yml") as f:
        trial_spec = yaml.safe_load(f)
    experiment_name_human = trial_spec["experiment_name_human"]
    plot_on_one_panel = trial_spec.get("plot_on_one_panel", False)

    # Collect the trial-wise slo_objectives and plotting info.
    slo_objectives: dict[str, SloObjective] = {}
    formatting_ids_of_final_points: dict[str, str] = {}
    trial_ids_human: dict[str, str] = {}
    for trial in trial_spec["trials"]:
        trial_id = trial["trial_id"]
        trial_config_path = (
            experiment_definition_dir / "configs" / f"tuner_{trial_id}.yml"
        )
        trial_config = load_yaml(trial_config_path)
        slo_objectives[trial_id] = SloObjective.from_config(trial_config)
        formatting_ids_of_final_points[trial_id] = trial[
            "formatting_id_of_final_point"
        ]
        trial_ids_human[trial_id] = trial["trial_id_human"]

    # Create one plot per SLO metric.
    for slo_metric_name in ["binary", "absolute_s", "relative"]:

        slo_metric = SloMetric(slo_metric_name)
        col_name = slo_metric.to_column_name()

        if not plot_on_one_panel:
            # Create one panel for each trial.
            num_trials = summary["trial"].nunique()
            if num_trials == 3:
                fig, axs = plt.subplots(
                    3, 1, figsize=(10, 12), sharex=True, sharey=True
                )
            elif num_trials == 4:
                fig, axs = plt.subplots(
                    2, 2, figsize=(12, 10), sharex=True, sharey=True
                )
            axs = axs.flatten()

            prev_panel_xlims: Optional[tuple[float, float]] = None
            prev_panel_ylims: Optional[tuple[float, float]] = None
            for panel_idx, (trial_id, trial_rows) in enumerate(
                summary.groupby("trial", sort=False)
            ):
                points: list[ScatterPoint] = []
                for _, row in trial_rows.iterrows():
                    formatting_id = row["label"]
                    if formatting_id == "final":
                        formatting_id = formatting_ids_of_final_points[trial_id]
                    points.append(
                        ScatterPoint(
                            formatting_id=formatting_id,
                            x=row[col_name],
                            y=row["cost"],
                        )
                    )

                _, _, prev_panel_xlims, prev_panel_ylims = (
                    cost_vs_compliance_scatter(
                        points,
                        x_metric=slo_metric,
                        title=trial_ids_human[trial_id],
                        x_threshold_objective=slo_objectives[trial_id],
                        ax=axs[panel_idx],  # type: ignore
                        existing_xlims=prev_panel_xlims,
                        existing_ylims=prev_panel_ylims,
                    )
                )
        else:
            # Assert that for each non-final point, we see the same values
            # across all trials (i.e. they are truly static baselines).
            nonfinals = summary[summary["label"] != "final"].drop(
                columns=["trial"]
            )
            for _, label_df in nonfinals.groupby("label", sort=False):
                if not label_df.nunique().eq(1).all():
                    raise ValueError(
                        f"Non-final points for label "
                        f"'{label_df['label'].iloc[0]}' are not the same "
                        f"across trials."
                    )

            # Also assert that they all have the same SLO objective.
            if not len(set(slo_objectives.values())) == 1:
                raise ValueError(
                    f"SLO objective for trial {trial_id} does not match."
                )

            # If we got here, we can plot all points on the same panel.
            points: list[ScatterPoint] = []

            for _, row in summary[summary["label"] == "final"].iterrows():
                points.append(
                    ScatterPoint(
                        formatting_id=formatting_ids_of_final_points[
                            row["trial"]
                        ],
                        x=row[col_name],
                        y=row["cost"],
                    )
                )
            for _, row in nonfinals.drop_duplicates().iterrows():
                points.append(
                    ScatterPoint(
                        formatting_id=row["label"],
                        x=row[col_name],
                        y=row["cost"],
                    )
                )

            fig, _, _, _ = cost_vs_compliance_scatter(
                points,
                title=experiment_name_human,
                x_metric=slo_metric,
                x_threshold_objective=slo_objectives[summary["trial"].iloc[0]],
            )

        plot_path = (
            experiment_definition_dir / "plots" / f"{slo_metric_name}.png"
        )
        os.makedirs(plot_path.parent, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plot_legend_to(experiment_definition_dir / "plots" / "legend.png")
        plt.close(fig)
        print(f"Wrote {plot_path}.")


# ---------------------------------------------------------------------------
# YAML / config helpers (unchanged from aggregate_results.py)
# ---------------------------------------------------------------------------


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


_CHECKPOINT_KEY = "capacity_checkpoints"


def _diff_configs(
    base_flat: dict[str, Any], tuned_flat: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Return (key, base_val, tuned_val) for differing keys.

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
# Rich / formatting helpers
# ---------------------------------------------------------------------------

_VIOLATION_SUFFIXES = [
    "violation_rate",
    "violation_amount_s",
    "violation_relative_mean",
]

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
    label_column_name: str = "Scenario",
) -> Table:
    """Build a Rich table comparing initial vs final for each scenario.

    Each row has: scenario label, then for each of the 4 metrics
    (3 violations + cost): initial value, final value, Δ.
    """
    table = Table(title=title, show_lines=True)
    table.add_column(label_column_name, justify="left")
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
    scenario_dir: Path,
    scenario_id: str,
    plot_label: str,
    is_reference: bool = False,
) -> dict[str, Any]:
    """Collect all metrics for a single scenario into a flat dict."""
    row: dict[str, Any] = {
        "scenario": scenario_id,
        "label": plot_label,
        "is_reference": is_reference,
    }

    # --- Holdout data (Phase 9: target-period evaluation) -----------------
    holdout = _load_yaml(scenario_dir / "09_holdout" / "summary.yml")
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
        summary = _load_yaml(scenario_dir / "07_final" / f"{split}_summary.yml")
        if summary:
            for suffix in _VIOLATION_SUFFIXES:
                row[f"final_{split}_{suffix}"] = summary.get(suffix)
            row[f"final_{split}_cost"] = summary.get("cost")

    # --- Baseline train/val summaries (Phase 3) ---------------------------
    for split in ("train", "val"):
        summary = _load_yaml(
            scenario_dir / "03_baseline" / f"{split}_summary.yml"
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


def _print_holdout_table(
    console: Console, rows: list[dict], trial_column_label: str
) -> None:
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
        label_column_name=trial_column_label,
    )
    console.print(table)


def _print_train_val_table(
    console: Console,
    rows: list[dict],
    phase: str,
    title: str,
    trial_column_label: str,
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
            label_column_name=trial_column_label,
        )
        console.print()
        console.print(table)


def _print_checkpoint_table(
    console: Console, rows: list[dict], trial_column_label: str
) -> None:
    """Print the capacity checkpoints table."""
    if not any(r.get("checkpoints") for r in rows):
        return
    table = Table(
        title="Capacity Checkpoints Added by Tuner",
        show_lines=True,
    )
    table.add_column(trial_column_label, justify="left")
    table.add_column("# Checkpts", justify="right")
    table.add_column("Times (s)", justify="left")
    table.add_column("RPU Sizes", justify="left")
    for r in rows:
        cps = r.get("checkpoints", [])
        if not cps:
            table.add_row(r["label"], "0", "—", "—")
        else:
            times = ", ".join(
                f"{cp.get('rel_time_s', cp.get('time_s', '?')):.0f}"
                for cp in cps
            )
            rpus = ", ".join(str(cp.get("min_rpus", "?")) for cp in cps)
            table.add_row(r["label"], str(len(cps)), times, rpus)
    console.print()
    console.print(table)


def _print_param_diff_table(
    console: Console,
    rows: list[dict],
    trial_column_label: str,
    results_dir: Path,
) -> None:
    """Print the parameter changes table and export a CSV."""
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
    table.add_column(trial_column_label, justify="left")
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
    results_dir.mkdir(parents=True, exist_ok=True)
    param_csv_path = results_dir / "param_changes.csv"
    with open(param_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "parameter", "initial", "final"])
        writer.writerows(param_diff_rows)
    console.print(f"Param changes written to: {param_csv_path}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], results_dir: Path) -> Path:
    """Write the main comparison CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "comparison.csv"

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


def _write_checkpoints_csv(rows: list[dict], results_dir: Path) -> Path | None:
    """Write a per-checkpoint CSV."""
    if not any(r.get("checkpoints") for r in rows):
        return None
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "checkpoints.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "checkpoint_idx", "time_s", "min_rpus"])
        for r in rows:
            for i, cp in enumerate(r.get("checkpoints", [])):
                writer.writerow(
                    [
                        r["label"],
                        i,
                        cp.get("time_s", cp.get("rel_time_s", "?")),
                        json.dumps(cp.get("min_rpus", [])),
                    ]
                )
    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate results for a template-driven sweep experiment.",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to the trial_spec.yml file.",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        parser.error(f"Spec file not found: {spec_path}")

    spec_dir = spec_path.parent
    spec = _load_spec(spec_path)
    console = Console()

    # --- Resolve paths from spec -----------------------------------------
    experiment_name: str = spec["experiment_name"]
    run_root = Path("data") / "tuner_runs" / experiment_name
    results_dir = spec_dir / spec.get("results_dir", "results")
    trial_column_label: str = spec.get("trial_column_label", "Scenario")

    scenarios_list = sorted(
        spec.get("scenarios", []), key=lambda s: s.get("sort_order", 0)
    )
    if not scenarios_list:
        parser.error("No scenarios found in spec.")

    # --- Validate that run_root exists -----------------------------------
    if not run_root.exists():
        parser.error(
            f"Run directory does not exist: {run_root}.\n"
            f"Run the tuner first with:\n"
            f"  python src/autoslo/experiments/run.py {args.spec}"
        )

    missing_dirs = [
        str(run_root / f"tuner_{s['id']}")
        for s in scenarios_list
        if not (run_root / f"tuner_{s['id']}").exists()
    ]
    if missing_dirs:
        parser.error(
            "Missing expected scenario directories under run root "
            f"({run_root}):\n  " + "\n  ".join(missing_dirs)
        )

    # --- Collect data per scenario ---------------------------------------
    rows: list[dict] = []
    for scenario in scenarios_list:
        sid = scenario["id"]
        scenario_dir = run_root / f"tuner_{sid}"
        row = _collect_scenario_data(
            scenario_dir=scenario_dir,
            scenario_id=sid,
            plot_label=scenario.get("plot_label", sid),
            is_reference=scenario.get("is_reference", False),
        )
        rows.append(row)

    # --- Tables ----------------------------------------------------------
    console.print()
    _print_holdout_table(console, rows, trial_column_label)

    _print_train_val_table(
        console,
        rows,
        phase="final",
        title="Final vs Baseline",
        trial_column_label=trial_column_label,
    )

    _print_checkpoint_table(console, rows, trial_column_label)

    _print_param_diff_table(console, rows, trial_column_label, results_dir)

    # --- CSV export ------------------------------------------------------
    csv_path = _write_csv(rows, results_dir)
    console.print(f"\nCSV written to: {csv_path}")

    ckpt_csv = _write_checkpoints_csv(rows, results_dir)
    if ckpt_csv:
        console.print(f"Checkpoints written to: {ckpt_csv}")

    console.print(
        "\n[dim]Run plot_results.py in this experiment directory to "
        "generate scatter plots.[/dim]"
    )


if __name__ == "__main__":
    main()
