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
import json
from pathlib import Path
from typing import Any

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


_CHECKPOINT_KEY = "autoscaling_config.capacity_checkpoints"


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
        reservoir_meta = _load_yaml(
            scenario_dir / "reservoir" / "reservoir_meta.yml"
        )

        row = {
            "scenario": scenario,
            "label": LABELS[scenario],
            "reservoir_arrivals": (
                reservoir_meta.get("num_arrivals") if reservoir_meta else None
            ),
        }

        if holdout:
            row["holdout_baseline_violation"] = holdout.get(
                "baseline_violation"
            )
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
            (
                f"{r.get('reservoir_arrivals', '?'):,}"
                if r.get("reservoir_arrivals")
                else "?"
            ),
            f"{bv:.4f}" if bv is not None else "—",
            f"{tv:.4f}" if tv is not None else "—",
            dv_str or "—",
            f"{bc:.2f}" if bc is not None else "—",
            f"{tc:.2f}" if tc is not None else "—",
            dc_str or "—",
        )

    console.print(table)

    # ---- Configuration diff tables --------------------------------------
    param_diff_rows: list[tuple[str, str, Any, Any]] = (
        []
    )  # (label, key, base, tuned)
    checkpoint_rows: list[dict] = []  # per-scenario checkpoint summary

    for scenario in SCENARIOS:
        scenario_dir = run_root / scenario
        label = LABELS[scenario]
        pair = _load_config_pair(scenario_dir)
        if pair is None:
            checkpoint_rows.append({"label": label, "checkpoints": []})
            continue
        base_flat, tuned_flat = pair
        for key, bv, tv in _diff_configs(base_flat, tuned_flat):
            param_diff_rows.append((label, key, bv, tv))
        checkpoint_rows.append(
            {"label": label, "checkpoints": _extract_checkpoints(tuned_flat)}
        )

    if param_diff_rows:
        ptable = Table(
            title="Parameter Changes (Baseline → Tuned)",
            show_lines=True,
        )
        ptable.add_column("Scenario", justify="left")
        ptable.add_column("Parameter", justify="left")
        ptable.add_column("Baseline", justify="right")
        ptable.add_column("Tuned", justify="right")
        prev_label = None
        for label, key, bv, tv in param_diff_rows:
            display_label = label if label != prev_label else ""
            prev_label = label
            ptable.add_row(
                display_label,
                key,
                str(bv) if bv is not None else "—",
                str(tv) if tv is not None else "—",
            )
        console.print()
        console.print(ptable)

        param_csv_path = RESULTS_DIR / "param_changes.csv"
        with open(param_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "parameter", "baseline", "tuned"])
            writer.writerows(param_diff_rows)
        console.print(f"Param changes written to: {param_csv_path}")
    else:
        console.print(
            "\n[yellow]No config diffs found (initial_config.yml / "
            "final_config.yml may be missing).[/yellow]"
        )

    if any(cr["checkpoints"] for cr in checkpoint_rows):
        ctable = Table(
            title="Capacity Checkpoints Added by Tuner",
            show_lines=True,
        )
        ctable.add_column("Scenario", justify="left")
        ctable.add_column("# Checkpts", justify="right")
        ctable.add_column("Times (s)", justify="left")
        ctable.add_column("RPU Sizes", justify="left")
        for cr in checkpoint_rows:
            cps = cr["checkpoints"]
            if not cps:
                ctable.add_row(cr["label"], "0", "—", "—")
            else:
                times = ", ".join(str(cp.get("time_s", "?")) for cp in cps)
                rpus = ", ".join(str(cp.get("min_rpus", "?")) for cp in cps)
                ctable.add_row(cr["label"], str(len(cps)), times, rpus)
        console.print()
        console.print(ctable)

        ckpt_csv_path = RESULTS_DIR / "checkpoints.csv"
        with open(ckpt_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["scenario", "checkpoint_idx", "time_s", "min_rpus"]
            )
            for cr in checkpoint_rows:
                for i, cp in enumerate(cr["checkpoints"]):
                    writer.writerow(
                        [
                            cr["label"],
                            i,
                            cp.get("time_s", ""),
                            json.dumps(cp.get("min_rpus", [])),
                        ]
                    )
        console.print(f"Checkpoints written to: {ckpt_csv_path}")

    # Extend row dicts with config-diff info for CSV.
    for row in rows:
        scenario_dir = run_root / row["scenario"]
        pair = _load_config_pair(scenario_dir)
        if pair is None:
            row["num_checkpoints"] = None
            row["checkpoint_details"] = None
            continue
        _base_flat, tuned_flat = pair
        cps = _extract_checkpoints(tuned_flat)
        row["num_checkpoints"] = len(cps)
        row["checkpoint_details"] = json.dumps(cps) if cps else ""
        diffs = _diff_configs(_base_flat, tuned_flat)
        for key, _bv, tv in diffs:
            col = "tuned_" + key.rsplit(".", 1)[-1]
            row[col] = tv

    # ---- CSV ------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "comparison.csv"
    # Collect all tuned_* columns dynamically.
    tuned_cols = sorted({k for r in rows for k in r if k.startswith("tuned_")})
    fieldnames = [
        "scenario",
        "label",
        "reservoir_arrivals",
        "holdout_queries",
        "holdout_baseline_violation",
        "holdout_tuned_violation",
        "holdout_baseline_cost",
        "holdout_tuned_cost",
        "final_train_violation",
        "final_val_violation",
        "final_train_cost",
        "final_val_cost",
        "num_checkpoints",
        "checkpoint_details",
        *tuned_cols,
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\nCSV written to: {csv_path}")

    # ---- Scatter plot (optional, needs matplotlib) -----------------------
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
        r.get("holdout_tuned_violation") is not None for r in rows
    )
    if not has_holdout:
        console.print(
            "[yellow]Holdout data missing for some scenarios; "
            "skipping plot.[/yellow]"
        )
        return

    scenario_colors = {
        "prev_day": Palette.light_blue,
        "prev_week": Palette.dark_blue,
        "prev_month": Palette.dark_green,
    }

    points: list[ScatterPoint] = []

    # Baseline point — use the average across scenarios.
    baseline_viols = [r["holdout_baseline_violation"] for r in rows]
    baseline_costs = [r["holdout_baseline_cost"] for r in rows]
    points.append(
        ScatterPoint(
            label="Baseline",
            x=sum(baseline_viols) / len(baseline_viols),
            y=sum(baseline_costs) / len(baseline_costs),
            color=Palette.gray,
            marker="x",
        )
    )

    for r in rows:
        points.append(
            ScatterPoint(
                label=r["label"],
                x=r["holdout_tuned_violation"],
                y=r["holdout_tuned_cost"],
                color=scenario_colors.get(r["scenario"], Palette.dark_orange),
            )
        )

    # ---- Static baselines (D-SB7: deduplicate across scenarios) ---------
    _STATIC_TOL = 1e-6  # tolerance for cross-scenario consistency check

    static_lists: list[list[dict]] = []
    for scenario in SCENARIOS:
        holdout = _load_yaml(run_root / scenario / "holdout" / "summary.yml")
        if holdout and holdout.get("static_baselines"):
            static_lists.append(holdout["static_baselines"])

    if static_lists:
        # Use the first scenario's results as the reference.
        ref = static_lists[0]

        # Verify consistency across scenarios.
        for idx, sl in enumerate(static_lists[1:], start=1):
            if len(sl) != len(ref):
                console.print(
                    f"[yellow]Warning: static_baselines count differs in "
                    f"{SCENARIOS[idx]} ({len(sl)}) vs {SCENARIOS[0]} "
                    f"({len(ref)}). Using {SCENARIOS[0]}.[/yellow]"
                )
                continue
            for j, (r_entry, s_entry) in enumerate(zip(ref, sl)):
                for metric in ("violation", "cost"):
                    rv = r_entry.get(metric, 0.0)
                    sv = s_entry.get(metric, 0.0)
                    if abs(rv - sv) > _STATIC_TOL:
                        console.print(
                            f"[yellow]Warning: static baseline "
                            f"'{r_entry.get('label', j)}' {metric} differs: "
                            f"{SCENARIOS[0]}={rv}, "
                            f"{SCENARIOS[idx]}={sv}. "
                            f"Using {SCENARIOS[0]}.[/yellow]"
                        )

        # Use distinct markers and warm colours for static baselines.
        static_colors = [
            Palette.light_orange,
            Palette.dark_orange,
            Palette.light_red,
            Palette.dark_red,
        ]
        for i, entry in enumerate(ref):
            points.append(
                ScatterPoint(
                    label=entry.get("label", f"Static {i}"),
                    x=entry.get("violation", 0.0),
                    y=entry.get("cost", 0.0),
                    color=static_colors[i % len(static_colors)],
                    marker="s",
                )
            )

    # Read SLO config from the first scenario's initial config.
    base_cfg = _load_yaml(run_root / SCENARIOS[0] / "initial_config.yml")
    slo_section = (base_cfg or {}).get("slo_config", {})
    slo_metric = slo_section.get("slo_metric", "binary")
    slo_threshold = slo_section.get("slo_threshold")
    slo_s = slo_section.get("slo_s")
    slo_dict_filename = slo_section.get("slo_dict_filename")

    _METRIC_LABELS = {
        "binary": "SLO Violation Rate",
        "absolute_s": "Total SLO Violation Amount (s)",
        "relative": "Mean Relative SLO Violation",
    }
    xlabel = _METRIC_LABELS.get(slo_metric, f"SLO Violation ({slo_metric})")

    # Build the threshold label.
    if slo_threshold is not None:
        viol_threshold = float(slo_threshold)
        if slo_metric == "absolute_s":
            threshold_label = f"Target (≤{viol_threshold}s)"
        else:
            threshold_label = f"Target (≤{viol_threshold})"
    else:
        raise ValueError(
            "slo_threshold is required in slo_config for plotting."
        )

    slo_info = f"SLO: {slo_s}s" if (slo_s and not slo_dict_filename) else ""
    title_suffix = f" ({slo_info}, metric={slo_metric})" if slo_info else ""

    fig, ax = cost_vs_compliance_scatter(
        points,
        xlabel=xlabel,
        ylabel="Cost ($)",
        title=f"History-Window Experiment: Cost vs Compliance{title_suffix}",
        x_threshold=viol_threshold,
        x_threshold_label=threshold_label,
    )

    plot_path = RESULTS_DIR / "holdout_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
