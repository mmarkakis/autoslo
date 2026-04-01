"""Generate a visual report for a tuner run.

Usage
-----
::

    python -m autoslo.tuner.generate_report --run-dir data/tuner_runs/tuner_20250101_120000

Generates all applicable visualizations and saves them to ``<run_dir>/plots/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from autoslo.tuner import visualizations as viz

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save(fig: object | None, plots_dir: Path, name: str) -> None:
    """Save a matplotlib Figure to *plots_dir* as PNG."""
    if fig is None:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")  # type: ignore[union-attr]
    logger.info("Saved %s", out)
    import matplotlib.pyplot as plt

    plt.close(fig)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Phase-specific generators
# ------------------------------------------------------------------


def _gen_reservoir(run_dir: Path, plots_dir: Path) -> None:
    """V1: Reservoir & forecast visualizations."""
    res_dir = run_dir / "reservoir"
    meta_path = res_dir / "reservoir_meta.yml"
    meta = _load_yaml(meta_path)
    if not meta:
        logger.info("Skipping reservoir plots (no reservoir_meta.yml)")
        return

    # V1.1 — Windowed-template diagnostics.
    fig = viz.plot_windowed_template_diagnostics(meta)
    _save(fig, plots_dir, "v1_1_windowed_template_diagnostics")

    # V1.4 — Weight decay (static, uses defaults).
    fig = viz.plot_forecast_weight_decay()
    _save(fig, plots_dir, "v1_4_weight_decay")

    # V1 existing — Reservoir heatmap & hourly rates.
    # These functions expect a QueryReservoir object, not a raw DataFrame.
    # res_parquet = res_dir / "reservoir.parquet"
    # if res_parquet.exists():
    #     from autoslo.tuner.reservoir import QueryReservoir

    #     reservoir = QueryReservoir.load(res_dir)
    #     fig = viz.plot_reservoir_heatmap(reservoir)
    #     _save(fig, plots_dir, "reservoir_heatmap")

    #     init_cfg = _load_yaml(run_dir / "initial_config.yml") or {}
    #     slo_cfg = init_cfg.get("slo_config", {})
    #     target_start = slo_cfg.get("target_start")
    #     target_end = slo_cfg.get("target_end")
    #     if target_start and target_end:
    #         from datetime import datetime

    #         ts = datetime.fromisoformat(str(target_start))
    #         te = datetime.fromisoformat(str(target_end))
    #         fig = viz.plot_hourly_rates(reservoir, ts, te)
    #         _save(fig, plots_dir, "hourly_rates")


def _gen_sampling(run_dir: Path, plots_dir: Path) -> None:
    """V1.2–V1.3: Sampling fidelity visualizations."""
    import pandas as pd

    sampled_train = run_dir / "sampled_workloads" / "train"
    if not sampled_train.exists():
        logger.info("Skipping sampling plots (no sampled_workloads/train)")
        return

    # # V1 existing — workload arrivals for each scenario.
    # parquets = sorted(sampled_train.glob("*.parquet"))
    # for i, pq in enumerate(parquets[:3]):  # Limit to first 3 to avoid huge reports.
    #     wdf = pd.read_parquet(pq)
    #     if "abs_start_time" in wdf.columns:
    #         fig = viz.plot_workload_arrivals(wdf, title=f"Train scenario {i}")
    #         _save(fig, plots_dir, f"workload_arrivals_train_{i:03d}")

    # # V1 existing — template frequency for first scenario.
    # if parquets:
    #     wdf = pd.read_parquet(parquets[0])
    #     if "query_text_id" in wdf.columns:
    #         fig = viz.plot_template_frequency(wdf)
    #         _save(fig, plots_dir, "template_frequency_train")
    #         fig = viz.plot_query_count_distribution(wdf)
    #         _save(fig, plots_dir, "query_count_distribution_train")


def _gen_baseline(run_dir: Path, plots_dir: Path) -> None:
    """V2: Baseline evaluation visualizations."""
    baseline_dir = run_dir / "baseline"
    summary_path = baseline_dir / "summary.yml"
    if not summary_path.exists():
        logger.info("Skipping baseline plots (no baseline/summary.yml)")
        return

    # V2.1 — Scenario metric distributions.
    fig = viz.plot_scenario_distributions(summary_path, title="Baseline — Scenario Distributions")
    _save(fig, plots_dir, "v2_1_baseline_scenario_distributions")

    # V2.2 — Cost breakdown per scenario.
    for split in ["train", "val"]:
        split_dir = baseline_dir / split
        if not split_dir.exists():
            continue
        scenario_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        if scenario_dirs:
            fig = viz.plot_cost_breakdown_by_cluster(
                scenario_dirs, title=f"Baseline {split.title()} — Cost by Cluster"
            )
            _save(fig, plots_dir, f"v2_2_baseline_{split}_cost_breakdown")


def _gen_checkpoints(run_dir: Path, plots_dir: Path) -> None:
    """V3: Checkpoint optimization visualizations."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        logger.info("Skipping checkpoint plots (no checkpoints/)")
        return

    # V3.1 — Violation-window timeline.
    fig = viz.plot_violation_window_timeline(run_dir)
    _save(fig, plots_dir, "v3_1_violation_window_timeline")

    # V3.2 — Checkpoint round trajectory.
    fig = viz.plot_checkpoint_round_trajectory(run_dir)
    _save(fig, plots_dir, "v3_2_checkpoint_round_trajectory")

    # V3.3 — RPU candidate comparison.
    fig = viz.plot_rpu_candidate_comparison(run_dir)
    _save(fig, plots_dir, "v3_3_rpu_candidate_comparison")


def _gen_sweep(
    run_dir: Path,
    plots_dir: Path,
    phase_name: str,
    sweep_subdir: str,
) -> None:
    """V4: Sweep analysis visualizations for a given phase."""
    sweep_dir = run_dir / sweep_subdir
    sweep_results_path = sweep_dir / "sweep_results.json"
    if not sweep_results_path.exists():
        logger.info("Skipping %s sweep plots (no sweep_results.json)", phase_name)
        return

    # Determine SLO threshold.
    init_cfg = _load_yaml(run_dir / "initial_config.yml") or {}
    slo_cfg = init_cfg.get("slo_config", {})
    slo_threshold = slo_cfg.get("slo_threshold")

    prefix = sweep_subdir.lower()

    # V4.1 — Pareto scatter.
    fig = viz.plot_sweep_pareto(
        sweep_results_path,
        phase_name=phase_name,
        slo_threshold=float(slo_threshold) if slo_threshold is not None else None,
    )
    _save(fig, plots_dir, f"v4_1_{prefix}_pareto")

    # V4.2 — Heatmap (only if 2 params).
    fig = viz.plot_sweep_heatmap(sweep_results_path, phase_name=phase_name)
    if fig is not None:
        _save(fig, plots_dir, f"v4_2_{prefix}_heatmap")

    # V4.3 — Train vs. val agreement.
    fig = viz.plot_train_val_agreement(sweep_results_path, phase_name=phase_name)
    _save(fig, plots_dir, f"v4_3_{prefix}_train_val_agreement")


def _gen_simulator_diagnostics(run_dir: Path, plots_dir: Path) -> None:
    """V5: Per-scenario simulator diagnostics (uses first train or val scenario)."""
    init_cfg = _load_yaml(run_dir / "initial_config.yml") or {}
    slo_s = float((init_cfg.get("slo_config") or {}).get("slo_s", 10.0))

    # Pick a representative scenario: prefer final/train/0, else baseline/train/0.
    scenario_dir = None
    for phase in ["final", "baseline"]:
        phase_dir = run_dir / phase / "train"
        if phase_dir.exists():
            subs = sorted(d for d in phase_dir.iterdir() if d.is_dir())
            if subs:
                scenario_dir = subs[0]
                break

    if scenario_dir is None:
        logger.info("Skipping simulator diagnostics (no scenario dirs found)")
        return

    slog_path = scenario_dir / "structured_log.parquet"
    billing_path = scenario_dir / "billing_interval_analysis.yml"

    if slog_path.exists():
        # V5.1 — Cluster Gantt.
        fig = viz.plot_cluster_gantt(slog_path, title=f"Cluster Lifecycle — {scenario_dir.name}")
        _save(fig, plots_dir, "v5_1_cluster_gantt")

        # V5.2 — Headroom.
        tuner_cfg = _load_yaml(run_dir / "tuner_config.yml") or {}
        eta_crit = tuner_cfg.get("eta_crit")
        fig = viz.plot_headroom_timeseries(
            slog_path,
            eta_crit=float(eta_crit) if eta_crit is not None else None,
        )
        _save(fig, plots_dir, "v5_2_headroom_timeseries")

        # V5.3 — Latency vs SLO.
        fig = viz.plot_latency_vs_slo(slog_path, slo_s=slo_s)
        _save(fig, plots_dir, "v5_3_latency_vs_slo")

        # V5.4 — Routing distribution.
        fig = viz.plot_routing_distribution(slog_path)
        _save(fig, plots_dir, "v5_4_routing_distribution")

    if billing_path.exists():
        # V5.5 — Billing utilisation.
        fig = viz.plot_billing_utilisation(billing_path)
        _save(fig, plots_dir, "v5_5_billing_utilisation")


def _gen_evolution(run_dir: Path, plots_dir: Path) -> None:
    """V6: Cross-phase summary visualizations."""
    evo_path = run_dir / "evolution.parquet"
    if evo_path.exists():
        # V6.1 — Evolution strip.
        fig = viz.plot_evolution_strip(evo_path)
        _save(fig, plots_dir, "v6_1_evolution_strip")

    # V6.2 — Holdout comparison.
    holdout_summary = run_dir / "holdout" / "summary.yml"
    if holdout_summary.exists():
        init_cfg = _load_yaml(run_dir / "initial_config.yml") or {}
        slo_threshold = (init_cfg.get("slo_config") or {}).get("slo_threshold")
        fig = viz.plot_holdout_comparison(
            holdout_summary,
            slo_threshold=float(slo_threshold) if slo_threshold is not None else None,
        )
        _save(fig, plots_dir, "v6_2_holdout_comparison")

    # V6.3 — Phase waterfall: violation & cost.
    if (run_dir / "baseline" / "summary.yml").exists() and (run_dir / "final" / "summary.yml").exists():
        for metric in ["violation", "cost"]:
            fig = viz.plot_phase_waterfall(run_dir, metric=metric)
            _save(fig, plots_dir, f"v6_3_phase_waterfall_{metric}")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def generate_report(run_dir: Path) -> Path:
    """Generate all applicable visualizations for a tuner run.

    Parameters
    ----------
    run_dir :
        Root tuner run directory (e.g. ``data/runs/tuner_20250101_120000``).

    Returns
    -------
    Path
        The plots output directory.
    """
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating report for %s → %s", run_dir, plots_dir)

    _gen_reservoir(run_dir, plots_dir)
    _gen_sampling(run_dir, plots_dir)
    _gen_baseline(run_dir, plots_dir)
    _gen_checkpoints(run_dir, plots_dir)
    _gen_sweep(run_dir, plots_dir, "Autoscaler", "autoscaler")
    _gen_sweep(run_dir, plots_dir, "Routing", "routing")
    _gen_simulator_diagnostics(run_dir, plots_dir)
    _gen_evolution(run_dir, plots_dir)

    logger.info("Report complete: %d plots generated", len(list(plots_dir.glob("*.png"))))
    return plots_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a visual report for a tuner run.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the tuner run output directory.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"Run directory not found: {run_dir}")

    generate_report(run_dir)


if __name__ == "__main__":
    main()
