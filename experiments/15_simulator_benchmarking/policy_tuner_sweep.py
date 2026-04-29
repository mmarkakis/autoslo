"""
policy_tuner_sweep.py
=====================
CLI script that sweeps a grid of policy parameters across multiple sampled
workload scenarios, identifies the Pareto-optimal trade-offs, and saves /
plots the results.

This implements the **Layer 3b** vision from the design doc: for every
candidate ``PolicyParams`` (a combination of ``eta_crit``,
``idle_periods_before_tear_down``, ``min_cluster_lifetime_s``, and
``min_rpu_override``), we run *N* sampled workload scenarios through the
``WorkloadSimulator`` and aggregate cost / violation statistics
into a ``ScenarioOutcome``.  The ``PolicyTuner`` then computes the
Pareto front over the full grid.

Usage example
-------------
::

    python policy_tuner_sweep.py \
        --workload_name redset_provisioned_cluster12 \
        --num_samples 10 \
        --eta_crit 0.05 0.10 0.20 \
        --idle_periods 5 10 15 \
        --min_cluster_lifetime_s 600 1200 1800 \
        --plot
"""

# DEPRECATED: This script is superseded by the new tuner CLI:
#   python -m autoslo.tuner --config ... --tuner-config ... --traces ...
# This file will be removed in a future release.

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import warnings
from datetime import datetime

warnings.warn(
    "policy_tuner_sweep.py is deprecated. "
    "Use 'python -m autoslo.tuner' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import pandas as pd
import yaml

import autoslo.utils.paths as pu
from autoslo.workload_definition.query import SloMetric
from autoslo.simulator.workload_simulator import (
    WorkloadSimulator,
)
from autoslo.capacity.policy_tuner import (
    DynamicClusterConfig,
    PolicyParams,
    PolicyTuner,
    ScenarioOutcome,
    SweepResult,
    plot_pareto_front,
    print_pareto_summary,
)
from autoslo.workload_definition.redset_workload import (
    RedsetWorkloadSamplingSpec,
)
from autoslo.workload_definition.tpcds_sampler import TPCDSSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_run_stats(
    sim: WorkloadSimulator,
) -> dict[str, float]:
    """Extract cost and violation statistics from the most recent run.

    Mirrors the logic in ``WorkloadSimulator._write_experiment_meta``
    but returns the values instead of persisting them.

    Returns
    -------
    dict with keys ``total_cost``, ``violation_rate``,
    ``violation_amount_s``, ``relative_violation``, ``num_queries``.
    """
    out_dir = sim._out_dir

    # --- Cost from billing analysis --------------------------------------
    total_cost = 0.0
    billing_path = os.path.join(out_dir, "billing_interval_analysis.yml")
    if os.path.exists(billing_path):
        with open(billing_path) as f:
            billing = yaml.safe_load(f) or {}
        for cluster_data in billing.values():
            total_cost += cluster_data.get("total_billed_cost", 0.0)

    # --- Violations from solve log ---------------------------------------
    violation_rate = 0.0
    violation_amount_s = 0.0
    relative_violation = 0.0
    num_queries = 0

    log_path = os.path.join(out_dir, "solve_log.parquet")
    if os.path.exists(log_path):
        log = pd.read_parquet(log_path)
        completions = log[log["event_type"] == "completion"].copy()
        num_queries = len(completions)
        if num_queries > 0 and sim._slo_s:
            durations = completions["latency_s"].fillna(0.0)
            per_row_slo = (
                completions["tpcds_temp_and_q_idx"]
                .map(sim._slo_resolver.resolve)
                .fillna(sim._slo_s)
            )
            violations_mask = durations > per_row_slo
            violation_rate = float(violations_mask.mean())
            violation_amount_s = float(
                (durations - per_row_slo).clip(lower=0.0).sum()
            )
            # Relative violation: mean of (observed - slo) / slo over
            # violating queries.  Zero when no violations.
            if violations_mask.any():
                relative_violation = float(
                    (
                        (
                            durations[violations_mask]
                            - per_row_slo[violations_mask]
                        )
                        / per_row_slo[violations_mask]
                    ).mean()
                )

    return {
        "total_cost": total_cost,
        "violation_rate": violation_rate,
        "violation_amount_s": violation_amount_s,
        "relative_violation": relative_violation,
        "num_queries": num_queries,
    }


def _run_scenarios(
    sim: WorkloadSimulator,
    base_sampling_spec: RedsetWorkloadSamplingSpec,
    num_samples: int,
    params: PolicyParams,
) -> ScenarioOutcome:
    """Run *num_samples* scenarios under the given policy params.

    Reconfigures the simulator's capacity-controller parameters, then
    runs ``reset()`` + ``simulate_one()`` for each sample seed.
    Aggregates per-scenario cost / violation stats into a
    ``ScenarioOutcome``.
    """
    # Reconfigure the simulator's stored CC parameters so that the next
    # reset() → _init_dynamic_clusters() picks up the new values.
    sim._cc_eta_crit = params.eta_crit
    sim._cc_idle_periods = params.idle_periods_before_tear_down
    sim._cc_min_cluster_lifetime_s = params.min_cluster_lifetime_s

    per_costs: list[float] = []
    per_viol_rates: list[float] = []
    per_viol_amounts: list[float] = []
    per_rel_viols: list[float] = []

    for i in range(num_samples):
        sampling_spec = RedsetWorkloadSamplingSpec(
            tpcds_prob_distribution_dir=base_sampling_spec.tpcds_prob_distribution_dir,
            seed=base_sampling_spec.seed + i,
            abs_start_time=base_sampling_spec.abs_start_time,
            abs_end_time=base_sampling_spec.abs_end_time,
            real_queries_per_output_queries=base_sampling_spec.real_queries_per_output_queries,
            real_s_per_output_s=base_sampling_spec.real_s_per_output_s,
        )

        sim.reset()
        sim.simulate_one(sampling_spec=sampling_spec)

        stats = _extract_run_stats(sim)
        per_costs.append(stats["total_cost"])
        per_viol_rates.append(stats["violation_rate"])
        per_viol_amounts.append(stats["violation_amount_s"])
        per_rel_viols.append(stats["relative_violation"])

        logger.info(
            "  Sample %d/%d — cost=%.4f, viol_rate=%.4f, viol_amount=%.2fs",
            i + 1,
            num_samples,
            stats["total_cost"],
            stats["violation_rate"],
            stats["violation_amount_s"],
        )

    return ScenarioOutcome(
        mean_cost=statistics.mean(per_costs),
        mean_violation_rate=statistics.mean(per_viol_rates),
        mean_violation_amount_s=statistics.mean(per_viol_amounts),
        mean_relative_violation=statistics.mean(per_rel_viols),
        per_scenario_costs=per_costs,
        per_scenario_violation_rates=per_viol_rates,
        per_scenario_violation_amounts_s=per_viol_amounts,
        per_scenario_relative_violations=per_rel_viols,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _sweep_result_to_dict(result: SweepResult) -> dict:
    """Convert a SweepResult to a JSON-serialisable dictionary."""

    def _entry_to_dict(entry):
        return {
            "params": {
                "eta_crit": entry.params.eta_crit,
                "idle_periods_before_tear_down": entry.params.idle_periods_before_tear_down,
                "min_cluster_lifetime_s": entry.params.min_cluster_lifetime_s,
                "min_rpu_override": entry.params.min_rpu_override,
            },
            "outcome": {
                "mean_cost": entry.outcome.mean_cost,
                "mean_violation_rate": entry.outcome.mean_violation_rate,
                "mean_violation_amount_s": entry.outcome.mean_violation_amount_s,
                "mean_relative_violation": entry.outcome.mean_relative_violation,
                "per_scenario_costs": entry.outcome.per_scenario_costs,
                "per_scenario_violation_rates": entry.outcome.per_scenario_violation_rates,
                "per_scenario_violation_amounts_s": entry.outcome.per_scenario_violation_amounts_s,
                "per_scenario_relative_violations": entry.outcome.per_scenario_relative_violations,
            },
            "is_pareto": entry.is_pareto,
        }

    cfg = None
    if result.cluster_config is not None:
        cfg = {
            "initial_rpus": list(result.cluster_config.initial_rpus),
            "allowed_rpu_sizes": list(result.cluster_config.allowed_rpu_sizes),
            "spin_up_delay_s": result.cluster_config.spin_up_delay_s,
        }

    return {
        "objective_x": result.objective_x,
        "objective_y": result.objective_y,
        "cluster_config": cfg,
        "num_entries": len(result.entries),
        "num_pareto": len(result.pareto_front),
        "entries": [_entry_to_dict(e) for e in result.entries],
        "pareto_front": [_entry_to_dict(e) for e in result.pareto_front],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    # --- Output directory (needed early for the log file) ----------------
    experiment_name = args.experiment_name or (
        f"sweep__{args.workload_name}__slo{args.slo_s}"
    )
    out_dir = os.path.join(
        pu.get_data_path(), "simulator_runs", experiment_name
    )
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "sweep.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.FileHandler(log_path, mode="a")],
    )
    print(f"Logging to {log_path}")

    # --- Sampling spec ---------------------------------------------------
    base_sampling_spec = RedsetWorkloadSamplingSpec(
        tpcds_prob_distribution_dir=args.tpcds_prob_distribution_dir,
        seed=42,
        abs_start_time=datetime(2024, 4, 1, 0, 0, 0),
        abs_end_time=datetime(2024, 4, 2, 0, 0, 0),
        real_queries_per_output_queries=24,
        real_s_per_output_s=24,
    )

    # --- Dynamic cluster config ------------------------------------------
    initial_rpus = tuple(args.initial_rpus)
    allowed_rpu_sizes = tuple(args.allowed_rpu_sizes)
    dynamic_config = DynamicClusterConfig(
        initial_rpus=initial_rpus,
        allowed_rpu_sizes=allowed_rpu_sizes,
        spin_up_delay_s=args.spin_up_delay_s,
    )

    # Use the first grid point's values as initial simulator construction
    # params (they will be overwritten before each sweep point).
    eta_crit_values = args.eta_crit
    idle_periods_values = args.idle_periods
    min_lifetime_values = args.min_cluster_lifetime_s
    min_rpu_override_values = args.min_rpu_override

    # --- Build the parameter grid ----------------------------------------
    grid = PolicyTuner.make_grid(
        eta_crit=eta_crit_values,
        idle_periods=idle_periods_values,
        min_cluster_lifetime_s=min_lifetime_values,
        min_rpu_override=min_rpu_override_values,
    )

    print(
        f"Policy Tuner Sweep\n"
        f"  Workload:             {args.workload_name}\n"
        f"  IconQ model:          {args.iconq_model_id}\n"
        f"  SLO:                  {args.slo_s}s\n"
        f"  Num samples/point:    {args.num_samples}\n"
        f"  Grid size:            {len(grid)} parameter combinations\n"
        f"  Total simulations:    {len(grid) * args.num_samples}\n"
        f"  eta_crit values:      {eta_crit_values}\n"
        f"  idle_periods values:  {idle_periods_values}\n"
        f"  min_lifetime values:  {min_lifetime_values}\n"
        f"  min_rpu_override:     {min_rpu_override_values}\n"
        f"  Dynamic config:       initial_rpus={initial_rpus}, "
        f"allowed={allowed_rpu_sizes}, delay={args.spin_up_delay_s}s\n"
        f"  Objectives:           x={args.objective_x}, y={args.objective_y}\n"
    )

    # --- Create simulator (loads heavy objects once) ---------------------
    simulator = WorkloadSimulator(
        workload_name=args.workload_name,
        iconq_model_id=args.iconq_model_id,
        blueprint_name="dynamic",
        slo_s=args.slo_s,
        slo_dict_filename=args.slo_dict_filename,
        slo_metric=SloMetric(args.slo_metric),
        slo_threshold=args.slo_threshold,
        verbose=True,
        export_video=False,
        experiment_name=experiment_name,
        overwrite_experiment=args.overwrite_experiment,
        dynamic_cluster_config=dynamic_config,
        eta_crit=eta_crit_values[0],
        idle_periods_before_tear_down=idle_periods_values[0],
        capacity_poll_interval_s=args.capacity_poll_interval_s,
        min_cluster_lifetime_s=min_lifetime_values[0],
    )

    # Warm up featurisation caches before the sweep.
    sampler = TPCDSSampler.from_dir(args.tpcds_prob_distribution_dir)
    tpcds_vocab = list(sampler.column_dict.values())
    simulator._iconq_model.iconq_query_featurizer.warm_up_cache(tpcds_vocab)
    print("Featurisation cache warm-up complete.\n")

    # --- Define simulate_fn for the tuner --------------------------------
    def simulate_fn(params: PolicyParams) -> ScenarioOutcome:
        return _run_scenarios(
            simulator, base_sampling_spec, args.num_samples, params
        )

    # --- Run the sweep ---------------------------------------------------
    tuner = PolicyTuner(
        grid=grid,
        simulate_fn=simulate_fn,
        cluster_config=dynamic_config,
    )

    sweep_start = datetime.now()
    result = tuner.sweep(
        objective_x=args.objective_x,
        objective_y=args.objective_y,
    )
    sweep_duration = (datetime.now() - sweep_start).total_seconds()

    # --- Report ----------------------------------------------------------
    print(f"\nSweep completed in {sweep_duration:.1f}s.")
    print_pareto_summary(result)

    # --- Save results ----------------------------------------------------
    results_path = os.path.join(out_dir, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(_sweep_result_to_dict(result), f, indent=2)
    print(f"Results saved to {results_path}")

    # --- Plot (optional) -------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt

        ax = plot_pareto_front(result, show=False)
        plot_path = os.path.join(out_dir, "pareto_front.png")
        ax.figure.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(ax.figure)
        print(f"Pareto front plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sweep policy parameters across sampled workload scenarios "
            "and identify Pareto-optimal trade-offs."
        ),
    )

    # -- Workload & model -------------------------------------------------
    parser.add_argument(
        "--workload_name",
        type=str,
        default="redset_provisioned_cluster12",
        help="Name of the workload to simulate.",
    )
    parser.add_argument(
        "--iconq_model_id",
        type=str,
        default="1771539369",
        help="ID of the IconQ model to use.",
    )
    parser.add_argument(
        "--tpcds_prob_distribution_dir",
        type=str,
        default=os.path.join(
            pu.get_data_path(), "generation_parameters", "dist_16_rpu"
        ),
        help="Directory containing the TPCDS probability distributions.",
    )

    # -- SLO --------------------------------------------------------------
    parser.add_argument(
        "--slo_s",
        type=float,
        default=180.0,
        help="Global SLO target in seconds.",
    )
    parser.add_argument(
        "--slo_dict_filename",
        type=str,
        default=None,
        help=(
            "YAML file under data/generation_parameters/ with per-template "
            "SLO overrides."
        ),
    )
    parser.add_argument(
        "--slo_metric",
        type=str,
        default="relative",
        choices=["binary", "absolute_s", "relative"],
        help="SLO violation metric for routing (binary, absolute_s, relative).",
    )
    parser.add_argument(
        "--slo_threshold",
        type=float,
        default=0.0,
        help="Threshold for the chosen SLO metric.",
    )

    # -- Sampling ---------------------------------------------------------
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of workload scenario samples per grid point.",
    )

    # -- Dynamic cluster config -------------------------------------------
    parser.add_argument(
        "--initial_rpus",
        type=int,
        nargs="+",
        default=[8],
        metavar="RPU",
        help="RPU sizes for initial clusters. E.g. --initial_rpus 8 16.",
    )
    parser.add_argument(
        "--allowed_rpu_sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        metavar="RPU",
        help="RPU sizes the capacity controller may spin up.",
    )
    parser.add_argument(
        "--spin_up_delay_s",
        type=float,
        default=120.0,
        help="Simulated spin-up delay in seconds.",
    )
    parser.add_argument(
        "--capacity_poll_interval_s",
        type=float,
        default=60.0,
        help="Capacity controller polling interval in seconds.",
    )

    # -- Policy grid axes (sweep ranges) ----------------------------------
    parser.add_argument(
        "--eta_crit",
        type=float,
        nargs="+",
        default=[0.1],
        help="eta_crit values to sweep. E.g. --eta_crit 0.05 0.10 0.20.",
    )
    parser.add_argument(
        "--idle_periods",
        type=int,
        nargs="+",
        default=[5],
        help=(
            "idle_periods_before_tear_down values to sweep. "
            "E.g. --idle_periods 5 10 15."
        ),
    )
    parser.add_argument(
        "--min_cluster_lifetime_s",
        type=float,
        nargs="+",
        default=[1200.0],
        help=(
            "Minimum cluster lifetime values to sweep (seconds). "
            "E.g. --min_cluster_lifetime_s 600 1200 1800."
        ),
    )
    parser.add_argument(
        "--min_rpu_override",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Fixed RPU overrides to sweep.  Omit for adaptive (None). "
            "E.g. --min_rpu_override 8 16."
        ),
    )

    # -- Objectives -------------------------------------------------------
    parser.add_argument(
        "--objective_x",
        type=str,
        default="mean_cost",
        choices=[
            "mean_cost",
            "mean_violation_rate",
            "mean_violation_amount_s",
            "mean_relative_violation",
        ],
        help="Primary Pareto objective (x-axis, minimised).",
    )
    parser.add_argument(
        "--objective_y",
        type=str,
        default="mean_violation_rate",
        choices=[
            "mean_cost",
            "mean_violation_rate",
            "mean_violation_amount_s",
            "mean_relative_violation",
        ],
        help="Secondary Pareto objective (y-axis, minimised).",
    )

    # -- Output -----------------------------------------------------------
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Experiment group name. Results saved under "
            "data/simulator_runs/<experiment_name>/. "
            "Defaults to 'sweep__<workload>__slo<slo_s>'."
        ),
    )
    parser.add_argument(
        "--overwrite_experiment",
        action="store_true",
        help="Overwrite existing experiment with the same name.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the Pareto front plot after the sweep.",
    )

    args = parser.parse_args()

    # Normalise min_rpu_override: None (CLI absent) → [None] for the grid.
    if args.min_rpu_override is None:
        args.min_rpu_override = [None]

    main(args)
