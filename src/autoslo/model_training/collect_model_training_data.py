import argparse
import asyncio
import csv
import itertools
import os
import re
import time
from pathlib import Path

from rich import print
from rich.progress import track
from rich.table import Table

from autoslo.config.component_configs import WorkloadConfig
from autoslo.config.utils import (
    copy_and_apply_overrides,
    make_run_id,
    parse_params,
)
from autoslo.filesystem.path_utils import (
    append_to_run_log,
    find_most_recent_live_run_id,
    get_runs_path,
)
from autoslo.filesystem.yaml_helpers import load_yaml_with_params
from autoslo.workload_definition.poisson_workload_creator import (
    PoissonArrivalPhase,
    PoissonWorkloadCreator,
)
from autoslo.workload_definition.workload import Workload
from autoslo.workload_execution.workload_runner import WorkloadRunner

# (num_templates, num_query_texts_per_template, num_queries_per_query_text,
# poisson_lambda)
UNPHASED_COMBOS = [
    (66, 2, 1, 0.1),
    (66, 2, 1, 0.05),
    (66, 3, 1, 0.1),
    (66, 3, 1, 0.05),
    (99, 2, 1, 0.1),
    (99, 2, 1, 0.05),
    (99, 3, 1, 0.1),
    (99, 3, 1, 0.05),
    (99, 3, 1, 0.2),
    (99, 3, 1, 0.5),
    (99, 3, 1, 1.0),
]

# (num_templates, num_query_texts_per_template,
# total_num_queries, num_lull_burst_cycles, lull_poisson_lambda,
# burst_poisson_lambda, fraction_of_queries_in_bursts)
PHASED_COMBOS = [
    (99, 3, 300, 6, 0.05, 0.2, 0.2),  # Mild intensity, medium burstiness
    (99, 3, 300, 12, 0.05, 0.2, 0.4),  # Mild intensity, high burstiness
    (99, 3, 300, 6, 0.05, 0.5, 0.2),  # Medium intensity, medium burstiness
    (99, 3, 300, 12, 0.05, 0.5, 0.4),  # Medium intensity, high burstiness
]


SEED = 42


def _fmt(seconds: float) -> str:
    total_s = int(seconds)
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    frac = seconds - total_s
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s + frac:.2f}s")
    return f"{seconds:.2f} s  ({' '.join(parts)})"


def create_training_workloads() -> list[Workload]:
    """Create (or overwrite) all training workloads and return them."""

    workloads = []

    # Unphased workloads.
    unphased_table = Table(title=f"Unphased Training workloads")
    unphased_table.add_column("subset", justify="left")
    unphased_table.add_column("num_templates", justify="right")
    unphased_table.add_column("num_query_texts_per_template", justify="right")
    unphased_table.add_column("num_queries_per_query_text", justify="right")
    unphased_table.add_column("poisson_lambda", justify="right")
    unphased_table.add_column("workload_name", justify="left")
    unphased_table.add_column("num_queries", justify="left")
    unphased_table.add_column("max_arrival_rel_time_s", justify="left")

    for num_templates, num_qtpt, num_qpqt, lam in track(
        UNPHASED_COMBOS, description="Creating workloads..."
    ):
        workload = PoissonWorkloadCreator.create_poisson_workload(
            num_templates=num_templates,
            num_query_texts_per_template=num_qtpt,
            num_queries_per_query_text=num_qpqt,
            poisson_lambda=lam,
            seed=SEED,
            print_summary=False,
        )
        workloads.append(workload)
        max_rel_start_time_s = workload.rel_start_time_range()[1]
        unphased_table.add_row(
            "unphased",
            str(num_templates),
            str(num_qtpt),
            str(num_qpqt),
            str(lam),
            workload.workload_name,
            str(workload.num_queries),
            f"{_fmt(max_rel_start_time_s)}",
        )
    print(unphased_table)

    # Phased workloads.
    phased_table = Table(title=f"Phased Training workloads")
    phased_table.add_column("subset", justify="left")
    phased_table.add_column("num_templates", justify="right")
    phased_table.add_column("num_query_texts_per_template", justify="right")
    phased_table.add_column("total_num_queries", justify="right")
    phased_table.add_column("num_lull_burst_cycles", justify="right")
    phased_table.add_column("lull_poisson_lambda", justify="right")
    phased_table.add_column("burst_poisson_lambda", justify="right")
    phased_table.add_column("fraction_of_queries_in_bursts", justify="right")
    phased_table.add_column("workload_name", justify="left")
    phased_table.add_column("max_arrival_rel_time_s", justify="left")

    for (
        num_templates,
        num_qtpt,
        total_num_queries,
        num_lull_burst_cycles,
        lull_poisson_lambda,
        burst_poisson_lambda,
        fraction_of_queries_in_bursts,
    ) in track(PHASED_COMBOS, description="Creating workloads..."):
        phases = PoissonWorkloadCreator.make_bursty_profile(
            total_num_queries=total_num_queries,
            num_lull_burst_cycles=num_lull_burst_cycles,
            lull_poisson_lambda=lull_poisson_lambda,
            burst_poisson_lambda=burst_poisson_lambda,
            fraction_of_queries_in_bursts=fraction_of_queries_in_bursts,
        )
        workload = PoissonWorkloadCreator.create_poisson_workload_phased(
            num_templates=num_templates,
            num_query_texts_per_template=num_qtpt,
            num_queries_per_query_text=None,
            phases=phases,
            seed=SEED,
            print_summary=False,
        )
        workloads.append(workload)
        max_rel_start_time_s = workload.rel_start_time_range()[1]
        phased_table.add_row(
            "phased",
            str(num_templates),
            str(num_qtpt),
            str(total_num_queries),
            str(num_lull_burst_cycles),
            str(lull_poisson_lambda),
            str(burst_poisson_lambda),
            str(fraction_of_queries_in_bursts),
            workload.workload_name,
            f"{_fmt(max_rel_start_time_s)}",
        )
    print(phased_table)

    max_rel_times = [w.df["rel_start_time_s"].max() for w in workloads]
    summary_table = Table(title="Model Training Workloads Summary")
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Workloads created", str(len(workloads)))
    summary_table.add_row("Min highest rel_time_s", _fmt(min(max_rel_times)))
    summary_table.add_row("Max highest rel_time_s", _fmt(max(max_rel_times)))
    summary_table.add_row("Sum highest rel_time_s", _fmt(sum(max_rel_times)))
    print(summary_table)

    return workloads


def sequentially_execute_training_workloads(
    workload_configs: list[WorkloadConfig],
    execution_config_path: str | Path,
    params: list[str],
    force: bool = False,
) -> None:
    """Execute each workload config sequentially against live clusters."""
    exec_cfg_path = Path(execution_config_path)
    parsed_params = parse_params(params)
    runs_path = Path(get_runs_path())

    t_start = time.monotonic()
    total = len(workload_configs)
    for i, workload_config in enumerate(workload_configs, start=1):
        config_id = make_run_id([exec_cfg_path.stem], parsed_params)
        wid = workload_config.id()

        if not force:
            recent_run_id = find_most_recent_live_run_id(config_id, wid)
            if recent_run_id is not None:
                print(
                    f"[dim]Skipping '{workload_config.workload_name}' (up to date)[/]"
                )
                continue

        cfg = load_yaml_with_params(exec_cfg_path, parsed_params)
        cfg = copy_and_apply_overrides(
            cfg, {"workload_config": workload_config.to_dict()}
        )
        print(
            f"\n[bold cyan]── Workload {i}/{total}: '{workload_config.workload_name}' ──[/]"
        )
        runner = WorkloadRunner(cfg)
        append_to_run_log(
            run_id=runner.run_id,
            config_id=config_id,
            workload_id=wid,
        )
        asyncio.run(runner.run())

    elapsed = time.monotonic() - t_start
    timing_table = Table(title="Sequential Execution Timing")
    timing_table.add_column("Metric", style="bold cyan")
    timing_table.add_column("Value", justify="right")
    timing_table.add_row("Workloads executed", str(len(workload_configs)))
    timing_table.add_row("Total wall time", _fmt(elapsed))
    print(timing_table)


def print_run_status_table() -> None:
    """Print a table showing the most recent run_id per RPU size for every combo."""
    _RPU_RE = re.compile(r"RPU=(\d+)")

    # workload_id → combo tuple, preserving UNPHASED_COMBOS and PHASED_COMBOS order.
    wid_to_combo = {
        WorkloadConfig(
            workload_name=PoissonWorkloadCreator.name_from_params(
                num_templates=t,
                num_query_texts_per_template=q,
                num_queries_per_query_text=n,
                phases=[
                    PoissonArrivalPhase(
                        num_queries=t * q * n,
                        poisson_lambda=lam,
                    )
                ],
                seed=SEED,
            )
        ).id(): (t, q, n, lam)
        for t, q, n, lam in UNPHASED_COMBOS
    }
    wid_to_combo_phased = {
        WorkloadConfig(
            workload_name=PoissonWorkloadCreator.name_from_params(
                num_templates=t,
                num_query_texts_per_template=q,
                num_queries_per_query_text=None,
                phases=PoissonWorkloadCreator.make_bursty_profile(
                    total_num_queries=total,
                    num_lull_burst_cycles=num_cycles,
                    lull_poisson_lambda=lull_lam,
                    burst_poisson_lambda=burst_lam,
                    fraction_of_queries_in_bursts=frac_burst,
                ),
                seed=SEED,
            )
        ).id(): (t, q, total, num_cycles, lull_lam, burst_lam, frac_burst)
        for t, q, total, num_cycles, lull_lam, burst_lam, frac_burst in PHASED_COMBOS
    }

    # (workload_id, rpu) → best run_id
    best: dict[tuple[str, int], str] = {}
    log_path = os.path.join(get_runs_path(), "run_log.csv")
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                if not (wid := row.get("workload_id", "")) or (
                    wid not in wid_to_combo and wid not in wid_to_combo_phased
                ):
                    continue
                if not (m := _RPU_RE.search(row["config_id"])):
                    continue
                rpu, rid = int(m.group(1)), row["run_id"]
                key = (wid, rpu)
                if key not in best or int(rid) > int(best[key]):
                    best[key] = rid

    sorted_rpus = sorted({rpu for _, rpu in best})

    table = Table(title="Unphased Workloads Run Status")
    table.add_column("subset", justify="left")
    for col in (
        "num_templates",
        "num_qtpt",
        "num_qpqt",
        "poisson_lambda",
        "workload_id",
    ):
        table.add_column(col, justify="right")
    for rpu in sorted_rpus:
        table.add_column(f"RPU={rpu}", justify="left")

    for wid, (t, q, n, lam) in wid_to_combo.items():
        cells = [
            (
                f"[green]{best[(wid, rpu)]}[/]"
                if (wid, rpu) in best
                else "[dim]not run yet[/]"
            )
            for rpu in sorted_rpus
        ]
        table.add_row("unphased", str(t), str(q), str(n), str(lam), wid, *cells)

    print(table)

    phased_table = Table(title="Phased Workloads Run Status")
    phased_table.add_column("subset", justify="left")
    for col in (
        "num_templates",
        "num_qtpt",
        "total_num_queries",
        "num_lull_burst_cycles",
        "lull_poisson_lambda",
        "burst_poisson_lambda",
        "fraction_of_queries_in_bursts",
        "workload_id",
    ):
        phased_table.add_column(col, justify="right")
    for rpu in sorted_rpus:
        phased_table.add_column(f"RPU={rpu}", justify="left")

    for wid, (
        t,
        q,
        total,
        num_cycles,
        lull_lam,
        burst_lam,
        frac_burst,
    ) in wid_to_combo_phased.items():
        cells = [
            (
                f"[green]{best[(wid, rpu)]}[/]"
                if (wid, rpu) in best
                else "[dim]not run yet[/]"
            )
            for rpu in sorted_rpus
        ]
        phased_table.add_row(
            "phased",
            str(t),
            str(q),
            str(total),
            str(num_cycles),
            str(lull_lam),
            str(burst_lam),
            str(frac_burst),
            wid,
            *cells,
        )
    print(phased_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and/or execute model training workloads."
    )
    parser.add_argument(
        "--mode",
        choices=["create", "execute", "status", "all"],
        default="create",
        help=(
            "Whether to create workloads, execute them, print the run status "
            "table, or all of the above. "
        ),
    )
    parser.add_argument(
        "--execution_config",
        help=(
            "Path to the YAML execution config file. "
            "Required when --mode is 'execute' or 'all'."
        ),
    )
    parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Substitute <KEY> placeholder in the execution config with VALUE. "
            "May be repeated: --param TARGET_DATE=2024-05-27."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run workloads even if an up-to-date run already exists.",
    )
    args = parser.parse_args()

    # Create workloads if needed.
    if args.mode == "create" or args.mode == "all":
        create_training_workloads()

    # Execute workloads if needed.
    if args.mode == "execute" or args.mode == "all":
        if not args.execution_config:
            parser.error(
                "--execution_config is required when --mode is 'execute' or "
                "'all'."
            )
        unphased_workload_configs = [
            WorkloadConfig(
                workload_name=PoissonWorkloadCreator.name_from_params(
                    num_templates=num_templates,
                    num_query_texts_per_template=num_qtpt,
                    num_queries_per_query_text=num_qpqt,
                    phases=[
                        PoissonArrivalPhase(
                            num_queries=num_templates * num_qtpt * num_qpqt,
                            poisson_lambda=lam,
                        )
                    ],
                    seed=SEED,
                )
            )
            for num_templates, num_qtpt, num_qpqt, lam in UNPHASED_COMBOS
        ]
        sequentially_execute_training_workloads(
            workload_configs=unphased_workload_configs,
            execution_config_path=args.execution_config,
            params=args.param,
            force=args.force,
        )
        phased_workload_configs = [
            WorkloadConfig(
                workload_name=PoissonWorkloadCreator.name_from_params(
                    num_templates=num_templates,
                    num_query_texts_per_template=num_qtpt,
                    num_queries_per_query_text=None,
                    phases=PoissonWorkloadCreator.make_bursty_profile(
                        total_num_queries=total_num_queries,
                        num_lull_burst_cycles=num_lull_burst_cycles,
                        lull_poisson_lambda=lull_poisson_lambda,
                        burst_poisson_lambda=burst_poisson_lambda,
                        fraction_of_queries_in_bursts=fraction_of_queries_in_bursts,
                    ),
                    seed=SEED,
                )
            )
            for (
                num_templates,
                num_qtpt,
                total_num_queries,
                num_lull_burst_cycles,
                lull_poisson_lambda,
                burst_poisson_lambda,
                fraction_of_queries_in_bursts,
            ) in PHASED_COMBOS
        ]
        sequentially_execute_training_workloads(
            workload_configs=phased_workload_configs,
            execution_config_path=args.execution_config,
            params=args.param,
            force=args.force,
        )

    # Print status table if requested.
    if args.mode == "status" or args.mode == "all":
        print_run_status_table()
