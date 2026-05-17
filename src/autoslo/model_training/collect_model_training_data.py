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
    is_up_to_date,
)
from autoslo.filesystem.yaml_helpers import load_yaml_with_params
from autoslo.workload_definition.poisson_workload_creator import (
    PoissonWorkloadCreator,
)
from autoslo.workload_definition.workload import Workload
from autoslo.workload_execution.workload_runner import WorkloadRunner

NUM_TEMPLATES_OPTIONS = [66, 99]
NUM_QUERY_TEXTS_PER_TEMPLATE_OPTIONS = [2, 3]
NUM_QUERIES_PER_QUERY_TEXT_OPTIONS = [1]
POISSON_LAMBDA_OPTIONS = [0.1, 0.05]
SEED = 42

_COMBOS = list(
    itertools.product(
        NUM_TEMPLATES_OPTIONS,
        NUM_QUERY_TEXTS_PER_TEMPLATE_OPTIONS,
        NUM_QUERIES_PER_QUERY_TEXT_OPTIONS,
        POISSON_LAMBDA_OPTIONS,
    )
)


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
    cross_product_table = Table(
        title=f"Cross-product ({len(_COMBOS)} workloads)"
    )
    cross_product_table.add_column("num_templates", justify="right")
    cross_product_table.add_column(
        "num_query_texts_per_template", justify="right"
    )
    cross_product_table.add_column(
        "num_queries_per_query_text", justify="right"
    )
    cross_product_table.add_column("poisson_lambda", justify="right")
    for num_templates, num_qtpt, num_qpqt, lam in _COMBOS:
        cross_product_table.add_row(
            str(num_templates), str(num_qtpt), str(num_qpqt), str(lam)
        )
    print(cross_product_table)

    workloads = []
    for num_templates, num_qtpt, num_qpqt, lam in track(
        _COMBOS, description="Creating workloads..."
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
            if recent_run_id is not None and is_up_to_date(
                runs_path / recent_run_id / "execution_config.yml",
                exec_cfg_path,
            ):
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

    # workload_id → combo tuple, preserving _COMBOS order.
    wid_to_combo = {
        WorkloadConfig(
            workload_name=PoissonWorkloadCreator.name_from_params(
                num_templates=t,
                num_query_texts_per_template=q,
                num_queries_per_query_text=n,
                poisson_lambda=lam,
                seed=SEED,
            )
        ).id(): (t, q, n, lam)
        for t, q, n, lam in _COMBOS
    }

    # (workload_id, rpu) → best run_id
    best: dict[tuple[str, int], str] = {}
    log_path = os.path.join(get_runs_path(), "run_log.csv")
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                if (
                    not (wid := row.get("workload_id", ""))
                    or wid not in wid_to_combo
                ):
                    continue
                if not (m := _RPU_RE.search(row["config_id"])):
                    continue
                rpu, rid = int(m.group(1)), row["run_id"]
                key = (wid, rpu)
                if key not in best or int(rid) > int(best[key]):
                    best[key] = rid

    sorted_rpus = sorted({rpu for _, rpu in best})

    table = Table(title="Model Training Run Status")
    for col in ("num_templates", "num_qtpt", "num_qpqt", "poisson_lambda"):
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
        table.add_row(str(t), str(q), str(n), str(lam), *cells)

    print(table)


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
        workload_configs = [
            WorkloadConfig(
                workload_name=PoissonWorkloadCreator.name_from_params(
                    num_templates=num_templates,
                    num_query_texts_per_template=num_qtpt,
                    num_queries_per_query_text=num_qpqt,
                    poisson_lambda=lam,
                    seed=SEED,
                )
            )
            for num_templates, num_qtpt, num_qpqt, lam in _COMBOS
        ]
        sequentially_execute_training_workloads(
            workload_configs=workload_configs,
            execution_config_path=args.execution_config,
            params=args.param,
            force=args.force,
        )

    # Print status table if requested.
    if args.mode == "status" or args.mode == "all":
        print_run_status_table()
