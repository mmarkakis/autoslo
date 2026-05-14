import argparse
import asyncio
import itertools
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
from autoslo.filesystem.path_utils import append_to_run_log
from autoslo.filesystem.yaml_helpers import load_yaml_with_params
from autoslo.workload_definition.poisson_workload_creator import (
    PoissonWorkloadCreator,
)
from autoslo.workload_definition.workload import Workload
from autoslo.workload_execution.workload_runner import WorkloadRunner

NUM_TEMPLATES_OPTIONS = [66, 99]
NUM_QUERY_TEXTS_PER_TEMPLATE_OPTIONS = [1, 3]
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
) -> None:
    """Execute each workload config sequentially against live clusters."""
    exec_cfg_path = Path(execution_config_path)
    parsed_params = parse_params(params)

    t_start = time.monotonic()
    total = len(workload_configs)
    for i, workload_config in enumerate(workload_configs, start=1):
        cfg = load_yaml_with_params(exec_cfg_path, parsed_params)
        cfg = copy_and_apply_overrides(
            cfg, {"workload_config": workload_config.to_dict()}
        )
        config_id = make_run_id([exec_cfg_path.stem], parsed_params)
        print(
            f"\n[bold cyan]── Workload {i}/{total}: '{workload_config.workload_name}' ──[/]"
        )
        runner = WorkloadRunner(cfg)
        append_to_run_log(
            run_id=runner.run_id,
            config_id=config_id,
            workload_id=workload_config.id(),
        )
        asyncio.run(runner.run())

    elapsed = time.monotonic() - t_start
    timing_table = Table(title="Sequential Execution Timing")
    timing_table.add_column("Metric", style="bold cyan")
    timing_table.add_column("Value", justify="right")
    timing_table.add_row("Workloads executed", str(len(workload_configs)))
    timing_table.add_row("Total wall time", _fmt(elapsed))
    print(timing_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and/or execute model training workloads."
    )
    parser.add_argument(
        "--mode",
        choices=["create", "execute", "both"],
        default="create",
        help="Whether to create workloads, execute them, or both.",
    )
    parser.add_argument(
        "--execution_config",
        help=(
            "Path to the YAML execution config file. "
            "Required when --mode is 'execute' or 'both'."
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
    args = parser.parse_args()

    # Create workloads if needed.
    if args.mode == "create" or args.mode == "both":
        create_training_workloads()

    # Execute workloads if needed.
    if args.mode == "execute" or args.mode == "both":
        if not args.execution_config:
            parser.error(
                "--execution_config is required when --mode is 'execute' or "
                "'both'."
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
        )
