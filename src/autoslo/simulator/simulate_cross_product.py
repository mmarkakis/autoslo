from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.config.component_configs import WorkloadConfig
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator

logger = logging.getLogger(__name__)
console = Console()


def evaluate(evaluation_spec_name: str) -> None:
    """Evaluate baseline, tuned, and static-baseline configs on real data.

    All evaluations (baseline, tuned, and static baselines) are submitted
    to a single process pool via :meth:`evaluate_batch` for maximum
    parallelism.
    """

    # --- Load evaluation spec --------------------------------------
    if not evaluation_spec_name.endswith(".yml"):
        evaluation_spec_name += ".yml"
    spec_path = os.path.join(
        pu.get_data_path(), "cross_product_specs", evaluation_spec_name
    )
    if not os.path.exists(spec_path):
        raise ValueError(
            f"Evaluation spec {evaluation_spec_name} not found at {spec_path}."
        )
    spec = load_yaml(spec_path)
    workload_configs = [
        WorkloadConfig.from_config(cfg) for cfg in spec["workload_configs"]
    ]
    configs = []
    config_labels = []
    for config_path in spec["config_paths"]:
        full_config_path = os.path.join(
            pu.get_data_path(), "configs", config_path
        )
        config = load_yaml(full_config_path)
        configs.append(config)
        config_labels.append(Path(config_path).stem)

    # --- Set up output directory --------------------------------------
    out_dir = os.path.join(pu.get_data_path(), "simulator_runs")
    os.makedirs(out_dir, exist_ok=True)

    # --- Run evaluation.
    evaluator = ScenarioEvaluator()
    evaluator.evaluate_batch_from_configs(
        progress_bar_label="target",
        out_dir=out_dir,
        workload_configs=workload_configs,
        configs=configs,
        config_labels=config_labels,
        nest_outputs_by_config=False,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "evaluation_spec_name",
        type=str,
        help="Name of the evaluation spec to run, e.g. 'observation_period'.",
        required=True,
    )
    args = parser.parse_args()
    evaluate(args.evaluation_spec_name)
