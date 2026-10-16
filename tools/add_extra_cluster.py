#!/usr/bin/env python3
"""
add_extra_cluster.py
---------------------
End-to-end pipeline to add one more redset cluster to the paper's
"extra clusters" generalization check (see
agent_planning/20260916_additional_redset_clusters_for_paper.md), given only
its cluster ID. 

1. Run Redbench matching against TPC-DS 1GB for the given cluster.
2. Copy the matched output into chunkbench's data/redset_executable_workloads/.
3. Convert it to executable format (workload.parquet, warmup, query_texts).
4. Generate the 5 cluster-specific config files (4 execution configs + 1
   tuner config), by substituting the workload name into a template
   cluster's existing configs.
5. Append the new cluster's entries into the tuning, execution, and plotting
   "main_eval_v8_extra_clusters.yml" manifests.

Each step is skipped (with a message) if its output already exists, so the
script is safe to re-run/resume.

Usage:
    python tools/add_extra_cluster.py 75
    python tools/add_extra_cluster.py 75 --template_cluster 184
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

REDBENCH_ROOT = Path("/home/markakis/Redbench")
CHUNKBENCH_ROOT = Path("/home/markakis/chunkbench")
REDBENCH_VENV_PYTHON = REDBENCH_ROOT / ".venv" / "bin" / "python"
CHUNKBENCH_VENV_PYTHON = Path("/home/markakis/markos-3.11.9-venv/bin/python")
REDSET_RAW_DIR = Path("/home/markakis/redset/provisioned/parts")

MATCHING_CONFIG = "src/redbench/matching/config/tpcds.json"

TUNING_MANIFEST = (
    CHUNKBENCH_ROOT / "data/manifests/tuning/main_eval_v8_extra_clusters.yml"
)
EXECUTION_MANIFEST = (
    CHUNKBENCH_ROOT / "data/manifests/execution/main_eval_v8_extra_clusters.yml"
)
PLOTTING_MANIFEST = (
    CHUNKBENCH_ROOT / "data/manifests/plotting/main_eval_v8_extra_clusters.yml"
)

HANDMADE_DIR = CHUNKBENCH_ROOT / "data/execution_configs/handmade"
TUNER_CONFIGS_DIR = CHUNKBENCH_ROOT / "data/tuner_configs"

CONFIG_TEMPLATES = [
    ("handmade", "main_eval_base_v8_c{id}.yml"),
    ("handmade", "main_eval_round_robin_v8_c{id}.yml"),
    ("handmade", "main_eval_round_robin_c{id}.yml"),
    ("handmade", "main_eval_rais_v8_c{id}.yml"),
    ("tuner", "forecasted_v8_c{id}.yml"),
]


def step_run_matching(cluster_id: int) -> None:
    raw_path = REDSET_RAW_DIR / f"{cluster_id}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw redset data found at {raw_path}")

    out_dir = (
        REDBENCH_ROOT
        / "output"
        / "generated_workloads"
        / "tpcds"
        / "provisioned"
        / f"cluster_{cluster_id}"
        / "database_0"
    )
    if list(out_dir.glob("matching_*/workload.csv")):
        print(f"[1/5] Redbench matching for cluster {cluster_id}: already done, skipping.")
        return

    print(f"[1/5] Running Redbench matching for cluster {cluster_id}...")
    subprocess.run(
        [
            str(REDBENCH_VENV_PYTHON),
            "src/redbench/run.py",
            "--redset_path",
            str(raw_path),
            "--output_dir",
            "output",
            "--generation_strategy",
            "matching",
            "--instance_id",
            str(cluster_id),
            "--database_id",
            "0",
            "--config_path_matching",
            MATCHING_CONFIG,
        ],
        cwd=REDBENCH_ROOT,
        check=False,  # Redbench's own (optional) plot step is known-flaky;
        # queries.json is written well before it, so don't hard-fail here.
    )
    if not list(out_dir.glob("matching_*/workload.csv")):
        raise RuntimeError(
            f"Redbench matching did not produce workload.csv under {out_dir} "
            f"(the subprocess may have been interrupted before assembly finished)."
        )


def step_copy_matched_output(cluster_id: int) -> None:
    src_parents = list(
        (
            REDBENCH_ROOT
            / "output/generated_workloads/tpcds/provisioned"
            / f"cluster_{cluster_id}/database_0"
        ).glob("matching_*")
    )
    if not src_parents:
        raise RuntimeError(f"No matching_* output found for cluster {cluster_id}")
    src_dir = src_parents[0]

    dest_dir = (
        CHUNKBENCH_ROOT
        / "data/redset_executable_workloads"
        / f"redbench_provisioned_{cluster_id}_0"
        / src_dir.name
    )
    if dest_dir.exists():
        print(f"[2/5] Matched output for cluster {cluster_id}: already copied, skipping.")
        return

    print(f"[2/5] Copying matched output for cluster {cluster_id}...")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", str(src_dir), str(dest_dir.parent)], check=True)


def step_convert_to_executable(cluster_id: int) -> None:
    workload_name = f"redbench_provisioned_{cluster_id}_0"
    out_parquet = CHUNKBENCH_ROOT / "data/workloads" / f"{workload_name}.parquet"
    if out_parquet.exists():
        print(f"[3/5] Executable workload for cluster {cluster_id}: already exists, skipping.")
        return

    print(f"[3/5] Converting cluster {cluster_id} to executable format...")
    subprocess.run(
        [
            str(CHUNKBENCH_VENV_PYTHON),
            "src/autoslo/workload_definition/redbench_to_executable.py",
            "--workload_name",
            workload_name,
        ],
        cwd=CHUNKBENCH_ROOT,
        check=True,
    )


def step_generate_configs(cluster_id: int, template_cluster: int) -> None:
    print(f"[4/5] Generating cluster-specific configs for cluster {cluster_id}...")
    old_name = f"redbench_provisioned_{template_cluster}_0"
    new_name = f"redbench_provisioned_{cluster_id}_0"

    for kind, pattern in CONFIG_TEMPLATES:
        template_filename = pattern.format(id=template_cluster)
        out_filename = pattern.format(id=cluster_id)
        src_dir = HANDMADE_DIR if kind == "handmade" else TUNER_CONFIGS_DIR
        template_path = src_dir / template_filename
        out_path = src_dir / out_filename

        if out_path.exists():
            print(f"  {out_path.name}: already exists, skipping.")
            continue
        if not template_path.exists():
            raise FileNotFoundError(f"Template config not found: {template_path}")

        content = template_path.read_text()
        out_path.write_text(content.replace(old_name, new_name))
        print(f"  wrote {out_path.relative_to(CHUNKBENCH_ROOT)}")


def _existing_cluster_ids(manifest_text: str) -> list[int]:
    return sorted(
        {int(m) for m in re.findall(r"_c(\d+)\b", manifest_text)}
    )


def update_tuning_manifest(cluster_id: int) -> None:
    text = TUNING_MANIFEST.read_text()
    if cluster_id in _existing_cluster_ids(text):
        print(f"  tuning manifest: cluster {cluster_id} already present, skipping.")
        return

    locals_anchor = "    may27_default_exec_params: &may27_default_exec_params"
    new_locals = (
        f"    default_configs_c{cluster_id}: &default_configs_c{cluster_id}\n"
        f'        initial_execution_config: "exec:handmade/main_eval_base_v8_c{cluster_id}"\n'
        f'        tuner_config: "tuner:forecasted_v8_c{cluster_id}"\n'
    )
    assert text.count(locals_anchor) == 1
    text = text.replace(locals_anchor, new_locals + locals_anchor)

    new_entries = (
        f"\n    # Cluster {cluster_id}, May 27, kappa = 4, threshold = 0.10\n"
        f"    may27_1month_k4_thresh10_nocache_v8_c{cluster_id}:\n"
        f"        configs: *default_configs_c{cluster_id}\n"
        f"        params:\n"
        f"            <<: [*may27_default_exec_params, *default_may27_sampling_tuner_params]\n"
        f"\n    # Cluster {cluster_id}, May 27, kappa = 2, threshold = 0.10\n"
        f"    may27_1month_k2_thresh10_nocache_v8_c{cluster_id}:\n"
        f"        configs: *default_configs_c{cluster_id}\n"
        f"        params:\n"
        f"            <<: [*may27_default_exec_params, *default_may27_sampling_tuner_params]\n"
        f"            KAPPA: 2\n"
    )
    text = text.rstrip("\n") + "\n" + new_entries
    TUNING_MANIFEST.write_text(text)
    print(f"  updated {TUNING_MANIFEST.relative_to(CHUNKBENCH_ROOT)}")


def update_execution_manifest(cluster_id: int) -> None:
    text = EXECUTION_MANIFEST.read_text()
    if cluster_id in _existing_cluster_ids(text):
        print(f"  execution manifest: cluster {cluster_id} already present, skipping.")
        return

    new_locals = (
        f"  may27_workload_config_c{cluster_id}: &may27_workload_config_c{cluster_id}\n"
        f"    workload_name: redbench_provisioned_{cluster_id}_0\n"
        f'    target_date: "2024-05-27"\n'
        f"    rescale_factor: 0.16666667\n"
    )
    anchor = "\nmain_content:\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "\n" + new_locals + anchor)

    c = cluster_id
    new_entries = f"""
  ## Cluster {c}, May 27

  # Baselines (kappa-independent; reused when plotting both kappa=4 and kappa=2)
  - workload_config: *may27_workload_config_c{c}
    sim_execution_config: exec:handmade/main_eval_round_robin_v8_c{c}.yml
    live_execution_config: exec:handmade/main_eval_round_robin_c{c}.yml
    params:
      TARGET_DATE: "2024-05-27"
      INITIAL_RPU: [64]
  - workload_config: *may27_workload_config_c{c}
    sim_execution_config: exec:handmade/main_eval_round_robin_v8_c{c}.yml
    live_execution_config: exec:handmade/main_eval_round_robin_c{c}.yml
    params:
      TARGET_DATE: "2024-05-27"
      INITIAL_RPU: [32, 32]
  - workload_config: *may27_workload_config_c{c}
    execution_config: exec:handmade/main_eval_rais_v8_c{c}.yml
    params:
      TARGET_DATE: "2024-05-27"
      RAIS_LEVEL: 50
      RAIS_MAX_CAPACITY: 128
  - workload_config: *may27_workload_config_c{c}
    execution_config: exec:handmade/main_eval_rais_v8_c{c}.yml
    params:
      TARGET_DATE: "2024-05-27"
      RAIS_LEVEL: 100
      RAIS_MAX_CAPACITY: 128

  # AutoSLO (tuned by main_eval_v8_extra_clusters.yml tuning manifest)
  - workload_config: *may27_workload_config_c{c}
    execution_config: exec:tuned/may27_1month_k4_thresh10_nocache_v8_c{c}.yml
  - workload_config: *may27_workload_config_c{c}
    execution_config: exec:tuned/may27_1month_k2_thresh10_nocache_v8_c{c}.yml
"""
    text = text.rstrip("\n") + "\n" + new_entries.lstrip("\n")
    EXECUTION_MANIFEST.write_text(text)
    print(f"  updated {EXECUTION_MANIFEST.relative_to(CHUNKBENCH_ROOT)}")


def update_plotting_manifest(cluster_id: int) -> None:
    text = PLOTTING_MANIFEST.read_text()
    if cluster_id in _existing_cluster_ids(text):
        print(f"  plotting manifest: cluster {cluster_id} already present, skipping.")
        return

    existing_cluster_count = len(
        set(re.findall(r"&may27_workload_config_c(\d+)", text))
    )
    new_row = existing_cluster_count
    new_rows_total = existing_cluster_count + 1

    text = re.sub(r"rows: \d+", f"rows: {new_rows_total}", text, count=1)
    text = re.sub(
        r"figsize: \[8, \d+\]", f"figsize: [8, {new_rows_total * 3}]", text, count=1
    )

    c = cluster_id
    new_locals = (
        f"  may27_workload_config_c{c}: &may27_workload_config_c{c}\n"
        f"    workload_name: redbench_provisioned_{c}_0\n"
        f'    target_date: "2024-05-27"\n'
        f"    rescale_factor: 0.16666667\n"
    )
    anchor = "\nmain_content:\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "\n" + new_locals + anchor)

    def panel(col: int, kappa: int) -> str:
        return f"""
    - row: {new_row}
      col: {col}
      title: "Cluster {c}, May 27, \u03ba={kappa}"
      show_legend: false
      slo_objective_config:
        slo_metric: binary
        slo_threshold: 0
      slo_resolver_config:
        slo_s: 0.0
        slo_dict_filename: "1778469201294_p100_k{kappa}.yml"
      threshold_lines:
        - value: 0.10
          color: light_green
      points:
        - label: "Single 64"
          formatting_id: main_eval_64
          workload_config: *may27_workload_config_c{c}
          sim_execution_config: exec:handmade/main_eval_round_robin_v8_c{c}.yml
          live_execution_config: exec:handmade/main_eval_round_robin_c{c}.yml
          params:
            TARGET_DATE: "2024-05-27"
            INITIAL_RPU: [64]
        - label: "Round Robin 32+32"
          formatting_id: main_eval_32_32
          workload_config: *may27_workload_config_c{c}
          sim_execution_config: exec:handmade/main_eval_round_robin_v8_c{c}.yml
          live_execution_config: exec:handmade/main_eval_round_robin_c{c}.yml
          params:
            TARGET_DATE: "2024-05-27"
            INITIAL_RPU: [32, 32]
        - label: "RAIS - Balanced"
          formatting_id: main_eval_rais50
          workload_config: *may27_workload_config_c{c}
          sim_execution_config: exec:handmade/main_eval_rais_v8_c{c}.yml
          live_execution_config: exec:handmade/main_eval_rais_v8_c{c}.yml
          params:
            TARGET_DATE: "2024-05-27"
            RAIS_LEVEL: 50
            RAIS_MAX_CAPACITY: 128
        - label: "RAIS - High Perf."
          formatting_id: main_eval_rais100
          workload_config: *may27_workload_config_c{c}
          sim_execution_config: exec:handmade/main_eval_rais_v8_c{c}.yml
          live_execution_config: exec:handmade/main_eval_rais_v8_c{c}.yml
          params:
            TARGET_DATE: "2024-05-27"
            RAIS_LEVEL: 100
            RAIS_MAX_CAPACITY: 128
        - label: "AutoSLO"
          formatting_id: main_eval_chosen
          workload_config: *may27_workload_config_c{c}
          execution_config: exec:tuned/may27_1month_k{kappa}_thresh10_nocache_v8_c{c}.yml
"""

    new_panels = panel(0, 4) + panel(1, 2)
    text = text.rstrip("\n") + "\n" + new_panels.lstrip("\n")
    PLOTTING_MANIFEST.write_text(text)
    print(f"  updated {PLOTTING_MANIFEST.relative_to(CHUNKBENCH_ROOT)}")


def step_update_manifests(cluster_id: int) -> None:
    print(f"[5/5] Updating tuning/execution/plotting manifests for cluster {cluster_id}...")
    update_tuning_manifest(cluster_id)
    update_execution_manifest(cluster_id)
    update_plotting_manifest(cluster_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add a new redset cluster to the paper's extra-clusters "
            "generalization check: Redbench matching -> conversion -> "
            "cluster-specific configs -> manifest updates."
        )
    )
    parser.add_argument("cluster_id", type=int, help="Redset provisioned cluster ID.")
    parser.add_argument(
        "--template_cluster",
        type=int,
        default=184,
        help="Existing cluster whose configs to copy as templates (default: 184).",
    )
    args = parser.parse_args()

    step_run_matching(args.cluster_id)
    step_copy_matched_output(args.cluster_id)
    step_convert_to_executable(args.cluster_id)
    step_generate_configs(args.cluster_id, args.template_cluster)
    step_update_manifests(args.cluster_id)

    print(f"\nDone. Cluster {args.cluster_id} is wired into "
          f"main_eval_v8_extra_clusters.yml (tuning/execution/plotting).")


if __name__ == "__main__":
    main()
