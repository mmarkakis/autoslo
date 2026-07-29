"""
Compare performance of two or more trained IconqModels using a YAML manifest.

The manifest file specifies which models to compare and controls plot settings.
See the Manifest format section below for the full schema.

Produces two figures saved to the output directory:

  qerror_by_split.png
      Scatter plot.  X = model, columns within each group = split
      (train / val / test).  Markers at p50 / p90 / p95 Q-error.

  factor_error_by_rpu.png
      Scatter plot.  X = model, columns within each group = RPU value
      (test set only).  Markers at p5 / p10 / p50 / p90 / p95 factor
      error on a log scale.

  overprediction_fraction_vs_concurrency.png
      Grouped scatter.  One subplot per RPU value (test set only),
      subplots share the Y axis (0-100%).  X groups = concurrency bins
      (0 | 1-31 | 32-63 | 64+); one dot per model per bin showing the
      fraction of queries where the model overpredicted.  Dashed line
      at 50% marks the unbiased baseline.

Manifest format
---------------
  # reference_split_model_id, output_dir and min_rpu_samples are optional.
  output_dir: data/plots/iconq_comparison
  min_rpu_samples: 30
  # When set, every model's split DataFrames are filtered to only the
  # (cluster_name, query_id) pairs present in this model's corresponding
  # split, ensuring all percentile statistics are computed over identical
  # query sets.
  reference_split_model_id: model_v7

  models:
    - iconq_model_id: model_v7
      label: "v7 baseline"      # shown on X-axis tick
      annotate: false            # annotate every point for this model
    - iconq_model_id: model_v8
      label: "v8 new feature"
      annotate: true

Usage
-----
  python tools/compare_iconq_models.py MANIFEST.yml [options]

Examples
--------
  python tools/compare_iconq_models.py data/plots/iconq_comparison/v7_v8.yml
  python tools/compare_iconq_models.py data/plots/iconq_comparison/v7_v8.yml \\
      --output-dir data/plots/my_run --show
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from autoslo.config.component_configs import WorkloadConfig
from autoslo.models.iconq_model import DataSplit, IconqModel
from autoslo.visualizations.colors import Palette
from autoslo.visualizations.iconq_model_comparison import (
    ModelEntry,
    _DEFAULT_OUTPUT_DIR,
    _MIN_RPU_SAMPLES_DEFAULT,
    plot_factor_error_by_rpu,
    plot_factor_error_vs_concurrency,
    plot_inference_time_by_arrival,
    plot_qerror_by_split,
)
from autoslo.workload_definition.workload import Workload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare IconqModel performance using a YAML manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "manifest",
        metavar="MANIFEST.yml",
        help="Path to the YAML plotting manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Override the output directory specified in the manifest "
            f"(manifest default: {_DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--min-rpu-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override the minimum test-set observations required to include an RPU "
            f"value (manifest default: {_MIN_RPU_SAMPLES_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


def _load_manifest(
    path: str,
) -> tuple[
    list[ModelEntry],
    str,
    int,
    bool,
    bool,
    bool,
    WorkloadConfig | None,
    int,
    int | None,
    str | None,
]:
    """Parse *path* and return
    ``(models, output_dir, min_rpu_samples, highlight_best, annotate_best,
    show_titles, workload_config, inference_rpu, max_arrivals,
    reference_split_model_id)``."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "models" not in raw or not raw["models"]:
        print(
            "error: manifest must contain a non-empty 'models' list.",
            file=sys.stderr,
        )
        sys.exit(1)

    models: list[ModelEntry] = []
    for entry in raw["models"]:
        if "iconq_model_id" not in entry:
            print(
                "error: each model entry must have an 'iconq_model_id' field.",
                file=sys.stderr,
            )
            sys.exit(1)
        model_id = str(entry["iconq_model_id"])
        label = str(entry.get("label", model_id))
        annotate = bool(entry.get("annotate", False))
        color: str | None = entry.get("color") or None
        if color is not None and not hasattr(Palette, color):
            print(
                f"error: model '{model_id}' has unknown color '{color}'. "
                f"Must be a Palette attribute name.",
                file=sys.stderr,
            )
            sys.exit(1)
        models.append(ModelEntry(model_id=model_id, label=label, annotate=annotate, color=color))

    output_dir = str(raw.get("output_dir", _DEFAULT_OUTPUT_DIR))
    min_rpu_samples = int(raw.get("min_rpu_samples", _MIN_RPU_SAMPLES_DEFAULT))
    highlight_best = bool(raw.get("highlight_best", True))
    annotate_best = bool(raw.get("annotate_best", True))
    show_titles = bool(raw.get("show_titles", True))
    # Optional inference-timing section.
    workload_config: WorkloadConfig | None = None
    if "workload_config" in raw:
        workload_config = WorkloadConfig.from_config(raw)
    inference_rpu: int = int(raw.get("inference_rpu", 8))
    raw_max = raw.get("max_arrivals")
    max_arrivals: int | None = int(raw_max) if raw_max is not None else None
    raw_ref = raw.get("reference_split_model_id")
    reference_split_model_id: str | None = str(raw_ref) if raw_ref is not None else None

    return (
        models,
        output_dir,
        min_rpu_samples,
        highlight_best,
        annotate_best,
        show_titles,
        workload_config,
        inference_rpu,
        max_arrivals,
        reference_split_model_id,
    )


def main() -> None:
    args = _parse_args()

    (
        models,
        output_dir,
        min_rpu_samples,
        highlight_best,
        annotate_best,
        show_titles,
        workload_config,
        inference_rpu,
        max_arrivals,
        reference_split_model_id,
    ) = _load_manifest(args.manifest)

    # CLI flags override manifest values when provided.
    if args.output_dir is not None:
        output_dir = args.output_dir
    if args.min_rpu_samples is not None:
        min_rpu_samples = args.min_rpu_samples

    if len(models) < 2:
        print(
            "error: manifest must list at least two models to compare.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load split DataFrames for every model up front so both plots share the
    # same in-memory data without double-loading.
    all_split_dfs = {}
    for m in models:
        print(f"Loading {m.model_id} ...")
        all_split_dfs[m.model_id] = IconqModel.optimized_load_final_dfs_per_split(
            m.model_id
        )

    # If a reference split model is specified, reassign every model's rows to
    # the split that the reference model placed each (cluster_name, query_id)
    # in.  Rows not present in the reference model's splits are dropped.  This
    # guarantees all statistics are computed over exactly the same observations
    # regardless of any differences in each model's own train/val/test split.
    if reference_split_model_id is not None:
        if reference_split_model_id not in all_split_dfs:
            print(
                f"Loading reference split model {reference_split_model_id!r} "
                "for split remapping ..."
            )
            ref_split_dfs = IconqModel.optimized_load_final_dfs_per_split(
                reference_split_model_id
            )
        else:
            ref_split_dfs = all_split_dfs[reference_split_model_id]
        print(
            f"Remapping all models to {reference_split_model_id!r} "
            "split assignments ..."
        )

        # Determine composite join key (prefer cluster_name + query_id).
        _sample = next(iter(ref_split_dfs.values()))
        _key_cols = [
            c for c in ("cluster_name", "query_id") if c in _sample.columns
        ]
        _sep = "\x00"

        # Build mapping: composite key string -> DataSplit enum value.
        _ref_key_to_split: dict[str, DataSplit] = {}
        for _split, _ref_df in ref_split_dfs.items():
            if len(_key_cols) == 2:
                _keys = (
                    _ref_df[_key_cols[0]].astype(str)
                    + _sep
                    + _ref_df[_key_cols[1]].astype(str)
                )
            else:
                _keys = _ref_df[_key_cols[0]].astype(str)
            for _k in _keys:
                _ref_key_to_split[_k] = _split

        # For each model: concatenate all splits, assign the reference split
        # label, then re-partition.
        for model_id in list(all_split_dfs.keys()):
            _combined = pd.concat(
                list(all_split_dfs[model_id].values()),
                ignore_index=True,
            )
            if len(_key_cols) == 2:
                _join_keys = (
                    _combined[_key_cols[0]].astype(str)
                    + _sep
                    + _combined[_key_cols[1]].astype(str)
                )
            else:
                _join_keys = _combined[_key_cols[0]].astype(str)
            _combined = _combined.copy()
            _combined["_ref_split"] = _join_keys.map(_ref_key_to_split)
            _combined = _combined.dropna(subset=["_ref_split"])
            new_split_dfs = {}
            for _split in DataSplit:
                _mask = _combined["_ref_split"] == _split
                new_split_dfs[_split] = (
                    _combined[_mask]
                    .drop(columns=["_ref_split"])
                    .reset_index(drop=True)
                )
            all_split_dfs[model_id] = new_split_dfs

    # ── Figure 1: Q-error by split ───────────────────────────────────────────
    print("\nPlotting Q-error by split ...")
    fig1, path1 = plot_qerror_by_split(
        models,
        all_split_dfs,
        output_dir=output_dir,
        highlight_best=highlight_best,
        annotate_best=annotate_best,
        show_title=show_titles,
    )
    print(f"  saved: {path1}")
    if not args.show:
        plt.close(fig1)

    # ── Figure 2: Factor error by RPU (test set) ─────────────────────────────
    print("\nPlotting factor error by RPU (test set) ...")
    try:
        fig2, path2 = plot_factor_error_by_rpu(
            models,
            all_split_dfs,
            output_dir=output_dir,
            min_rpu_samples=min_rpu_samples,
            highlight_best=highlight_best,
            annotate_best=annotate_best,
            show_title=show_titles,
        )
        print(f"  saved: {path2}")
        if not args.show:
            plt.close(fig2)
    except ValueError as exc:
        print(f"  warning: RPU chart skipped — {exc}", file=sys.stderr)

    # ── Figure 3: Factor error vs. concurrency (test set) ─────────────────────
    print("\nPlotting factor error vs. concurrency (test set) ...")
    try:
        fig3, path3 = plot_factor_error_vs_concurrency(
            models,
            all_split_dfs,
            output_dir=output_dir,
            min_rpu_samples=min_rpu_samples,
            highlight_best=highlight_best,
            annotate_best=annotate_best,
            show_title=show_titles,
        )
        print(f"  saved: {path3}")
        if not args.show:
            plt.close(fig3)
    except ValueError as exc:
        print(f"  warning: concurrency chart skipped — {exc}", file=sys.stderr)

    if args.show:
        plt.show()

    # ── Figure 4: Inference time by arrival (optional) ────────────────────────
    if workload_config is not None:
        print("\nLoading workload for inference timing ...")
        workload = Workload(workload_config)
        print(
            f"Plotting inference time by arrival "
            f"(RPU={inference_rpu}, max_arrivals={max_arrivals}) ..."
        )
        fig4, path4 = plot_inference_time_by_arrival(
            models,
            workload=workload,
            rpu=inference_rpu,
            output_dir=output_dir,
            max_arrivals=max_arrivals,
            show_title=show_titles
        )
        print(f"  saved: {path4}")
        if args.show:
            plt.show()
        else:
            plt.close(fig4)


if __name__ == "__main__":
    main()
