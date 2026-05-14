import argparse
import os
from datetime import datetime

import yaml

import autoslo.filesystem.path_utils as pu
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.cache_model import CacheModel
from autoslo.models.iconq_model import (
    IconqModel,
    IconqModelInitConfig,
    NNModelTrainConfig,
)
from autoslo.model_training.iconq_model_trainer import iconq_model_trainer
from autoslo.models.stage_model import StageModel
from autoslo.models.xgboost_model import XGBoostModel


def find_run_ids() -> tuple[list[str], dict[str, list[str]]]:
    """Finds all relevant run IDs for training IconQ models."""

    df = pu.RunLocator.get_runs_df()
    pct_heavy_options = [0, 10, 25, 50]
    mean_interarrival_options = [10, 30, 60, 120]
    rpus = [4, 8, 16, 32]

    run_ids = []

    val_index_combinations = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 0),
        (3, 0, 1),
        (0, 2, 1),
        (1, 3, 2),
        (2, 0, 3),
        (3, 1, 0),
    ]
    test_idx_combinations = [
        (0, 1, 1),
        (1, 2, 2),
        (2, 3, 3),
        (3, 0, 0),
        (0, 2, 3),
        (1, 3, 0),
        (2, 0, 1),
        (3, 1, 2),
        (0, 3, 2),
        (1, 0, 3),
        (2, 1, 0),
        (3, 2, 1),
    ]
    train_val_test_assignments: dict[str, list[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Most recent chunk runs.
    for i, pct_heavy in enumerate(pct_heavy_options):
        for j, mean_interarrival in enumerate(mean_interarrival_options):
            for k, rpu in enumerate(rpus):
                this_workload_run_ids = pu.RunLocator.get_run_ids(
                    schema_name="ext_tpcds1000",
                    workload_name=f"tpcds_99templates_{pct_heavy:02d}pctheavy_{mean_interarrival}meaninterarrivals",
                    blueprint_name=f"single_{rpu}",
                )
                if len(this_workload_run_ids) > 0:
                    run_ids.append(max(this_workload_run_ids))
                    if (i, j, k) in val_index_combinations:
                        train_val_test_assignments["val"].append(
                            max(this_workload_run_ids)
                        )
                    elif (i, j, k) in test_idx_combinations:
                        train_val_test_assignments["test"].append(
                            max(this_workload_run_ids)
                        )
                    else:
                        train_val_test_assignments["train"].append(
                            max(this_workload_run_ids)
                        )

    # Benchmarking workload runs.
    # benchmark_runs = df[
    #     df["workload_name"] == "benchmarking_workload_99_3_3_shuffled_42"
    # ]
    # run_ids.extend(benchmark_runs["run_id"].to_list())
    return run_ids, train_val_test_assignments


def set_up_featurizer(run_ids: list[str], from_sys_query_explain: bool) -> str:
    """Sets up the IconQ query featurizer and returns its ID."""

    featurizer = IconqQueryFeaturizer(
        schema_name="tpcds1000",
        run_ids=run_ids,
        from_sys_query_explain=from_sys_query_explain,
    )
    return featurizer.save()


def train_cache_model(
    run_ids: list[str],
    only_non_overlapping_queries: bool,
    ignore_cluster_size: bool = False,
) -> str:
    """Trains a CacheModel and returns its ID."""

    cache_model = CacheModel(
        enable_template_cache=False,
        best_effort=False,
        ignore_cluster_size=ignore_cluster_size,
    )
    cache_model.train(
        run_ids=run_ids,
        from_scratch=True,
        only_non_overlapping_queries=only_non_overlapping_queries,
    )
    cache_model_id = cache_model.save()
    return cache_model_id


def train_xgboost_model(
    iconq_query_featurizer_id: str,
    run_ids: list[str],
    only_non_overlapping_queries: bool,
    ignore_cluster_size: bool = False,
) -> str:
    """Trains an XGBoostModel and returns its ID."""
    xgboost_model = XGBoostModel(
        train_on_log_runtime=True,
        n_estimators=100,
        iconq_query_featurizer_id=iconq_query_featurizer_id,
        ignore_cluster_size=ignore_cluster_size,
    )
    xgboost_model.train(
        run_ids=run_ids,
        only_non_overlapping_queries=only_non_overlapping_queries,
    )
    xgboost_model_id = xgboost_model.save()
    return xgboost_model_id


def initialize_stage_model(
    cache_model_id: str, xgboost_model_id: str
) -> StageModel:
    """Initializes a StageModel."""
    stage_model = StageModel(
        cache_model_id=cache_model_id, xgboost_model_id=xgboost_model_id
    )
    return stage_model


def train_iconq_model(
    iconq_query_featurizer_id: str,
    stage_model_id: str,
    run_ids: list[str],
    use_stage_for_isolated_queries: bool,
    explicit_run_ids_per_split: dict[str, list[str]] | None = None,
    penalize_based_on_overlap: bool = False,
    sensitive_q_error_loss_version: int = 1,
    ignore_cluster_size: bool = False,
) -> str:
    """Trains an IconqModel and returns its ID."""
    iconq_model_init_config = IconqModelInitConfig(
        iconq_query_featurizer_id=iconq_query_featurizer_id,
        stage_model_id=stage_model_id,
        is_bayesian=False,
        is_mdn=False,
        train_on_log_runtime=True,
        use_fixed_window_radius_s=None,
        use_fixed_window_max_neighbors_per_side=None,
        ignore_cluster_size=ignore_cluster_size,
    )
    nn_model_train_config = NNModelTrainConfig(
        run_ids=run_ids,
        use_stage_for_isolated_queries=use_stage_for_isolated_queries,
        explicit_run_ids_per_split=explicit_run_ids_per_split,
        sensitive_q_error_loss_small_val=5.0,
        penalize_based_on_overlap=penalize_based_on_overlap,
        learning_rate=1e-3,
        sensitive_q_error_loss_version=sensitive_q_error_loss_version,
    )
    iconq_model = IconqModel(
        init_config=iconq_model_init_config,
    )
    iconq_model_trainer(
        iconq_model=iconq_model,
        train_config=nn_model_train_config,
    )
    iconq_model_id = iconq_model.save()
    return iconq_model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains an IconqModel step-by-step, caching progress."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to force re-training even if cached progress exists.",
    )
    parser.add_argument(
        "--only_non_overlapping_queries",
        action="store_true",
        help="Whether to only train the cache and XGBoost models on queries that do not overlap with any other queries in the trace.",
    )
    parser.add_argument(
        "--use_stage_for_isolated_queries",
        action="store_true",
        help="Whether to use the StageModel to generate predictions for isolated queries when training the IconQ model.",
    )
    parser.add_argument(
        "--use_explicit_run_ids",
        action="store_true",
        help="If set, use explicitly provided run IDs for each data splti for the Iconq model.",
    )
    parser.add_argument(
        "--penalize_based_on_overlap",
        action="store_true",
        help="Whether to penalize based on overlap in the sensitive Q-error loss.",
    )
    parser.add_argument(
        "--from_sys_query_explain",
        action="store_true",
        help="Whether to derive features from sys_query_explain table.",
    )
    parser.add_argument(
        "--sensitive_q_error_loss_version",
        type=int,
        default=1,
        help="The version of the sensitive Q-error loss to use (1 or 2).",
    )
    parser.add_argument(
        "--ignore_cluster_size",
        action="store_true",
        help="Whether to ignore the cluster size when training models.",
    )
    args = parser.parse_args()
    print(
        f"Value of only_non_overlapping_queries: {args.only_non_overlapping_queries}"
    )

    progress_cache_path = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "06_iconq_training",
        "progress_cache.yml",
    )

    cached_progress = {}

    if os.path.exists(progress_cache_path):
        if args.force:
            ts_for_old_filename = str(int(datetime.now().timestamp()))
            print(
                f"Forcing re-training. Moving old cached progress to progress_cache_{ts_for_old_filename}.yml"
            )
            os.rename(
                progress_cache_path,
                os.path.join(
                    pu.AUTOSLO_ROOT,
                    "experiments",
                    "06_iconq_training",
                    f"progress_cache_{ts_for_old_filename}.yml",
                ),
            )
        else:
            with open(progress_cache_path, "r") as f:
                cached_progress = yaml.safe_load(f)

    print("Step 1: Finding run IDs...")
    if "run_ids" not in cached_progress:
        print("\tRun IDs not cached. Computing...")
        run_ids, train_val_test_assignments = find_run_ids()
        cached_progress["run_ids"] = run_ids
        cached_progress["train_val_test_assignments"] = (
            train_val_test_assignments
        )
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached run IDs.")
        run_ids = cached_progress["run_ids"]

    print("Step 2: Setting up featurizer...")
    if "featurizer_id" not in cached_progress:
        print("\tFeaturizer not cached. Computing...")
        run_ids = cached_progress["run_ids"]
        featurizer_id = set_up_featurizer(run_ids, args.from_sys_query_explain)
        cached_progress["featurizer_id"] = featurizer_id
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached featurizer ID.")
        featurizer_id = cached_progress["featurizer_id"]

    train_val_run_ids = (
        cached_progress["train_val_test_assignments"]["train"]
        + cached_progress["train_val_test_assignments"]["val"]
    )

    print("Step 3: Training CacheModel...")
    if "cache_model_id" not in cached_progress:
        print("\tCacheModel not cached. Training...")
        cache_model_id = train_cache_model(
            train_val_run_ids if args.use_explicit_run_ids else run_ids,
            only_non_overlapping_queries=args.only_non_overlapping_queries,
            ignore_cluster_size=args.ignore_cluster_size,
        )
        cached_progress["cache_model_id"] = cache_model_id
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached CacheModel ID.")
        cache_model_id = cached_progress["cache_model_id"]

    print("Step 4: Training XGBoostModel...")
    if "xgboost_model_id" not in cached_progress:
        print("\tXGBoostModel not cached. Training...")
        xgboost_model_id = train_xgboost_model(
            iconq_query_featurizer_id=featurizer_id,
            run_ids=train_val_run_ids if args.use_explicit_run_ids else run_ids,
            only_non_overlapping_queries=args.only_non_overlapping_queries,
            ignore_cluster_size=args.ignore_cluster_size,
        )
        cached_progress["xgboost_model_id"] = xgboost_model_id
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached XGBoostModel ID.")
        xgboost_model_id = cached_progress["xgboost_model_id"]

    print("Step 5: Initializing StageModel...")
    if "stage_model_id" not in cached_progress:
        print("\tStageModel not cached. Initializing...")
        stage_model = initialize_stage_model(
            cache_model_id=cache_model_id,
            xgboost_model_id=xgboost_model_id,
        )
        stage_model_id = stage_model.save()
        cached_progress["stage_model_id"] = stage_model_id
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached StageModel ID.")
        stage_model_id = cached_progress["stage_model_id"]

    print("Step 6: Training IconqModel...")
    explicit_run_ids_per_split = None
    if args.use_explicit_run_ids:
        explicit_run_ids_per_split = cached_progress[
            "train_val_test_assignments"
        ]
    iconq_model_id = train_iconq_model(
        iconq_query_featurizer_id=featurizer_id,
        stage_model_id=stage_model_id,
        run_ids=run_ids,
        use_stage_for_isolated_queries=args.use_stage_for_isolated_queries,
        explicit_run_ids_per_split=explicit_run_ids_per_split,
        penalize_based_on_overlap=args.penalize_based_on_overlap,
        sensitive_q_error_loss_version=args.sensitive_q_error_loss_version,
        ignore_cluster_size=args.ignore_cluster_size,
    )
