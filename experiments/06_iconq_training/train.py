import os

import yaml

import autoslo.utils.paths as pu
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.cache_model import CacheModel
from autoslo.models.iconq_model import (
    IconqModel,
    IconqModelInitConfig,
    NNModelTrainConfig,
)
from autoslo.models.stage_model import StageModel
from autoslo.models.xgboost_model import XGBoostModel
from autoslo.models.iconq_model_trainer import iconq_model_trainer


def find_run_ids() -> list[str]:
    """Finds all relevant run IDs for training IconQ models."""

    df = pu.RunLocator.get_runs_df()
    pct_heavy_options = [0, 10, 25, 50]
    mean_interarrival_options = [10, 30, 60, 120]
    rpus = [4, 8, 16, 32]

    run_ids = []

    # Most recent chunk runs.
    for pct_heavy in pct_heavy_options:
        for mean_interarrival in mean_interarrival_options:
            for rpu in rpus:
                this_workload_run_ids = pu.RunLocator.get_run_ids(
                    schema_name="ext_tpcds1000",
                    workload_name="{}pctheavy_{}meaninterarrival".format(
                        pct_heavy, mean_interarrival
                    ),
                    blueprint_name=f"single_{rpu}",
                )
                if len(this_workload_run_ids) > 0:
                    run_ids.append(max(this_workload_run_ids))

    # Benchmarking workload runs.
    benchmark_runs = df[
        df["workload_name"] == "benchmarking_workload_99_3_3_shuffled_42"
    ]
    run_ids.extend(benchmark_runs["run_id"].to_list())
    return run_ids


def set_up_featurizer(run_ids: list[str]) -> str:
    """Sets up the IconQ query featurizer and returns its ID."""

    featurizer = IconqQueryFeaturizer(schema_name="tpcds1000", run_ids=run_ids)
    return featurizer.save()


def train_cache_model(run_ids: list[str]) -> str:
    """Trains a CacheModel and returns its ID."""

    cache_model = CacheModel(enable_template_cache=False, best_effort=False)
    cache_model.train(
        run_ids=run_ids,
        from_scratch=True,
    )
    cache_model_id = cache_model.save()
    return cache_model_id


def train_xgboost_model(
    iconq_query_featurizer_id: str, run_ids: list[str]
) -> str:
    """Trains an XGBoostModel and returns its ID."""
    xgboost_model = XGBoostModel(
        train_on_log_runtime=True,
        n_estimators=100,
        iconq_query_featurizer_id=iconq_query_featurizer_id,
    )
    xgboost_model.train(
        run_ids=run_ids,
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
    iconq_query_featurizer_id: str, stage_model_id: str, run_ids: list[str]
) -> str:
    """Trains an IconqModel and returns its ID."""
    iconq_model_init_config = IconqModelInitConfig(
        iconq_query_featurizer_id=iconq_query_featurizer_id,
        stage_model_id=stage_model_id,
        is_bayesian=False,
        is_mdn=False,
        train_on_log_runtime=True,
    )
    nn_model_train_config = NNModelTrainConfig(
        run_ids=run_ids,
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
    progress_cache_path = os.path.join(
        pu.AUTOSLO_ROOT,
        "experiments",
        "06_iconq_training",
        "progress_cache.yml",
    )

    cached_progress = {}

    if os.path.exists(progress_cache_path):
        with open(progress_cache_path, "r") as f:
            cached_progress = yaml.safe_load(f)

    print("Step 1: Finding run IDs...")
    if "run_ids" not in cached_progress:
        print("\tRun IDs not cached. Computing...")
        run_ids = find_run_ids()
        cached_progress["run_ids"] = run_ids
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached run IDs.")
        run_ids = cached_progress["run_ids"]

    print("Step 2: Setting up featurizer...")
    if "featurizer_id" not in cached_progress:
        print("\tFeaturizer not cached. Computing...")
        run_ids = cached_progress["run_ids"]
        featurizer_id = set_up_featurizer(run_ids)
        cached_progress["featurizer_id"] = featurizer_id
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached featurizer ID.")
        featurizer_id = cached_progress["featurizer_id"]

    print("Step 3: Training CacheModel...")
    if "cache_model_id" not in cached_progress:
        print("\tCacheModel not cached. Training...")
        cache_model_id = train_cache_model(run_ids)
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
            run_ids=run_ids,
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
    if "iconq_model_id" not in cached_progress:
        print("\tIconqModel not cached. Training...")
        iconq_model_id = train_iconq_model(
            iconq_query_featurizer_id=featurizer_id,
            stage_model_id=stage_model_id,
            run_ids=run_ids,
        )
        cached_progress["iconq_model_id"] = iconq_model_id
        with open(progress_cache_path, "w") as f:
            yaml.safe_dump(cached_progress, f)
    else:
        print("\tUsing cached IconqModel ID.")
        iconq_model_id = cached_progress["iconq_model_id"]
