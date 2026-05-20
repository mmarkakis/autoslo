import argparse
import os
import shutil
from pathlib import Path

import autoslo.filesystem.path_utils as pu
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.filesystem.yaml_helpers import dump_yaml, load_yaml
from autoslo.models.cache_model import CacheModel
from autoslo.models.iconq_dataset_builder import build_dataset_from_trace
from autoslo.models.iconq_model import IconqModel
from autoslo.models.iconq_model_config import (
    IconqModelInitConfig,
    IconqModelTrainConfig,
)
from autoslo.models.stage_model import StageModel
from autoslo.models.xgboost_model import XGBoostModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.workload_execution.trace import Trace

parser = argparse.ArgumentParser(
    description="Trains an IconqModel step-by-step, caching progress."
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Force re-training even if cached progress exists.",
)
parser.add_argument(
    "--train_config_path",
    required=True,
    help="Path to the training config YAML (data/model_training_configs/).",
)

args = parser.parse_args()

cfg = load_yaml(args.train_config_path)
iconq_model_id = Path(args.train_config_path).stem
iconq_model_dir = os.path.join(
    pu.get_data_path(), "iconq_models", iconq_model_id
)
if os.path.exists(iconq_model_dir) and args.force:
    print(f"Forcing re-training. Deleting {iconq_model_dir}...")
    shutil.rmtree(iconq_model_dir)

os.makedirs(iconq_model_dir, exist_ok=True)
progress_cache_path = os.path.join(iconq_model_dir, "progress_cache.yml")
cached_progress: dict = {}

if os.path.exists(progress_cache_path):
    cached_progress = load_yaml(progress_cache_path)

iconq_model_init_config_dict = cfg.get("iconq_model_init_config", {})
iconq_model_train_config_dict = cfg.get("iconq_model_train_config", {})

ignore_cluster_size: bool = iconq_model_init_config_dict.get(
    "ignore_cluster_size", False
)
train_stage_only_on_isolated_queries: bool = iconq_model_train_config_dict.get(
    "train_stage_only_on_isolated_queries", False
)
use_client_side_latencies: bool = iconq_model_train_config_dict.get(
    "use_client_side_latencies", False
)
ignore_aborted_queries: bool = iconq_model_train_config_dict.get(
    "ignore_aborted_queries", False
)

#########
## Step 1: Read run IDs from config.
#########

print("Step 1: Reading run IDs from config...")
train_val_run_ids = iconq_model_train_config_dict["run_ids"]

#########
## Step 2: Setting up featurizer.
#########

print("Step 2: Setting up featurizer...")
if "featurizer_id" not in cached_progress:
    print("\tFeaturizer not cached. Computing...")
    featurizer = IconqQueryFeaturizer(
        schema_name=iconq_model_init_config_dict["schema_name"],
        run_ids=train_val_run_ids,
    )
    featurizer_id = featurizer.save()
    cached_progress["featurizer_id"] = featurizer_id
    dump_yaml(cached_progress, progress_cache_path)
else:
    print("\tUsing cached featurizer ID.")
    featurizer_id = cached_progress["featurizer_id"]

##########
## Step 3: Train CacheModel.
##########

print("Step 3: Training CacheModel...")
if "cache_model_id" not in cached_progress:
    print("\tCacheModel not cached. Training...")
    cache_model = CacheModel(
        enable_template_cache=False,
        best_effort=False,
        ignore_cluster_size=ignore_cluster_size,
    )
    cache_model.train(
        run_ids=train_val_run_ids,
        from_scratch=True,
        only_non_overlapping_queries=train_stage_only_on_isolated_queries,
        use_client_side_latencies=use_client_side_latencies,
        ignore_aborted_queries=ignore_aborted_queries,
    )
    cache_model_id = cache_model.save()
    cached_progress["cache_model_id"] = cache_model_id
    dump_yaml(cached_progress, progress_cache_path)
else:
    print("\tUsing cached CacheModel ID.")
    cache_model_id = cached_progress["cache_model_id"]

###########
## Step 4: Train XGBoostModel.
###########

print("Step 4: Training XGBoostModel...")
if "xgboost_model_id" not in cached_progress:
    print("\tXGBoostModel not cached. Training...")
    xgboost_model = XGBoostModel(
        train_on_log_runtime=True,
        n_estimators=100,
        schema_name=iconq_model_init_config_dict["schema_name"],
        iconq_query_featurizer_id=featurizer_id,
        ignore_cluster_size=ignore_cluster_size,
    )
    xgboost_model.train(
        run_ids=train_val_run_ids,
        only_non_overlapping_queries=train_stage_only_on_isolated_queries,
        use_client_side_latencies=use_client_side_latencies,
        ignore_aborted_queries=ignore_aborted_queries,
    )
    xgboost_model_id = xgboost_model.save()
    cached_progress["xgboost_model_id"] = xgboost_model_id
    dump_yaml(cached_progress, progress_cache_path)
else:
    print("\tUsing cached XGBoostModel ID.")
    xgboost_model_id = cached_progress["xgboost_model_id"]

###########
## Step 5: Initialize StageModel.
###########

print("Step 5: Initializing StageModel...")
if "stage_model_id" not in cached_progress:
    print("\tStageModel not cached. Initializing...")
    stage_model = StageModel(
        cache_model_id=cache_model_id,
        xgboost_model_id=xgboost_model_id,
    )
    stage_model_id = stage_model.save()
    cached_progress["stage_model_id"] = stage_model_id
    dump_yaml(cached_progress, progress_cache_path)
else:
    print("\tUsing cached StageModel ID.")
    stage_model_id = cached_progress["stage_model_id"]

###########
## Step 6: Train IconqModel.
###########

print("Step 6: Training IconqModel...")
iconq_model_init_config_dict["iconq_query_featurizer_id"] = featurizer_id
iconq_model_init_config_dict["stage_model_id"] = stage_model_id
iconq_model_init_config = IconqModelInitConfig(**iconq_model_init_config_dict)
iconq_model_train_config = IconqModelTrainConfig(
    **iconq_model_train_config_dict
)

iconq_model = IconqModel(
    init_config=iconq_model_init_config,
    train_config=iconq_model_train_config,
    model_id=iconq_model_id,
)

datasets = [
    build_dataset_from_trace(
        trace=Trace(run_id),
        iconq_model=iconq_model,
        use_log_runtime=iconq_model.trained_on_log_runtime,
        use_client_side_latencies=use_client_side_latencies,
        use_fixed_window_radius_s=iconq_model_init_config.use_fixed_window_radius_s,
        use_fixed_window_max_neighbors_per_side=iconq_model_init_config.use_fixed_window_max_neighbors_per_side,
        ignore_aborted_queries=ignore_aborted_queries,
    )
    for run_id in train_val_run_ids
]
overall_dataset = ConcurrentQueryDataset.concatenate(datasets)

train_dataloader, val_dataloader, test_dataloader = (
    iconq_model._get_dataloaders(
        overall_dataset, iconq_model_train_config, split=True, save_dataset=True
    )
)
assert val_dataloader is not None
iconq_model._run_training_loop(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
)
iconq_model_id_readout = iconq_model.save()
assert iconq_model_id_readout == iconq_model_id
print(f"Done. IconqModel ID: {iconq_model_id}")
