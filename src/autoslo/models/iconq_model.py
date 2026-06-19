from __future__ import annotations

import logging
import os
import pickle
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import torch
import yaml
from intervaltree import Interval, IntervalTree  # type: ignore[import]
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import Cluster
from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.model_training.iconq_model_training_checkpoint import (
    update_checkpoint,
    update_plots,
)
from autoslo.models.iconq_model_config import (
    IconqModelInitConfig,
    IconqModelTrainConfig,
)
from autoslo.models.model_prediction import ModelPrediction
from autoslo.models.stage_model import StageModel
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.nn.loss_functions import (
    LossType,
    mdn_negative_log_likelihood_loss,
    negative_log_likelihood_loss,
    sensitive_q_error_loss,
)
from autoslo.nn.runtime_net import RuntimeNet
from autoslo.workload_definition.query import ClusterAwareQueryId, Query
from autoslo.workload_execution.trace import Trace

logger = logging.getLogger(__name__)

_console = Console()


def _infer_runtime_net_input_size_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> Optional[int]:
    """Infer RuntimeNet input-size from a saved state dict, when possible."""
    if "_bn.weight" in state_dict:
        return int(state_dict["_bn.weight"].shape[0])
    if "_in_model.0.weight" in state_dict:
        return int(state_dict["_in_model.0.weight"].shape[1])
    return None


def _validate_runtime_net_input_size(
    state_dict: dict[str, torch.Tensor],
    expected_input_size: int,
    interaction_feature_version: str,
) -> None:
    """Raise a clear error if checkpoint feature dimensionality mismatches."""
    checkpoint_input_size = _infer_runtime_net_input_size_from_state_dict(
        state_dict
    )
    if (
        checkpoint_input_size is not None
        and checkpoint_input_size != expected_input_size
    ):
        raise ValueError(
            "Checkpoint/input feature mismatch: checkpoint expects "
            f"input_size={checkpoint_input_size}, but current model was "
            f"constructed with input_size={expected_input_size} "
            f"(interaction_feature_version={interaction_feature_version}). "
            "Load the model with matching feature version/config or retrain."
        )


class DataSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def print_errors_table(
    title: Optional[str],
    sets: list[tuple[str, dict[str, float]]],
) -> None:
    """Print a Rich Table of per-set error metrics.

    Groups within the same split are separated by a single horizontal line.
    Groups from different splits are separated by a double horizontal line
    (implemented as an empty spacer row each marked end_section=True).
    """
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("set / queries", no_wrap=True)
    table.add_column("metric", no_wrap=True)
    for col in ("mean", "p50", "p90", "p95"):
        table.add_column(col, justify="right")

    _EMPTY_ROW = ("", "", "", "", "", "")
    n_sets = len(sets)
    metrics_per_type = {
        "normal": ["q_error", "abs_error"],
        "aborted": ["factor_error", "underprediction_error_s"],
    }

    for set_idx, (set_name, errs) in enumerate(sets):
        is_last_set = set_idx == n_sets - 1

        for suffix, metrics in metrics_per_type.items():
            n = int(errs.get(f"n_{suffix}", 0))
            if n == 0:
                continue

            for i, metric in enumerate(metrics):
                table.add_row(
                    f"{set_name} / {suffix} (N={n})" if i == 0 else "",
                    metric,
                    *(
                        f'{errs[f"{stat}_{metric}_{suffix}"]:.4f}'
                        for stat in ("mean", "p50", "p90", "p95")
                    ),
                    style="white" if i == 0 else "dim",
                    end_section=True if i == len(metrics) - 1 else False,
                )
        # Double horizontal line between splits: the abs-error row above already
        # added one rule via end_section; a blank spacer row with end_section
        # adds a second immediately adjacent rule.
        if not is_last_set:
            table.add_row(*_EMPTY_ROW, end_section=True)

    if title is not None:
        _console.print(Rule(title))
    _console.print(table)


class IconqModel:
    """
    A query runtime model that uses an LSTM to predict query runtimes.
    Optionally, it can also predict the uncertainty of the predictions.
    """

    @staticmethod
    def default_save_dir(model_id: str) -> str:
        return os.path.join(pu.get_data_path(), "iconq_models", model_id)

    def __init__(
        self,
        init_config: IconqModelInitConfig,
        train_config: Optional[IconqModelTrainConfig] = None,
        device: torch.device = torch.device("cpu"),
        parent_save_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        inference_mode: bool = False,
    ) -> None:
        """
        Initializes the LSTM model.

        Parameters:
            init_config: The configuration for the LSTM model.
            train_config: The training configuration used to train the model.
            device: The device to use for training and prediction.
            parent_save_dir: The parent directory to save the model.
            model_id: The identifier of the model. If None, a new model ID
                will be generated.
            inference_mode: When ``True``, skip loading ``dataset.pkl`` and
                the split-index files.  Use this for inference-only contexts
                where the training dataset is never needed.
        """
        self._device = device
        self._init_config = init_config
        self._train_config: Optional[IconqModelTrainConfig] = train_config
        self._inference_mode = inference_mode

        # Create save directory and set model ID.
        if model_id is None:
            model_id = str(int(datetime.now().timestamp()))
        self._model_id = model_id
        self._parent_save_dir = parent_save_dir
        self._save_dir = self.default_save_dir(model_id)
        if parent_save_dir is not None:
            self._save_dir = os.path.join(parent_save_dir, model_id)
        os.makedirs(self._save_dir, exist_ok=True)

        # Initialize the query and interaction featurizer.
        if init_config.iconq_query_featurizer_id is None:
            if init_config.iconq_query_featurizer_init_params is None:
                raise ValueError(
                    "Must provide either iconq_query_featurizer_id or "
                    "iconq_query_featurizer_init_params."
                )
            self._iconq_query_featurizer = IconqQueryFeaturizer(
                **init_config.iconq_query_featurizer_init_params
            )
            self._iconq_query_featurizer_id = (
                self._iconq_query_featurizer.save()
            )
        else:
            self._iconq_query_featurizer_id = (
                init_config.iconq_query_featurizer_id
            )
            self._iconq_query_featurizer = IconqQueryFeaturizer.load(
                self._init_config.schema_name,
                init_config.iconq_query_featurizer_id,
            )
        self._iconq_interaction_featurizer = IconqInteractionFeaturizer(
            schema_name=init_config.schema_name,
            iconq_query_featurizer_id=self._iconq_query_featurizer_id,
            ignore_cluster_size=init_config.ignore_cluster_size,
            interaction_feature_version=init_config.interaction_feature_version,
        )

        # Initialize the stage model, if applicable.
        if init_config.stage_model_id is None:
            if init_config.stage_model_init_params is None:
                raise ValueError(
                    "Must provide either stage_model_id or "
                    "stage_model_init_params."
                )
            self._stage_model = StageModel(
                **init_config.stage_model_init_params
            )
            self._stage_model_id = self._stage_model.save()
        else:
            self._stage_model_id = init_config.stage_model_id
            self._stage_model = StageModel.load(init_config.stage_model_id)

        # Initialize the runtime net.
        self._nn_args = {
            "input_size": self._iconq_interaction_featurizer.num_dims,
            "embedding_size": init_config.embedding_size,
            "lstm_hidden_size": init_config.lstm_hidden_size,
            "lstm_num_layers": init_config.lstm_num_layers,
            "lstm_dropout": init_config.lstm_dropout,
            "is_bayesian": init_config.is_bayesian,
            "bayesian_samples": init_config.bayesian_samples,
            "is_mdn": init_config.is_mdn,
            "mdn_num_gaussians": init_config.mdn_num_gaussians,
            "device": self._device,
        }
        self._nn = RuntimeNet(**self._nn_args).to(self._device)  # type: ignore
        self._trained_on_log_runtime = init_config.train_on_log_runtime

        self._loss_type: LossType
        if self._nn_args["is_mdn"]:
            self._loss_type = LossType.MDN_NLL
        elif self._nn_args["is_bayesian"]:
            self._loss_type = LossType.NLL
        else:
            self._loss_type = LossType.SENSITIVE_Q_ERROR

        if (
            self._train_config is not None
            and self._train_config.neighbor_derived_censored_observation_prob
            > 0.0
            and self._loss_type != LossType.SENSITIVE_Q_ERROR
        ):
            raise ValueError(
                "neighbor_derived_censored_observation_prob requires "
                "LossType.SENSITIVE_Q_ERROR (is_mdn=False, is_bayesian=False). "
                f"Current loss type: {self._loss_type}."
            )

        # Initialize the dataset and split indices.
        self._idxs_for_split: dict[DataSplit, list[int]] = {}
        if not self._inference_mode:
            self._populate_dataset_and_split_idxs()
            self._save_params()
        else:
            self._dataset: Optional[ConcurrentQueryDataset] = None

    @property
    def init_config(self) -> "IconqModelInitConfig":
        """The initialisation config this model was constructed with."""
        return self._init_config

    @property
    def stage_model(self) -> StageModel:
        """
        Get the stage model used by the IconqModel.

        Returns:
            The StageModel instance.
        """
        return self._stage_model

    @property
    def trained_on_log_runtime(self) -> bool:
        """
        Returns whether the model was trained on log runtimes.

        Returns:
            True if the model was trained on log runtimes, False otherwise.
        """
        return self._trained_on_log_runtime

    @property
    def iconq_query_featurizer(self) -> IconqQueryFeaturizer:
        """
        Get the IconqQueryFeaturizer used by the IconqModel.

        Returns:
            The IconqQueryFeaturizer instance.
        """
        return self._iconq_query_featurizer

    @property
    def iconq_interaction_featurizer(self) -> IconqInteractionFeaturizer:
        """
        Get the IconqInteractionFeaturizer used by the IconqModel.

        Returns:
            The IconqInteractionFeaturizer instance.
        """
        return self._iconq_interaction_featurizer

    def predict_from_dataset(
        self,
        dataset: ConcurrentQueryDataset,
        inference_batch_size: int = 512,
    ) -> dict[ClusterAwareQueryId, ModelPrediction]:
        """
        Predicts the runtimes for the queries in the given dataset.

        Parameters:
            dataset: The dataset to predict runtimes for.
            inference_batch_size: Maximum number of items to process per
                forward pass. Defaults to 512, which handles typical simulator
                datasets in one shot while remaining memory-safe.

        Returns:
            A flat dictionary mapping each query's :class:`ClusterAwareQueryId`
            to its :class:`ModelPrediction`.
        """

        predictions: dict[ClusterAwareQueryId, ModelPrediction] = {}
        n = len(dataset)
        if n == 0:
            return predictions

        if self._nn.training:
            self._nn.eval()
        with torch.no_grad():
            # Bypass the DataLoader machinery entirely: directly collate
            # fixed-size slices and pass them to a loss-free inference method.
            # This eliminates per-batch DataLoader overhead (iterator init,
            # collate_fn wrapping) and avoids computing losses during inference.
            for start in range(0, n, inference_batch_size):
                end = min(start + inference_batch_size, n)
                batch = ConcurrentQueryDataset.collate_for_inference(
                    dataset, start, end
                )
                predictions.update(self._infer_batch(batch))

        return predictions

    def _predict_isolated_query(
        self,
        i: int,
        x: torch.Tensor,
        pinch_points: torch.Tensor,
        cluster_aware_query_ids: list[ClusterAwareQueryId],
        query_text_ids: list,
        y_is_lower_bound: torch.Tensor,
    ) -> ModelPrediction:
        """
        Predict a single isolated query (sequence length == 1) via the stage
        model and wrap the result in a ModelPrediction.
        """
        rpu = x[i][pinch_points[i]][
            self._iconq_interaction_featurizer.rpu_dim_idx
        ].item()
        pred = self.stage_model.predict_from_query_text_id(
            {cluster_aware_query_ids[i]: query_text_ids[i]},
            cluster_rpu=int(rpu),
        )[cluster_aware_query_ids[i]]
        return ModelPrediction(
            mean_s=pred.mean_s,
            std_dev_s=pred.std_dev_s,
            mix_coeffs=pred.mix_coeffs,
            metadata={
                "num_other_concurrent_queries": 0,
                "run_id": Cluster.run_id_for_cluster_name(
                    cluster_aware_query_ids[i].cluster_name
                ),
                "rpu": int(rpu),
                "model_source": "stage",
                "query_text_id": query_text_ids[i],
                "query_id": cluster_aware_query_ids[i].query_id,
                "target_is_lower_bound": y_is_lower_bound[i].item(),
                "loss": 0.0,
            },
        )

    def _build_predictions_from_nn_output(
        self,
        y_pred_mean: torch.Tensor,
        y_pred_logvar: torch.Tensor,
        y_pred_mix: torch.Tensor,
        x_len: torch.Tensor,
        cluster_aware_query_ids: list[ClusterAwareQueryId],
        query_text_ids: list,
        y_is_lower_bound: torch.Tensor,
        per_item_losses: Optional[np.ndarray] = None,
    ) -> dict[ClusterAwareQueryId, ModelPrediction]:
        """
        Convert NN output tensors into a dict mapping each
        :class:`ClusterAwareQueryId` to its :class:`ModelPrediction`.
        Handles all three loss types.

        Parameters:
            per_item_losses: Per-sample loss values (only used for
                LossType.SENSITIVE_Q_ERROR metadata). Pass None (default) during
                inference — the metadata field will be set to 0.0.
        """
        # Move all tensors to CPU and convert to numpy once upfront to minimize GPU sync points.
        x_len_np = x_len.detach().cpu().numpy()
        y_is_lower_bound_np = y_is_lower_bound.detach().cpu().numpy()
        mean_np = y_pred_mean.detach().cpu().numpy()
        logvar_np = y_pred_logvar.detach().cpu().numpy()
        # Vectorize logvar->stddev conversion (exp(logvar/2)) instead of per-item list comprehension.
        stddev_np = np.exp(logvar_np / 2)
        mix_np = (
            y_pred_mix.detach().cpu().numpy()
            if y_pred_mix is not None
            else None
        )
        result: dict[ClusterAwareQueryId, ModelPrediction] = {}

        # Pre-compute run_id and rpu once per unique cluster name so that
        # Cluster.run_id_for_cluster_name / rpu_for_cluster_name (which call
        # str.split) are not invoked once per prediction inside the hot loop.
        unique_cluster_names = {
            caqi.cluster_name for caqi in cluster_aware_query_ids
        }
        run_id_by_cluster: dict[str, str] = {
            cluster_name: Cluster.run_id_for_cluster_name(cluster_name)
            for cluster_name in unique_cluster_names
        }
        rpu_by_cluster: dict[str, int] = {
            cluster_name: Cluster.rpu_for_cluster_name(cluster_name)
            for cluster_name in unique_cluster_names
        }

        if self._loss_type == LossType.MDN_NLL:
            for i, (m, le, mix, q, t) in enumerate(
                zip(
                    mean_np,
                    x_len_np,
                    mix_np,
                    cluster_aware_query_ids,
                    query_text_ids,
                )
            ):
                result[q] = ModelPrediction(
                    mean_s=m,
                    std_dev_s=stddev_np[i],
                    mix_coeffs=mix,
                    metadata={
                        "num_other_concurrent_queries": int(le) - 1,
                        "run_id": run_id_by_cluster[q.cluster_name],
                        "rpu": rpu_by_cluster[q.cluster_name],
                        "model_source": "lstm",
                        "query_text_id": t,
                        "query_id": q.query_id,
                    },
                )
        elif self._loss_type == LossType.NLL:
            for i, (m, le, q, t) in enumerate(
                zip(
                    mean_np,
                    x_len_np,
                    cluster_aware_query_ids,
                    query_text_ids,
                )
            ):
                result[q] = ModelPrediction(
                    mean_s=m,
                    std_dev_s=stddev_np[i],
                    metadata={
                        "num_other_concurrent_queries": int(le) - 1,
                        "run_id": run_id_by_cluster[q.cluster_name],
                        "rpu": rpu_by_cluster[q.cluster_name],
                        "model_source": "lstm",
                        "query_text_id": t,
                        "query_id": q.query_id,
                    },
                )
        else:  # LossType.SENSITIVE_Q_ERROR
            losses: list | np.ndarray = (
                per_item_losses
                if per_item_losses is not None
                else [0.0] * len(mean_np)
            )
            for m, le, q, t, yislb, loss_val in zip(
                mean_np,
                x_len_np,
                cluster_aware_query_ids,
                query_text_ids,
                y_is_lower_bound_np,
                losses,
            ):
                result[q] = ModelPrediction(
                    mean_s=m,
                    metadata={
                        "num_other_concurrent_queries": int(le) - 1,
                        "run_id": run_id_by_cluster[q.cluster_name],
                        "rpu": rpu_by_cluster[q.cluster_name],
                        "model_source": "lstm",
                        "query_text_id": t,
                        "query_id": q.query_id,
                        "target_is_lower_bound": yislb,
                        "loss": loss_val,
                    },
                )

        return result

    def _infer_batch(
        self,
        batch: tuple,
    ) -> dict[ClusterAwareQueryId, "ModelPrediction"]:
        """
        Runs a single forward pass on a pre-collated batch without computing
        any loss. Returns a dict mapping each :class:`ClusterAwareQueryId` to its
        :class:`ModelPrediction`.

        This is faster than ``_process_batch(..., derive_individual_predictions=True)``
        because it skips the loss computation entirely.
        """

        (
            x,
            x_len,
            pinch_points,
            y,
            cluster_aware_query_ids,
            query_text_ids,
            y_is_lower_bound,
        ) = batch

        if self._train_config is None:
            raise RuntimeError(
                "No train_config set — cannot run inference without a "
                "train_config."
            )
        train_config = self._train_config
        result: dict[ClusterAwareQueryId, ModelPrediction] = {}

        # Handle isolated queries via the stage model when configured.
        # Batch the length check with a single CPU conversion instead of per-item .item() calls.
        if train_config.use_stage_for_isolated_queries:
            x_len_np = (
                x_len.cpu().numpy()
                if isinstance(x_len, torch.Tensor)
                else x_len
            )
            non_isolated_indices = cast(
                list[int], np.where(x_len_np > 1)[0].tolist()
            )
            isolated_indices = cast(
                list[int], np.where(x_len_np <= 1)[0].tolist()
            )

            if isolated_indices:
                # Group by RPU so we issue one stage-model call per RPU rather
                # than one call per isolated query.  Isolated queries have
                # x_len == 1 so pinch_points[i] == 0; the RPU sits in row 0.
                rpu_dim = self._iconq_interaction_featurizer.rpu_dim_idx
                rpu_to_indices: dict[int, list[int]] = {}
                for i in isolated_indices:
                    rpu = int(x[i, 0, rpu_dim].item())
                    rpu_to_indices.setdefault(rpu, []).append(i)

                for rpu, indices in rpu_to_indices.items():
                    qtid_map = {
                        cluster_aware_query_ids[i]: query_text_ids[i]
                        for i in indices
                    }
                    preds = self.stage_model.predict_from_query_text_id(
                        qtid_map, cluster_rpu=rpu
                    )
                    for i in indices:
                        caqi = cluster_aware_query_ids[i]
                        pred = preds[caqi]
                        result[caqi] = ModelPrediction(
                            mean_s=pred.mean_s,
                            std_dev_s=pred.std_dev_s,
                            mix_coeffs=pred.mix_coeffs,
                            metadata={
                                "num_other_concurrent_queries": 0,
                                "run_id": Cluster.run_id_for_cluster_name(
                                    caqi.cluster_name
                                ),
                                "rpu": rpu,
                                "model_source": "stage",
                                "query_text_id": query_text_ids[i],
                                "query_id": caqi.query_id,
                                "target_is_lower_bound": y_is_lower_bound[
                                    i
                                ].item(),
                                "loss": 0.0,
                            },
                        )

            if not non_isolated_indices:
                return result

            x = x[non_isolated_indices]
            x_len = x_len[non_isolated_indices]
            pinch_points = pinch_points[non_isolated_indices]
            cluster_aware_query_ids = [
                cluster_aware_query_ids[i] for i in non_isolated_indices
            ]
            query_text_ids = [query_text_ids[i] for i in non_isolated_indices]
            y_is_lower_bound = y_is_lower_bound[non_isolated_indices]

        # Move tensors to device once at the start, not multiple times in the forward call.
        x = x.to(self._device)
        x_len = x_len.to(self._device)
        pinch_points = pinch_points.to(self._device)
        y_pred_mean, y_pred_logvar, y_pred_mix = self._nn(
            x,
            x_len,
            pinch_points,
            train_config.mdn_mix_softmax_temperature,
        )

        if self._trained_on_log_runtime:
            y_pred_mean = torch.exp(y_pred_mean)
            y_pred_logvar = torch.exp(y_pred_logvar)

        result.update(
            self._build_predictions_from_nn_output(
                y_pred_mean,
                y_pred_logvar,
                y_pred_mix,
                x_len,
                cluster_aware_query_ids,
                query_text_ids,
                y_is_lower_bound,
            )
        )
        return result

    def save(self) -> str:
        """
        Saves the IconqModel.

        Returns:
            The identifier of the saved IconqModel. This is a subdirectory under
                the parent_save_dir.
        """

        self._save_params()
        update_checkpoint(self._nn, self._save_dir)

        return self._model_id

    def _save_params(self) -> None:
        """
        Saves the model parameters to disk.
        """

        param_path = os.path.join(self._save_dir, "params.yml")
        with open(param_path, "w") as f:
            yaml.safe_dump(
                {
                    "init_config": asdict(self._init_config),
                    "train_config": (
                        asdict(self._train_config)
                        if self._train_config is not None
                        else None
                    ),
                    "device": str(self._device),
                    "parent_save_dir": self._parent_save_dir,
                    "model_id": self._model_id,
                },
                f,
            )

    def _load_nn_state_dict_with_feature_guard(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Load RuntimeNet weights after validating input feature compatibility."""
        _validate_runtime_net_input_size(
            state_dict=state_dict,
            expected_input_size=cast(int, self._nn_args["input_size"]),
            interaction_feature_version=self._init_config.interaction_feature_version,
        )
        self._nn.load_state_dict(state_dict)

    @staticmethod
    def load(
        model_id: str,
        parent_load_dir: Optional[str] = None,
        inference_mode: bool = True,
    ) -> "IconqModel":
        """
        Load the given IconqModel.

        Parameters:
            model_id: The identifier of the saved IconqModel to load.
            parent_load_dir: The parent directory where iconq models are stored.
                If None, defaults to `data/iconq_models/`.
        """

        if parent_load_dir is None:
            parent_load_dir = os.path.join(pu.get_data_path(), "iconq_models")
        load_dir = os.path.join(parent_load_dir, model_id)
        if not os.path.exists(load_dir):
            raise ValueError(f"IconqModel directory {load_dir} does not exist.")

        # Load model parameters
        param_path = os.path.join(load_dir, "params.yml")
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)

        if not isinstance(params, dict):
            raise ValueError(
                f"IconqModel params.yml is invalid (got {type(params).__name__}): "
                f"{param_path}"
            )
        if params.get("init_config") is None:
            raise ValueError(
                f"IconqModel params.yml missing 'init_config': {param_path}"
            )

        train_config_dict: dict[str, Any] = {}
        if "train_config_sequence" in params:
            train_config_dict = (
                params["train_config_sequence"][-1]
                if params["train_config_sequence"]
                else {}
            )
        else:
            train_config_dict = params.get("train_config", {}) or {}

        model = IconqModel(
            init_config=IconqModelInitConfig(**params["init_config"]),
            train_config=IconqModelTrainConfig(**train_config_dict),
            device=torch.device(params["device"]),
            parent_save_dir=parent_load_dir,
            model_id=model_id,
            inference_mode=inference_mode,
        )

        # Load the model checkpoint
        checkpoint_files = [
            file
            for file in os.listdir(load_dir)
            if file.startswith("model_") and file.endswith(".pth")
        ]
        if checkpoint_files:
            latest_checkpoint_file = max(
                checkpoint_files,
                key=lambda f: int(f[len("model_") : -len(".pth")]),
            )
            checkpoint_path = os.path.join(load_dir, latest_checkpoint_file)
            print(f"Loading model checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=model._device)
            model._load_nn_state_dict_with_feature_guard(state_dict)

        return model

    # ── Dataset / split helpers ────────────────────────────────────────────────

    def _populate_dataset_and_split_idxs(self) -> None:
        """Return the full dataset together with train / val / test index lists.

        Fast path
            If ``dataset.pkl``, ``train_indices.json``, ``val_indices.json``,
            and ``test_indices.json`` are all present in the model directory,
            they are loaded and returned immediately.

        Slow path
            Otherwise the dataset is rebuilt from the run IDs in
            ``self._train_config`` and the split indices are re-derived
            deterministically and saved to the model directory so that
            subsequent calls take the fast path.
        """
        dataset_pkl = os.path.join(self._save_dir, "dataset.pkl")
        train_idx_file = os.path.join(self._save_dir, "train_indices.pkl")
        val_idx_file = os.path.join(self._save_dir, "val_indices.pkl")
        test_idx_file = os.path.join(self._save_dir, "test_indices.pkl")

        if all(
            os.path.exists(p)
            for p in (dataset_pkl, train_idx_file, val_idx_file, test_idx_file)
        ):
            self._dataset = ConcurrentQueryDataset.load_from(dataset_pkl)
            with open(train_idx_file, "rb") as f:
                self._idxs_for_split[DataSplit.TRAIN] = pickle.load(f)
            with open(val_idx_file, "rb") as f:
                self._idxs_for_split[DataSplit.VAL] = pickle.load(f)
            with open(test_idx_file, "rb") as f:
                self._idxs_for_split[DataSplit.TEST] = pickle.load(f)
            return

        if self._train_config is None:
            raise RuntimeError(
                "No train_config — cannot build dataset without run_ids."
            )
        self._dataset = ConcurrentQueryDataset.concatenate(
            [
                self.build_dataset_from_run_id(run_id=run_id)
                for run_id in self._train_config.run_ids
            ]
        )
        self._dataset.save_to(os.path.join(self._save_dir, "dataset.pkl"))

        # Derive split indices.
        explicit = self._train_config.explicit_run_ids_per_split
        train_idxs: list[int] = []
        val_idxs: list[int] = []
        test_idxs: list[int] = []
        if explicit is None:
            rng = np.random.default_rng(self._train_config.split_seed)
            indices = rng.permutation(len(self._dataset))
            n_val = round(self._train_config.val_frac * len(self._dataset))
            n_test = round(self._train_config.test_frac * len(self._dataset))
            train_idxs = list(indices[n_val + n_test :])
            val_idxs = list(indices[:n_val])
            test_idxs = list(indices[n_val : n_val + n_test])
        else:
            train_run_ids = set(explicit.get("train", []))
            val_run_ids = set(explicit.get("val", []))
            test_run_ids = set(explicit.get("test", []))
            for i, caqi in enumerate(self._dataset.cluster_aware_query_ids):
                run_id = Cluster.run_id_for_cluster_name(caqi.cluster_name)
                if run_id in train_run_ids:
                    train_idxs.append(i)
                elif run_id in val_run_ids:
                    val_idxs.append(i)
                elif run_id in test_run_ids:
                    test_idxs.append(i)
                else:
                    raise ValueError(
                        f"Query with run ID {run_id} not found in any explicit "
                        "run ID split list."
                    )
        self._idxs_for_split = {
            DataSplit.TRAIN: train_idxs,
            DataSplit.VAL: val_idxs,
            DataSplit.TEST: test_idxs,
        }

        # Save out.
        for name, idxs in [
            ("train_indices.pkl", train_idxs),
            ("val_indices.pkl", val_idxs),
            ("test_indices.pkl", test_idxs),
        ]:
            with open(os.path.join(self._save_dir, name), "wb") as f:
                pickle.dump(idxs, f)

        return

    def _extract_lower_bounds_from_batch(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch_size, seq_len, num_features)
        # Across the batch size dimension, for each element of
        # the sequence, extract the arrival time diff and sign
        # using IconqInteractionFeaturizer.arrival_time_diff_dim_idx
        # and IconqInteractionFeaturizer.sign_dim_idx. Then, of
        # the entries that have a nonzero sign, compute the
        # maximum arrival time diff per batch element. This
        # will be used to penalize the loss based on overlap.
        arrival_time_diffs = x[
            :,
            :,
            self._iconq_interaction_featurizer.arrival_time_diff_dim_idx,
        ]
        signs = x[
            :,
            :,
            self._iconq_interaction_featurizer.arrival_time_sign_dim_idx,
        ]
        max_arrival_time_diffs = []
        for i in range(arrival_time_diffs.shape[0]):
            diffs = arrival_time_diffs[i]
            sgns = signs[i]
            nonzero_diffs = diffs[sgns != 0]
            if len(nonzero_diffs) > 0:
                max_arrival_time_diffs.append(torch.max(nonzero_diffs).item())
            else:
                max_arrival_time_diffs.append(0.001)
        max_arrival_time_diffs_tensor = torch.tensor(
            max_arrival_time_diffs, device=self._device
        )
        return max_arrival_time_diffs_tensor

    def train(self) -> tuple[float, float]:
        """
        Trains the model and returns the final training and validation loss.

        Returns:
            final_train_loss: The final training loss.
            final_val_loss: The final validation loss.
        """
        self._nn.train()
        if self._train_config is None:
            raise RuntimeError(
                "No train_config set — cannot run training loop without a "
                "train_config."
            )
        train_config = self._train_config

        train_generator = torch.Generator()
        train_generator.manual_seed(
            train_config.training_dataloader_shuffle_seed
        )
        train_dataloader = DataLoader(
            Subset(self._dataset, self._idxs_for_split[DataSplit.TRAIN]),
            batch_size=train_config.batch_size,
            shuffle=True,
            generator=train_generator,
            collate_fn=ConcurrentQueryDataset.collate_and_pad,
        )

        optimizer = optim.Adam(
            self._nn.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        scheduler = ExponentialLR(
            optimizer, gamma=train_config.learning_rate_gamma
        )
        total_train_batches = len(train_dataloader)
        train_loss_trajectory = {}
        val_loss_trajectory = {}
        lr_trajectory = {}

        best_val_loss: dict[str, Any] = {
            "epoch": 0,
            "val_loss": float("inf"),
        }
        epochs_without_improvement = 0

        for epoch in range(1, train_config.num_epochs + 1):
            total_train_batch_loss = 0.0

            _console.print(Rule(f"Epoch {epoch}/{train_config.num_epochs}"))
            logger.info("Starting Epoch %d/%d", epoch, train_config.num_epochs)

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=_console,
            ) as progress:
                task = progress.add_task("Training", total=total_train_batches)
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    batch_loss, _ = self._process_batch(
                        batch=batch,
                        train_config=train_config,
                        derive_individual_predictions=False,
                    )

                    # Some batches can legitimately have no trainable signal
                    # (e.g., all isolated queries routed to stage model).
                    # In that case, skip gradient steps for the batch.
                    if not batch_loss.requires_grad:
                        total_train_batch_loss += batch_loss.item()
                        progress.advance(task)
                        continue

                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self._nn.parameters(), train_config.grad_clip_max_norm
                    )
                    optimizer.step()
                    total_train_batch_loss += batch_loss.item()
                    progress.advance(task)

            # Update bookkeeping
            lr_trajectory[epoch] = optimizer.param_groups[0]["lr"]
            scheduler.step()
            train_loss_trajectory[epoch] = (
                total_train_batch_loss / total_train_batches
            )
            val_loss_trajectory[epoch], val_errors = self.eval_on_split(
                split=DataSplit.VAL, out_filename=f"val_predictions_{epoch}.csv"
            )
            _, train_errors = self.eval_on_split(split=DataSplit.TRAIN)

            update_plots(
                self._save_dir,
                best_val_loss["epoch"],
                train_loss_trajectory,
                val_loss_trajectory,
                lr_trajectory,
                self._loss_type,
                mark_prev_checkpoint=True,
            )
            self._nn.train()

            # Print out progress
            train_loss = train_loss_trajectory[epoch]
            val_loss = val_loss_trajectory[epoch]
            lr = lr_trajectory[epoch]
            _console.print(
                f"train loss [bold]{train_loss:.6f}[/]  "
                f"val loss [bold]{val_loss:.6f}[/]  "
                f"lr [bold]{lr:.2e}[/]"
            )
            logger.info(
                "train loss %.6f  val loss %.6f  lr %.2e",
                train_loss,
                val_loss,
                lr,
            )

            print_errors_table(
                title=None,
                sets=[("train", train_errors), ("val", val_errors)],
            )
            logger.info(
                "errors: %s",
                {
                    k: f"{v:.4f}"
                    for k, v in {**train_errors, **val_errors}.items()
                },
            )

            # Check for early stopping
            if val_loss_trajectory[epoch] > (
                best_val_loss["val_loss"] - train_config.min_loss_improvement
            ):
                epochs_without_improvement += 1
                if epochs_without_improvement < train_config.patience:
                    continue
                else:
                    msg = (
                        f"Early stopping at epoch {epoch} after {train_config.patience} "
                        "epochs with no validation loss improvement."
                    )
                    _console.print(f"[yellow]{msg}[/]")
                    logger.info(msg)
                    update_plots(
                        self._save_dir,
                        best_val_loss["epoch"],
                        train_loss_trajectory,
                        val_loss_trajectory,
                        lr_trajectory,
                        self._loss_type,
                        mark_prev_checkpoint=True,
                    )
                    break
            epochs_without_improvement = 0

            # Save model.
            _console.print("Saving model...")
            logger.info("Saving model...")

            update_checkpoint(
                self._nn,
                self._save_dir,
            )

            # Update best validation loss
            best_val_loss["epoch"] = epoch
            best_val_loss["val_loss"] = val_loss_trajectory[epoch]

        best_epoch = best_val_loss["epoch"]
        final_train_loss = train_loss_trajectory[best_epoch]
        final_val_loss = val_loss_trajectory[best_epoch]

        update_plots(
            self._save_dir,
            best_val_loss["epoch"],
            train_loss_trajectory,
            val_loss_trajectory,
            lr_trajectory,
            self._loss_type,
            mark_prev_checkpoint=True,
        )

        # ── Final evaluation on best checkpoint ──────────────────────────────
        checkpoint_files = [
            f
            for f in os.listdir(self._save_dir)
            if f.startswith("model_") and f.endswith(".pth")
        ]
        if checkpoint_files:
            checkpoint_path = os.path.join(self._save_dir, checkpoint_files[0])
            self._load_nn_state_dict_with_feature_guard(
                torch.load(
                    checkpoint_path,
                    map_location=self._device,
                    weights_only=True,
                )
            )
        _, final_train_errors = self.eval_on_split(
            split=DataSplit.TRAIN, out_filename="final_train.csv"
        )
        _, final_val_errors = self.eval_on_split(
            split=DataSplit.VAL, out_filename="final_val.csv"
        )
        sets = [("train", final_train_errors), ("val", final_val_errors)]
        if len(self._idxs_for_split[DataSplit.TEST]) > 0:
            _, test_errors = self.eval_on_split(
                split=DataSplit.TEST, out_filename="final_test.csv"
            )
            sets.append(("test", test_errors))

        print_errors_table(
            title="[bold cyan]Final evaluation — best checkpoint[/]",
            sets=sets,
        )

        return final_train_loss, final_val_loss

    def eval_on_split(
        self,
        split: DataSplit,
        out_filename: Optional[str] = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluates the model on a subset of *dataset* identified by *indices*.

        Parameters:
            dataset: The full dataset.
            indices: Indices into *dataset* to evaluate on.
            train_config: Training configuration used to compute the loss.
            var_reg_weight: Weight for the variance regularisation term (NLL
                loss only).
            out_filename: If given, the filename (relative to the model's save
                dir) to save the predictions on the evaluated subset.

        Returns:
            mean_batch_loss: The mean batch loss on the evaluated subset.
            errors: Mean, median, 90th and 95th percentile error metrics.
        """
        indices = self._idxs_for_split[split]
        sliced_dataset = Subset(self._dataset, indices)
        return self.eval_on_dataset(sliced_dataset, out_filename=out_filename)

    def eval_on_dataset(
        self,
        dataset: ConcurrentQueryDataset | Subset,
        out_filename: Optional[str] = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluates the model on the given dataset.
        """
        train_config = self._train_config
        assert train_config is not None, "train_config required for eval"
        dataloader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=ConcurrentQueryDataset.collate_and_pad,
        )
        total_batches = len(dataloader)
        total_loss = 0.0
        all_pred_v_true: list[tuple] = []
        if self._nn.training:
            self._nn.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_loss, batch_pred_v_true = self._process_batch(
                    batch=batch,
                    train_config=train_config,
                    derive_individual_predictions=True,
                )
                all_pred_v_true.extend(batch_pred_v_true)
                total_loss += batch_loss.item()

        mean_batch_loss = total_loss / total_batches

        # Calculate per-query error metrics.
        rows = []
        for pred, y_true, caqi, qtid in all_pred_v_true:
            y_true_safe = max(float(y_true), 1e-9)
            y_pred_safe = max(pred.overall_mean_s(), 1e-9)
            md = pred.metadata or {}
            target_is_lower_bound = bool(md.get("target_is_lower_bound", False))

            factor_error = y_pred_safe / y_true_safe
            underprediction_error_s = max(y_true_safe - y_pred_safe, 0.0)

            q_error: Optional[float] = None
            abs_error_s: Optional[float] = None
            if not target_is_lower_bound:
                abs_error_s = abs(y_pred_safe - y_true_safe)
                q_error = max(factor_error, 1 / factor_error)

            row = {
                "query_id": caqi.query_id,
                "query_text_id": str(qtid),
                "run_id": str(md.get("run_id", "")),
                "cluster_name": caqi.cluster_name,
                "rpu": Cluster.rpu_for_cluster_name(caqi.cluster_name),
                "num_other_concurrent_queries": int(
                    md.get("num_other_concurrent_queries", 0)
                ),
                "y": y_true_safe,
                "y_pred_mean": y_pred_safe,
                "abs_error": abs_error_s,
                "q_error": q_error,
                "factor_error": factor_error,
                "underprediction_error_s": underprediction_error_s,
                "target_is_lower_bound": target_is_lower_bound,
                "model_source": str(md.get("model_source", "lstm")),
            }
            if "loss" in md:
                row["individual_loss"] = md["loss"]
            rows.append(row)
        df = pd.DataFrame(rows)

        # Compute aggregates.
        errors: dict[str, float] = {}
        for suffix, mask, cols in zip(
            ["normal", "aborted"],
            [~df["target_is_lower_bound"], df["target_is_lower_bound"]],
            [
                ["abs_error", "q_error", "factor_error"],
                ["underprediction_error_s", "factor_error"],
            ],
        ):
            subset = df.loc[mask, cols]

            errors[f"n_{suffix}"] = len(subset)
            if subset.empty:
                continue

            for col in cols:
                errors[f"mean_{col}_{suffix}"] = subset[col].mean()
                for p in [50, 90, 95]:
                    errors[f"p{p}_{col}_{suffix}"] = subset[col].quantile(
                        p / 100
                    )

        if out_filename is not None:
            df.sort_values("y", inplace=True, ascending=False)
            if not out_filename.endswith(".csv"):
                raise ValueError(
                    f"out_filename must end with .csv (got {out_filename})"
                )
            df.to_csv(os.path.join(self._save_dir, out_filename))

        return mean_batch_loss, errors

    def _process_batch(
        self, batch, train_config, derive_individual_predictions=False
    ):

        (
            x,
            x_len,
            pinch_points,
            y,
            cluster_aware_query_ids,
            query_text_ids,
            y_is_lower_bound,
        ) = batch

        batch_pred_v_true = []
        # List of (ModelPrediction, true_runtime, cluster_aware_query_id, query_text_id)
        # tuples for the batch.

        if train_config.use_stage_for_isolated_queries:
            # Identify isolated queries in the batch and predict them,
            # since we will use the stage model for them.
            non_isolated_indices = []

            for i in range(len(x_len)):
                if x_len[i].item() > 1:
                    non_isolated_indices.append(i)
                else:
                    pred = self._predict_isolated_query(
                        i,
                        x,
                        pinch_points,
                        cluster_aware_query_ids,
                        query_text_ids,
                        y_is_lower_bound,
                    )
                    y_true = (
                        torch.exp(y[i])
                        if self._trained_on_log_runtime
                        else y[i]
                    )
                    batch_pred_v_true.append(
                        (
                            pred,
                            y_true,
                            cluster_aware_query_ids[i],
                            query_text_ids[i],
                        )
                    )

            if len(non_isolated_indices) == 0:
                # All queries were isolated; nothing more to do.
                total_loss = torch.tensor(0.0, device=self._device)
                return total_loss, batch_pred_v_true

            x = x[non_isolated_indices]
            x_len = x_len[non_isolated_indices]
            pinch_points = pinch_points[non_isolated_indices]
            y = y[non_isolated_indices]
            y_is_lower_bound = y_is_lower_bound[non_isolated_indices]

        x, x_len, pinch_points, y, y_is_lower_bound = (
            x.to(self._device),
            x_len.to(self._device),
            pinch_points.to(self._device),
            y.to(self._device),
            y_is_lower_bound.to(self._device),
        )

        y_pred_mean, y_pred_logvar, y_pred_mix = self._nn(
            x,
            x_len,
            pinch_points,
            train_config.mdn_mix_softmax_temperature,
        )

        if self._trained_on_log_runtime:
            y = torch.exp(y)
            y_pred_mean = torch.exp(y_pred_mean)
            y_pred_logvar = torch.exp(y_pred_logvar)

        batch_loss: torch.Tensor
        loss_return_mean = not derive_individual_predictions

        if self._loss_type == LossType.MDN_NLL:
            loss = mdn_negative_log_likelihood_loss(
                y_pred_mean,
                y,
                y_pred_logvar,
                y_pred_mix,
                train_config.var_reg_weight,
                return_mean=loss_return_mean,
            )
            if not derive_individual_predictions:
                batch_loss = loss
            else:
                batch_loss = loss.mean()
                y_np = y.detach().cpu().numpy()
                for (caqid, pred), y_, t in zip(
                    self._build_predictions_from_nn_output(
                        y_pred_mean,
                        y_pred_logvar,
                        y_pred_mix,
                        x_len,
                        cluster_aware_query_ids,
                        query_text_ids,
                        y_is_lower_bound,
                    ).items(),
                    y_np,
                    query_text_ids,
                ):
                    batch_pred_v_true.append((pred, y_, caqid, t))
        elif self._loss_type == LossType.NLL:
            loss = negative_log_likelihood_loss(
                y_pred_mean,
                y,
                y_pred_logvar,
                train_config.var_reg_weight,
                return_mean=loss_return_mean,
            )

            if not derive_individual_predictions:
                batch_loss = loss
            else:
                batch_loss = loss.mean()
                y_np = y.detach().cpu().numpy()
                for (caqid, pred), y_, t in zip(
                    self._build_predictions_from_nn_output(
                        y_pred_mean,
                        y_pred_logvar,
                        y_pred_mix,
                        x_len,
                        cluster_aware_query_ids,
                        query_text_ids,
                        y_is_lower_bound,
                    ).items(),
                    y_np,
                    query_text_ids,
                ):
                    batch_pred_v_true.append((pred, y_, caqid, t))
        else:
            min_val: float | torch.Tensor = 0.001
            if train_config.penalize_based_on_overlap:
                min_val = self._extract_lower_bounds_from_batch(x)

            loss = sensitive_q_error_loss(
                input=y_pred_mean,
                target=y,
                target_is_lower_bound=y_is_lower_bound,
                small_val=train_config.sensitive_q_error_loss_small_val,
                min_val=min_val,
                sensitive_q_error_loss_version=train_config.sensitive_q_error_loss_version,
                return_mean=loss_return_mean,
            )

            if not derive_individual_predictions:
                batch_loss = loss
            else:
                batch_loss = loss.mean()
                y_np = y.detach().cpu().numpy()
                for (caqid, pred), y_, t in zip(
                    self._build_predictions_from_nn_output(
                        y_pred_mean,
                        y_pred_logvar,
                        y_pred_mix,
                        x_len,
                        cluster_aware_query_ids,
                        query_text_ids,
                        y_is_lower_bound,
                        per_item_losses=loss.detach().cpu().numpy(),
                    ).items(),
                    y_np,
                    query_text_ids,
                ):
                    batch_pred_v_true.append((pred, y_, caqid, t))

        return (
            batch_loss,
            batch_pred_v_true,
        )

    @staticmethod
    def optimized_load_final_dfs_per_split(
        model_id: str,
    ) -> dict[DataSplit, pd.DataFrame]:
        split_dfs: dict[DataSplit, pd.DataFrame] = {}
        save_dir = Path(IconqModel.default_save_dir(model_id))
        model: Optional[IconqModel] = None
        for split in DataSplit:
            split_str = split.value
            csv_path = save_dir / f"final_{split_str}.csv"
            parquet_path = save_dir / f"final_{split_str}.parquet"
            loaded_from_disk = False
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                loaded_from_disk = True
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
                loaded_from_disk = True
            else:
                if model is None:
                    model = IconqModel.load(model_id, inference_mode=True)
                model.eval_on_split(split=split, out_filename=str(csv_path))
                df = pd.read_csv(csv_path)

            # Backfill cluster_rpu metadata for older final_*.csv/parquet files.
            if "rpu" not in df.columns or "model_source" not in df.columns:
                if model is None:
                    model = IconqModel.load(model_id, inference_mode=True)
                model.eval_on_split(split=split, out_filename=str(csv_path))
                df = pd.read_csv(csv_path)

            split_dfs[split] = df
            if loaded_from_disk or not parquet_path.exists():
                split_dfs[split].to_parquet(parquet_path)
        return split_dfs

    def build_dataset_from_run_id(
        self,
        run_id: str,
    ) -> ConcurrentQueryDataset:
        """Build a ConcurrentQueryDataset from a Trace for IconqModel training.

        When use_client_side_latencies=True, timing windows, latency targets, and
        is_lower_bound are sourced from the structured log.  Raises ValueError if
        no structured_log.parquet is present for the run.
        """
        trace = Trace(run_id)
        cluster_aware_query_ids = trace.cluster_aware_query_ids
        query_text_ids = trace.query_text_ids

        if not cluster_aware_query_ids:
            return ConcurrentQueryDataset.build_from_query_groups(
                iconq_interaction_featurizer=self._iconq_interaction_featurizer,
                cluster_to_base_to_neighbors={},
            )

        if self._train_config and self._train_config.use_client_side_latencies:
            arrival_s = trace.client_side_arrival_times_s
            completion_s = trace.client_side_completion_times_s
            latencies = trace.client_side_latencies_s
            query_success = trace.structured_log.query_success()  # type: ignore[union-attr]
            reference_s = float(arrival_s.min())

            def _start(qid: ClusterAwareQueryId) -> float:
                return float(arrival_s[qid]) - reference_s

            def _end(qid: ClusterAwareQueryId) -> float:
                return float(completion_s[qid]) - reference_s

            def _latency(qid: ClusterAwareQueryId) -> float:
                return float(latencies[qid])

            def _is_lb(qid: ClusterAwareQueryId) -> bool:
                return not bool(query_success.get(qid.query_id, False))

        else:
            arrival_times = trace.arrival_times()
            completion_times = trace.completion_times()
            was_aborted = trace.was_aborted()
            reference_ts = min(
                arrival_times[qid].timestamp()
                for qid in cluster_aware_query_ids
            )

            def _start(qid: ClusterAwareQueryId) -> float:  # type: ignore[misc]
                return arrival_times[qid].timestamp() - reference_ts

            def _end(qid: ClusterAwareQueryId) -> float:  # type: ignore[misc]
                return completion_times[qid].timestamp() - reference_ts

            def _latency(qid: ClusterAwareQueryId) -> float:  # type: ignore[misc]
                return (
                    completion_times[qid] - arrival_times[qid]
                ).total_seconds()

            def _is_lb(qid: ClusterAwareQueryId) -> bool:  # type: ignore[misc]
                return bool(was_aborted[qid])

        # Build one IntervalTree per cluster. Each interval's data payload is the
        # fully-featurized Query object, so _find_neighbors can return Query
        # objects directly without a second lookup.
        interval_trees: dict[str, IntervalTree] = defaultdict(IntervalTree)

        for cluster_aware_query_id in cluster_aware_query_ids:
            cluster_name = cluster_aware_query_id.cluster_name
            query_text_id = query_text_ids[cluster_aware_query_id]
            start_s = _start(cluster_aware_query_id)
            end_s = _end(cluster_aware_query_id)
            query = Query(
                query_id=cluster_aware_query_id.query_id,
                query_text_id=query_text_id,
                rel_start_time_s=start_s,
                featurization=self._iconq_query_featurizer.featurize_from_query_text_id(
                    query_text_id
                ),
                # Pre-compute stage-model predictions for every allowed RPU size so
                # build_from_query_groups can look up the right one per cluster.
                stage_predictions_per_rpu={
                    rpu: float(
                        self._stage_model.predict_from_query_text_id(
                            {cluster_aware_query_id: query_text_id},
                            cluster_rpu=rpu,
                        )[cluster_aware_query_id].overall_mean_s()
                    )
                    for rpu in Cluster.ALL_ALLOWED_RPU_SIZES
                },
            )
            interval_trees[cluster_name].add(Interval(start_s, end_s, query))

        assert self._train_config is not None
        cluster_to_base_to_neighbors = self._find_neighbors(
            interval_trees,
            self._init_config.use_fixed_window_radius_s,
            self._init_config.use_fixed_window_max_neighbors_per_side,
            (
                _is_lb
                if self._train_config.ignore_aborted_queries
                else lambda qid: False
            ),
        )

        targets: dict[ClusterAwareQueryId, float] = {}
        is_lower_bound: dict[ClusterAwareQueryId, bool] = {}
        for qid in cluster_aware_query_ids:
            is_lb = _is_lb(qid)
            if self._train_config.ignore_aborted_queries and is_lb:
                continue
            targets[qid] = _latency(qid)
            is_lower_bound[qid] = is_lb

        return ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=self._iconq_interaction_featurizer,
            cluster_to_base_to_neighbors=cluster_to_base_to_neighbors,
            targets=targets,
            is_lower_bound=is_lower_bound,
            use_log_runtime=self.trained_on_log_runtime,
            censored_observation_sample_prob=(
                self._train_config.neighbor_derived_censored_observation_prob
            ),
            censored_observation_rng_seed=(
                self._train_config.neighbor_derived_censored_observation_seed
            ),
        )

    @staticmethod
    def _find_neighbors(
        interval_trees: dict[str, IntervalTree],
        use_fixed_window_radius_s: Optional[float],
        use_fixed_window_max_neighbors_per_side: Optional[int],
        ignore_as_base: Callable[[ClusterAwareQueryId], bool],
    ) -> dict[str, dict[Query, list[Query]]]:
        """For each cluster, map every query to its ordered list of neighbors.

        Two neighbor strategies:
        - Overlap-based (use_fixed_window_radius_s is None): neighbors are queries
        whose execution interval overlaps with the base query's interval.
        - Fixed-window: neighbors are queries whose *start time* falls within
        ±use_fixed_window_radius_s of the base query's start time, optionally
        capped to use_fixed_window_max_neighbors_per_side on each side.

        Queries for which ignore_as_base returns True are excluded from the
        base-query set but are still kept as neighbors of other queries.
        """
        result: dict[str, dict[Query, list[Query]]] = {}

        for cluster_name, tree in interval_trees.items():
            base_to_neighbors: dict[Query, list[Query]] = {}

            for iv in sorted(tree, key=lambda x: x.begin):
                neighbor_ivs: list[Interval] = []

                if ignore_as_base(
                    ClusterAwareQueryId.make(cluster_name, iv.data.query_id)
                ):
                    continue

                if use_fixed_window_radius_s is None:
                    # Overlap-based: any interval that shares time with [begin, end).
                    neighbor_ivs = sorted(
                        [b for b in tree.overlap(iv.begin, iv.end) if b != iv],
                        key=lambda b: (b.begin, b.end),
                    )
                else:
                    # Fixed-window: split into before/after by start-time proximity.
                    neighbors_before = sorted(
                        [
                            b
                            for b in tree
                            if b.begin < iv.begin
                            and b.begin >= iv.begin - use_fixed_window_radius_s
                        ],
                        key=lambda b: (b.begin, b.end),
                    )
                    neighbors_after = sorted(
                        [
                            b
                            for b in tree
                            if b.begin > iv.begin
                            and b.begin <= iv.begin + use_fixed_window_radius_s
                        ],
                        key=lambda b: (b.begin, b.end),
                    )
                    if use_fixed_window_max_neighbors_per_side is not None:
                        # Keep the N closest on each side.
                        neighbors_before = neighbors_before[
                            -use_fixed_window_max_neighbors_per_side:
                        ]
                        neighbors_after = neighbors_after[
                            :use_fixed_window_max_neighbors_per_side
                        ]
                    neighbor_ivs = neighbors_before + neighbors_after

                base_to_neighbors[iv.data] = [b.data for b in neighbor_ivs]

            result[cluster_name] = base_to_neighbors

        return result
