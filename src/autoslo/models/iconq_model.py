import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import torch
import yaml
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
from autoslo.models.iconq_dataset_builder import build_dataset_from_trace
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
from autoslo.workload_definition.query import ClusterAwareQueryId
from autoslo.workload_execution.trace import Trace

logger = logging.getLogger(__name__)

_console = Console()


def _print_errors_table(
    title: str,
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
    for set_idx, (set_name, errs) in enumerate(sets):
        is_last_set = set_idx == n_sets - 1
        for suffix in ["normal", "aborted", "all"]:

            def _f(k: str, _e: dict = errs) -> str:  # default arg captures errs
                v = _e.get(k)
                return f"{v:.4f}" if v is not None else "-"

            n = int(errs.get(f"n_{suffix}", 0))
            if n == 0:
                continue
            label = f"{set_name} / {suffix} (N={n})"
            table.add_row(
                label,
                "q-error",
                _f(f"mean_q_error_{suffix}"),
                _f(f"p50_q_error_{suffix}"),
                _f(f"p90_q_error_{suffix}"),
                _f(f"p95_q_error_{suffix}"),
                style="white",
            )
            table.add_row(
                "",
                "abs error",
                _f(f"mean_abs_error_{suffix}"),
                _f(f"p50_abs_error_{suffix}"),
                _f(f"p90_abs_error_{suffix}"),
                _f(f"p95_abs_error_{suffix}"),
                style="dim",
                end_section=True,
            )
        # Double horizontal line between splits: the abs-error row above already
        # added one rule via end_section; a blank spacer row with end_section
        # adds a second immediately adjacent rule.
        if not is_last_set:
            table.add_row(*_EMPTY_ROW, end_section=True)

    _console.print(Rule(title))
    _console.print(table)


class IconqModel:
    """
    A query runtime model that uses an LSTM to predict query runtimes.
    Optionally, it can also predict the uncertainty of the predictions.
    """

    def __init__(
        self,
        init_config: IconqModelInitConfig,
        train_config: Optional[IconqModelTrainConfig] = None,
        device: torch.device = torch.device("cpu"),
        parent_save_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        _skip_save: bool = False,
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
        """
        self._device = device
        self._init_config = init_config
        self._train_config: Optional[IconqModelTrainConfig] = train_config

        # Create save directory and set model ID.
        if model_id is None:
            model_id = str(int(datetime.now().timestamp()))
        self._model_id = model_id
        if parent_save_dir is None:
            parent_save_dir = os.path.join(pu.get_data_path(), "iconq_models")
        self._parent_save_dir = parent_save_dir
        self._save_dir = os.path.join(parent_save_dir, self._model_id)
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

        # Save initial model parameters (skip when loading from disk to
        # avoid a write that races with concurrent readers).
        if not _skip_save:
            self._save_params()

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

        # Pre-compute run_id once per unique cluster name so that
        # Cluster.run_id_for_cluster_name (which calls str.split) is not
        # invoked once per prediction inside the hot loop.
        unique_cluster_names = {
            caqi.cluster_name for caqi in cluster_aware_query_ids
        }
        run_id_by_cluster: dict[str, str] = {
            cluster_name: Cluster.run_id_for_cluster_name(cluster_name)
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

    @staticmethod
    def load(
        model_id: str, parent_load_dir: Optional[str] = None
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
            _skip_save=True,
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
            model._nn.load_state_dict(state_dict)

        return model

    @staticmethod
    def print_performance_tables(
        model_ids: list[str],
        parent_load_dir: Optional[str] = None,
    ) -> None:
        """Print train / val / test performance tables for one or more saved models.

        Each model must have been trained with ``save_dataset=True`` so that
        ``dataset.pkl``, ``train_indices.json``, ``val_indices.json``, and
        ``test_indices.json`` are present in its model directory.

        Parameters:
            model_ids: Identifiers of the models to evaluate.
            parent_load_dir: Parent directory that contains the model
                subdirectories.  Defaults to ``data/iconq_models/``.
        """
        for model_id in model_ids:
            model = IconqModel.load(model_id, parent_load_dir)
            save_dir = model._save_dir
            if model._train_config is None:
                raise RuntimeError(
                    f"Model '{model_id}' has no train_config — cannot print "
                    f"performance tables."
                )
            train_config = model._train_config

            required = {
                "dataset.pkl": os.path.join(save_dir, "dataset.pkl"),
                "train_indices.json": os.path.join(
                    save_dir, "train_indices.json"
                ),
                "val_indices.json": os.path.join(save_dir, "val_indices.json"),
                "test_indices.json": os.path.join(
                    save_dir, "test_indices.json"
                ),
            }
            missing = [
                name
                for name, path in required.items()
                if not os.path.exists(path)
            ]
            if missing:
                raise FileNotFoundError(
                    f"Model '{model_id}' is missing: {', '.join(missing)}. "
                    "Re-train with save_dataset=True to enable performance tables."
                )

            dataset = ConcurrentQueryDataset.load_from(required["dataset.pkl"])
            with open(required["train_indices.json"]) as f:
                train_indices: list[int] = json.load(f)
            with open(required["val_indices.json"]) as f:
                val_indices: list[int] = json.load(f)
            with open(required["test_indices.json"]) as f:
                test_indices: list[int] = json.load(f)

            batch_size = train_config.batch_size
            make_dl = lambda idxs: DataLoader(
                Subset(dataset, idxs),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=ConcurrentQueryDataset.collate_and_pad,
            )
            _, train_errors = model._validate(
                make_dl(train_indices), train_config
            )
            _, val_errors = model._validate(make_dl(val_indices), train_config)
            _, test_errors = model._validate(
                make_dl(test_indices), train_config
            )

            _print_errors_table(
                title=f"[bold]{model_id}[/]",
                sets=[
                    ("train", train_errors),
                    ("val", val_errors),
                    ("test", test_errors),
                ],
            )

    def _get_dataloaders(  # pylint: disable=too-many-locals
        self,
        dataset: ConcurrentQueryDataset,
        train_config: IconqModelTrainConfig,
        split: bool = True,
        save_dataset: bool = False,
    ) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Converts the given dataset into DataLoaders for training, validation,
        and testing.

        Parameters:
            dataset: The dataset to convert.
            train_config: The training configuration for the LSTM model.
            split: Whether to split the data into training and validation sets.
            save_dataset: Whether to save the created dataset to disk.

        Returns:
            train_dataloader: The DataLoader for the training set, or the sole
                dataloader if split is False.
            val_dataloader: The DataLoader for the validation set, or None if
                split is False.
            test_dataloader: The DataLoader for the test set, or None if
                split is False.
        """

        if not split:
            dataloader = DataLoader(
                dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                collate_fn=ConcurrentQueryDataset.collate_and_pad,
            )
            return dataloader, None, None

        if train_config.explicit_run_ids_per_split is None:
            train_idxs, val_idxs, test_idxs = self._get_data_splits(
                len(dataset), train_config
            )
            train_dataset = Subset(dataset, train_idxs)
            val_dataset = Subset(dataset, val_idxs)
            test_dataset = Subset(dataset, test_idxs)
        else:
            explicit_run_ids_per_split = train_config.explicit_run_ids_per_split
            train_run_ids = explicit_run_ids_per_split.get("train", [])
            val_run_ids = explicit_run_ids_per_split.get("val", [])
            test_run_ids = explicit_run_ids_per_split.get("test", [])
            train_indices = []
            val_indices = []
            test_indices = []
            for i, cluster_aware_query_id in enumerate(
                dataset.cluster_aware_query_ids
            ):
                cluster_name = cluster_aware_query_id.cluster_name
                run_id = Cluster.run_id_for_cluster_name(cluster_name)
                if run_id in train_run_ids:
                    train_indices.append(i)
                elif run_id in val_run_ids:
                    val_indices.append(i)
                elif run_id in test_run_ids:
                    test_indices.append(i)
                else:
                    raise ValueError(
                        f"Query with run ID {run_id} not found in any explicit "
                        f"run ID split list."
                    )

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

            if save_dataset:
                dataset.save_to(os.path.join(self._save_dir, "dataset.pkl"))
                with open(
                    os.path.join(self._save_dir, "train_indices.json"), "w"
                ) as f:
                    json.dump(train_indices, f)
                with open(
                    os.path.join(self._save_dir, "val_indices.json"), "w"
                ) as f:
                    json.dump(val_indices, f)
                with open(
                    os.path.join(self._save_dir, "test_indices.json"), "w"
                ) as f:
                    json.dump(test_indices, f)

        train_generator = torch.Generator()
        train_generator.manual_seed(
            train_config.training_dataloader_shuffle_seed
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            generator=train_generator,
            collate_fn=ConcurrentQueryDataset.collate_and_pad,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=ConcurrentQueryDataset.collate_and_pad,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=ConcurrentQueryDataset.collate_and_pad,
        )

        return train_dataloader, val_dataloader, test_dataloader

    def _get_data_splits(  # pylint: disable=too-many-arguments
        self,
        data_size: int,
        train_config: IconqModelTrainConfig,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Splits the data into training, validation, and testing sets.

        Parameters:
            data_size: The size of the data to split.
            train_config: The training configuration for the LSTM model.

        Returns:
            train_idxs: The indices of the training set.
            val_idxs: The indices of the validation set.
            test_idxs: The indices of the testing set.
        """

        rng = np.random.default_rng(train_config.split_seed)
        indices = rng.permutation(data_size)

        n_val = round(train_config.val_frac * data_size)
        n_test = round(train_config.test_frac * data_size)

        val_idx = indices[:n_val]
        test_idx = indices[n_val : n_val + n_test]
        train_idx = indices[n_val + n_test :]

        return list(train_idx), list(val_idx), list(test_idx)

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

    def _run_training_loop(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None,
    ) -> tuple[float, float]:
        """
        Runs the training loop for the model and returns the final training and
        validation loss.

        Parameters:
            train_dataloader: The DataLoader for the training set.
            val_dataloader: The DataLoader for the validation set.
            test_dataloader: Optional DataLoader for the held-out test set.
                If provided, the best checkpoint is reloaded at the end and
                evaluated; results are printed as a summary table.
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
            val_loss_trajectory[epoch], val_errors = self._validate(
                dataloader=val_dataloader,
                var_reg_weight=train_config.var_reg_weight,
                training_dir=self._save_dir,
                epoch=epoch,
                train_config=train_config,
            )
            _, train_errors = self._validate(
                dataloader=train_dataloader,
                var_reg_weight=train_config.var_reg_weight,
                training_dir=None,
                epoch=None,
                train_config=train_config,
            )
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

            _print_errors_table(
                title=f"Epoch {epoch} — errors",
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

        # ── Final evaluation on all sets (best checkpoint) ────────────────────
        if test_dataloader is not None and len(test_dataloader) > 0:
            # Reload best checkpoint so we score the saved model, not the
            # last in-memory weights (which may be from a post-best epoch).
            checkpoint_files = [
                f
                for f in os.listdir(self._save_dir)
                if f.startswith("model_") and f.endswith(".pth")
            ]
            if checkpoint_files:
                checkpoint_path = os.path.join(
                    self._save_dir, checkpoint_files[0]
                )
                self._nn.load_state_dict(
                    torch.load(
                        checkpoint_path,
                        map_location=self._device,
                        weights_only=True,
                    )
                )
            _, final_train_errors = self._validate(
                dataloader=train_dataloader,
                var_reg_weight=train_config.var_reg_weight,
                training_dir=None,
                epoch=None,
                train_config=train_config,
            )
            _, final_val_errors = self._validate(
                dataloader=val_dataloader,
                var_reg_weight=train_config.var_reg_weight,
                training_dir=None,
                epoch=None,
                train_config=train_config,
            )
            _, test_errors = self._validate(
                dataloader=test_dataloader,
                var_reg_weight=train_config.var_reg_weight,
                training_dir=None,
                epoch=None,
                train_config=train_config,
            )
            _print_errors_table(
                title="[bold cyan]Final evaluation — best checkpoint[/]",
                sets=[
                    ("train", final_train_errors),
                    ("val", final_val_errors),
                    ("test", test_errors),
                ],
            )

        return final_train_loss, final_val_loss

    def _validate(
        self,
        dataloader: DataLoader,
        train_config: IconqModelTrainConfig,
        var_reg_weight: float = 0.0,
        training_dir: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluates the model on the validation set.

        Parameters:
            dataloader: The DataLoader to validate on.
            var_reg_weight: The weight for the variance regularization term for the negative
                log likelihood loss.
            training_dir: The directory where the training artifacts are saved.
            training_dir: The directory for the model training. If given, will save the validation
                predictions to a CSV file.
            epoch: The epoch number. If given and training_dir is given, will include the epoch
                number in the filename of the validation predictions CSV file.

        Returns:
            mean_val_batch_loss: The mean batch loss on the validation set.
            errors: Mean, median, 90th percentile, and 95th percentile error metrics on the
                validation set. The exact metrics depend on the loss type.
        """

        total_batches = len(dataloader)
        total_batch_loss = 0.0
        all_pred_v_true = []
        self._nn.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_loss, batch_pred_v_true = self._process_batch(
                    batch=batch,
                    train_config=train_config,
                    derive_individual_predictions=True,
                )
                all_pred_v_true.extend(batch_pred_v_true)
                total_batch_loss += batch_loss.item()

        mean_val_batch_loss = total_batch_loss / total_batches

        # Calculate error metrics
        errors: dict[str, float] = {}
        # (ModelPrediction, true_runtime, cluster_aware_query_id, query_text_id)
        abs_error = [
            abs(pred.overall_mean_s() - true)
            for pred, true, _, _ in all_pred_v_true
        ]
        q_error = [
            max(pred.overall_mean_s() / true, true / pred.overall_mean_s())
            for pred, true, _, _ in all_pred_v_true
        ]
        was_aborted = [
            pred.metadata.get("target_is_lower_bound", False)
            for pred, _, _, _ in all_pred_v_true
        ]

        for suffix, condition in [
            ("normal", lambda ab: not ab),
            ("aborted", lambda ab: ab),
            ("all", lambda ab: True),
        ]:
            filtered_abs_error = [
                ae for ae, ab in zip(abs_error, was_aborted) if condition(ab)
            ]
            filtered_q_error = [
                qe for qe, ab in zip(q_error, was_aborted) if condition(ab)
            ]
            errors[f"n_{suffix}"] = len(filtered_abs_error)
            if not filtered_abs_error:
                continue

            errors[f"mean_abs_error_{suffix}"] = np.mean(filtered_abs_error)
            errors[f"mean_q_error_{suffix}"] = np.mean(filtered_q_error)
            for p in [50, 90, 95]:
                errors[f"p{p}_abs_error_{suffix}"] = np.percentile(
                    filtered_abs_error, p
                )
                errors[f"p{p}_q_error_{suffix}"] = np.percentile(
                    filtered_q_error, p
                )

        # Print out a dataframe of predictions
        if training_dir is not None:
            val_df = pd.DataFrame()
            val_df["query_id"] = [
                cluster_aware_query_id.query_id
                for _, _, cluster_aware_query_id, _ in all_pred_v_true
            ]
            val_df["num_other_concurrent_queries"] = [
                pred.metadata["num_other_concurrent_queries"]
                for pred, _, _, _ in all_pred_v_true
            ]
            val_df["y"] = [true for _, true, _, _ in all_pred_v_true]
            val_df["y_pred"] = [pred for pred, _, _, _ in all_pred_v_true]
            val_df["target_is_lower_bound"] = [
                pred.metadata["target_is_lower_bound"]
                for pred, _, _, _ in all_pred_v_true
            ]
            val_df["query_text_id"] = [t for _, _, _, t in all_pred_v_true]
            val_df["abs_error"] = abs_error
            val_df["q_error"] = q_error
            val_df["run_id"] = [
                pred.metadata["run_id"] for pred, _, _, _ in all_pred_v_true
            ]
            if "loss" in all_pred_v_true[0][0].metadata:
                val_df["individual_loss"] = [
                    pred.metadata["loss"] for pred, _, _, _ in all_pred_v_true
                ]

            val_df.sort_values("y", inplace=True, ascending=False)
            suffix = f"_{epoch}" if epoch is not None else ""
            val_df.to_csv(
                os.path.join(training_dir, f"val_predictions{suffix}.csv")
            )

        return mean_val_batch_loss, errors

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
                    batch_pred_v_true.append(
                        (
                            pred,
                            y[i],
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
