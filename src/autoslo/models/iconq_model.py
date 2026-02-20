import logging
import os
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.model_training.iconq_model_training_checkpoint import (
    update_checkpoint,
    update_plots,
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

logger = logging.getLogger(__name__)


@dataclass
class IconqModelInitConfig:
    """
    A dataclass for the configuration of the Iconq model at initialization.
    """

    iconq_query_featurizer_id: Optional[str] = None
    iconq_query_featurizer_init_params: Optional[dict[str, Any]] = None
    stage_model_id: Optional[str] = None
    stage_model_init_params: Optional[dict[str, Any]] = None

    featurizer_num_operators: int = (
        10  # The number of operators to consider in the query featurizer.
    )
    featurizer_num_tables: int = (
        10  # The number of tables to consider in the query featurizer.
    )
    embedding_size: int = 64  # The size of the embedding layer.
    lstm_hidden_size: int = 128  # The size of the hidden layer in the LSTM.
    lstm_num_layers: int = 1  # The number of layers in the LSTM.
    lstm_dropout: float = 0.2  # The dropout rate in the LSTM.
    is_bayesian: bool = True  # Whether the model is bayesian.
    bayesian_samples: int = (
        5  # The number of samples to take in the bayesian model.
    )
    is_mdn: bool = False  # Whether the model is an MDN model.
    mdn_num_gaussians: int = (
        3  # The number of Gaussian components in the mixture.
    )
    train_on_log_runtime: bool = (
        False  # Whether the model will be trained on the log of the query
        # runtime, as opposed to the runtime itself.
    )

    use_fixed_window_radius_s: Optional[float] = (
        None  # The fixed window radius in seconds for selecting neighboring queries.
    )
    use_fixed_window_max_neighbors_per_side: Optional[int] = (
        None  # The maximum number of neighboring queries to consider on each side when using a fixed window.
    )

    ignore_cluster_size: bool = (
        False  # Whether to ignore the cluster size when featurizing queries.
    )


@dataclass
class NNModelTrainConfig:
    """
    A dataclass for the configuration of the NN-based models at training.
    """

    run_ids: list[str] = field(
        default_factory=list
    )  # The run IDs to use for training the model.

    split_seed: int = (
        42  # The seed for the random split of the data into training, validation and testing sets.
        # Unused if both val_split_type and test_split_type are "temporal".
    )
    training_dataloader_shuffle_seed: int = (
        42  # The seed for shuffling the training DataLoader.
    )
    test_frac: float = (
        0.25  # The fraction of the data to use for testing (i.e. ignore for training).
    )
    test_split_type: str = (
        "random"  # The type of split to use for the testing set.
    )
    test_split_per_trace: bool = (
        True  # Whether to split the test set per trace.
    )
    test_split_phases: Optional[list[str]] = (
        None  # The phases to use for the test set.
    )
    val_frac: float = 0.1  # The fraction of the data to use for validation.
    val_split_type: str = (
        "random"  # The type of split to use for the validation set.
    )

    learning_rate: float = 5e-3  # The learning rate for the optimizer.
    learning_rate_gamma: float = 0.9  # The learning rate decay factor.
    weight_decay: float = 2e-5  # The weight decay for the optimizer.
    var_reg_weight: float = (
        0.01  # The weight for the variance regularization term.
    )

    num_epochs: int = 100  # The number of epochs to train for.
    batch_size: int = 32  # The batch size for the DataLoader.
    grad_clip_max_norm: float = 2.0  # The maximum norm for gradient clipping.

    patience: int = (
        5  # The number of epochs to wait for improvement before early stopping.
    )
    min_loss_improvement: float = (
        1e-3  # The minimum improvement in validation loss to consider as an improvement for early
        # stopping.
    )

    optuna_trials: int = (
        100  # The number of trials to run for hyperparameter optimization.
    )

    mdn_mix_softmax_temperature: float = (
        1.0  # The softmax temperature for the mixture weights.
    )

    use_stage_for_isolated_queries: bool = (
        False  # Whether to use the stage model for isolated queries.
    )

    explicit_run_ids_per_split: Optional[dict[str, list[str]]] = (
        None  # Explicitly specify run IDs per split (train/val/test).
    )

    sensitive_q_error_loss_small_val: float = (
        5.0  # The small value for the sensitive Q-error loss.
    )

    penalize_based_on_overlap: bool = (
        False  # Whether to penalize based on overlap in the sensitive Q-error loss.
    )

    sensitive_q_error_loss_version: int = (
        1  # The version of the sensitive Q-error loss to use.
    )


class IconqModel:
    """
    A query runtime model that uses an LSTM to predict query runtimes.
    Optionally, it can also predict the uncertainty of the predictions.
    """

    def __init__(
        self,
        init_config: IconqModelInitConfig,
        train_config_sequence: Optional[list[NNModelTrainConfig]] = None,
        device: torch.device = torch.device("cpu"),
        parent_save_dir: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        """
        Initializes the LSTM model.

        Parameters:
            init_config: The configuration for the LSTM model.
            train_config_sequence: The sequence of training configurations
                that have been used to train the model so far.
            device: The device to use for training and prediction.
            parent_save_dir: The parent directory to save the model.
            model_id: The identifier of the model. If None, a new model ID
                will be generated.
        """

        self._device = device
        self._init_config = init_config
        self._train_config_sequence: list[NNModelTrainConfig] = (
            [] if train_config_sequence is None else train_config_sequence
        )

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
                init_config.iconq_query_featurizer_id
            )
        self._iconq_interaction_featurizer = IconqInteractionFeaturizer(
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

        # Save initial model parameters.
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
    ) -> dict[str, dict[str, ModelPrediction]]:
        """
        Predicts the runtimes for the queries in the given dataset.

        Parameters:
            dataset: The dataset to predict runtimes for.
            inference_batch_size: Maximum number of items to process per
                forward pass. Defaults to 512, which handles typical simulator
                datasets in one shot while remaining memory-safe.

        Returns:
            A dictionary mapping (run_id, query_id) tuples to their predicted ModelPrediction.
        """

        predictions: dict[str, dict[str, ModelPrediction]] = defaultdict(dict)
        n = len(dataset)
        if n == 0:
            return predictions

        self._nn.eval()
        with torch.no_grad():
            # Bypass the DataLoader machinery entirely: directly collate
            # fixed-size slices and pass them to a loss-free inference method.
            # This eliminates per-batch DataLoader overhead (iterator init,
            # collate_fn wrapping) and avoids computing losses during inference.
            for start in range(0, n, inference_batch_size):
                end = min(start + inference_batch_size, n)
                batch = ConcurrentQueryDataset.collate_and_pad(
                    [dataset[i] for i in range(start, end)]
                )
                for prediction, run_id, query_id in self._infer_batch(batch):
                    predictions[run_id][query_id] = prediction

        return predictions

    def _predict_isolated_query(
        self,
        i: int,
        x: torch.Tensor,
        pinch_points: torch.Tensor,
        query_ids: list[str],
        tpcds_temp_and_q_idxs: list,
        run_ids: list[str],
        y_is_lower_bound: torch.Tensor,
    ) -> ModelPrediction:
        """
        Predict a single isolated query (sequence length == 1) via the stage
        model and wrap the result in a ModelPrediction.
        """
        rpu = x[i][pinch_points[i]][
            self._iconq_interaction_featurizer.rpu_dim_idx
        ].item()
        cluster_name = Cluster.ordered_cluster_names_per_rpu()[int(rpu)][0]
        pred = self.stage_model.predict_from_tpcds_temp_and_q_idx(
            {query_ids[i]: tpcds_temp_and_q_idxs[i]},
            cluster_name=cluster_name,
        )[query_ids[i]]
        return ModelPrediction(
            mean_s=pred.mean_s,
            std_dev_s=pred.std_dev_s,
            mix_coeffs=pred.mix_coeffs,
            metadata={
                "num_other_concurrent_queries": 0,
                "run_id": run_ids[i],
                "tpcds_temp_and_q_idx": tpcds_temp_and_q_idxs[i],
                "query_id": query_ids[i],
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
        query_ids: list[str],
        tpcds_temp_and_q_idxs: list,
        run_ids: list[str],
        y_is_lower_bound: torch.Tensor,
        per_item_losses: Optional[np.ndarray] = None,
    ) -> list[tuple[ModelPrediction, str, str]]:
        """
        Convert NN output tensors into a list of (ModelPrediction, run_id, query_id)
        tuples. Handles all three loss types.

        Parameters:
            per_item_losses: Per-sample loss values (only used for
                LossType.SENSITIVE_Q_ERROR metadata). Pass None (default) during
                inference — the metadata field will be set to 0.0.
        """
        x_len_np = x_len.detach().cpu().numpy()
        y_is_lower_bound_np = y_is_lower_bound.detach().cpu().numpy()
        result: list[tuple[ModelPrediction, str, str]] = []

        if self._loss_type == LossType.MDN_NLL:
            mean_np = y_pred_mean.detach().cpu().numpy()
            logvar_np = y_pred_logvar.detach().cpu().numpy()
            mix_np = y_pred_mix.detach().cpu().numpy()
            for m, l, mix, le, q, t, r in zip(
                mean_np, logvar_np, mix_np,
                x_len_np, query_ids, tpcds_temp_and_q_idxs, run_ids,
            ):
                result.append((
                    ModelPrediction(
                        mean_s=m,
                        std_dev_s=[np.exp(li / 2) for li in l],
                        mix_coeffs=mix,
                        metadata={
                            "num_other_concurrent_queries": int(le) - 1,
                            "run_id": r,
                            "tpcds_temp_and_q_idx": t,
                            "query_id": q,
                        },
                    ),
                    r,
                    q,
                ))
        elif self._loss_type == LossType.NLL:
            mean_np = y_pred_mean.detach().cpu().numpy()
            logvar_np = y_pred_logvar.detach().cpu().numpy()
            for m, l, le, q, t, r in zip(
                mean_np, logvar_np,
                x_len_np, query_ids, tpcds_temp_and_q_idxs, run_ids,
            ):
                result.append((
                    ModelPrediction(
                        mean_s=m,
                        std_dev_s=[np.exp(li / 2) for li in l],
                        metadata={
                            "num_other_concurrent_queries": int(le) - 1,
                            "run_id": r,
                            "tpcds_temp_and_q_idx": t,
                            "query_id": q,
                        },
                    ),
                    r,
                    q,
                ))
        else:  # LossType.SENSITIVE_Q_ERROR
            mean_np = y_pred_mean.detach().cpu().numpy()
            losses: list | np.ndarray = (
                per_item_losses if per_item_losses is not None
                else [0.0] * len(mean_np)
            )
            for m, le, q, t, r, yislb, loss_val in zip(
                mean_np, x_len_np, query_ids, tpcds_temp_and_q_idxs, run_ids,
                y_is_lower_bound_np, losses,
            ):
                result.append((
                    ModelPrediction(
                        mean_s=m,
                        metadata={
                            "num_other_concurrent_queries": int(le) - 1,
                            "run_id": r,
                            "tpcds_temp_and_q_idx": t,
                            "query_id": q,
                            "target_is_lower_bound": yislb,
                            "loss": loss_val,
                        },
                    ),
                    r,
                    q,
                ))

        return result

    def _infer_batch(
        self,
        batch: tuple,
    ) -> list[tuple["ModelPrediction", str, str]]:
        """
        Runs a single forward pass on a pre-collated batch without computing
        any loss. Returns a list of (ModelPrediction, run_id, query_id) tuples.

        This is faster than ``_process_batch(..., derive_individual_predictions=True)``
        because it skips the loss computation entirely.
        """

        (
            x,
            x_len,
            pinch_points,
            y,
            query_ids,
            tpcds_temp_and_q_idxs,
            run_ids,
            y_is_lower_bound,
        ) = batch

        train_config = self._train_config_sequence[-1]
        result: list[tuple[ModelPrediction, str, str]] = []

        # Handle isolated queries via the stage model when configured.
        if train_config.use_stage_for_isolated_queries:
            non_isolated_indices = []
            for i in range(len(x_len)):
                if x_len[i].item() > 1:
                    non_isolated_indices.append(i)
                else:
                    pred = self._predict_isolated_query(
                        i, x, pinch_points, query_ids, tpcds_temp_and_q_idxs,
                        run_ids, y_is_lower_bound,
                    )
                    result.append((pred, run_ids[i], query_ids[i]))

            if not non_isolated_indices:
                return result

            x = x[non_isolated_indices]
            x_len = x_len[non_isolated_indices]
            pinch_points = pinch_points[non_isolated_indices]
            query_ids = [query_ids[i] for i in non_isolated_indices]
            tpcds_temp_and_q_idxs = [tpcds_temp_and_q_idxs[i] for i in non_isolated_indices]
            run_ids = [run_ids[i] for i in non_isolated_indices]
            y_is_lower_bound = y_is_lower_bound[non_isolated_indices]

        y_pred_mean, y_pred_logvar, y_pred_mix = self._nn(
            x.to(self._device),
            x_len.to(self._device),
            pinch_points.to(self._device),
            train_config.mdn_mix_softmax_temperature,
        )

        if self._trained_on_log_runtime:
            y_pred_mean = torch.exp(y_pred_mean)
            y_pred_logvar = torch.exp(y_pred_logvar)

        result.extend(self._build_predictions_from_nn_output(
            y_pred_mean, y_pred_logvar, y_pred_mix,
            x_len, query_ids, tpcds_temp_and_q_idxs, run_ids, y_is_lower_bound,
        ))
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
                    "train_config_sequence": [
                        asdict(tc) for tc in self._train_config_sequence
                    ],
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

        model = IconqModel(
            init_config=IconqModelInitConfig(**params["init_config"]),
            train_config_sequence=[
                NNModelTrainConfig(**tc_dict)
                for tc_dict in params["train_config_sequence"]
            ],
            device=torch.device(params["device"]),
            parent_save_dir=parent_load_dir,
            model_id=model_id,
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

    def _get_dataloaders(  # pylint: disable=too-many-locals
        self,
        dataset: ConcurrentQueryDataset,
        train_config: NNModelTrainConfig,
        split: bool = True,
        save_dataset: bool = False,
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """
        Converts the given dataset into DataLoaders for training and validation.

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
        """

        if not split:
            dataloader = DataLoader(
                dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                collate_fn=ConcurrentQueryDataset.collate_and_pad,
            )
            return dataloader, None

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

            train_indices = [
                i
                for i in range(len(dataset))
                if dataset.run_ids[i] in train_run_ids
            ]
            val_indices = [
                i
                for i in range(len(dataset))
                if dataset.run_ids[i] in val_run_ids
            ]
            test_indices = [
                i
                for i in range(len(dataset))
                if dataset.run_ids[i] in test_run_ids
            ]

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            test_dataset = Subset(dataset, test_indices)

        if save_dataset:
            dataset.save_to(os.path.join(self._save_dir, "dataset.pkl"))
            with open(
                os.path.join(self._save_dir, "train_indices.pkl"), "wb"
            ) as f:
                pickle.dump(train_indices, f)
            with open(
                os.path.join(self._save_dir, "val_indices.pkl"), "wb"
            ) as f:
                pickle.dump(val_indices, f)
            with open(
                os.path.join(self._save_dir, "test_indices.pkl"), "wb"
            ) as f:
                pickle.dump(test_indices, f)

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

        return train_dataloader, val_dataloader

    def _get_data_splits(  # pylint: disable=too-many-arguments
        self,
        data_size: int,
        train_config: NNModelTrainConfig,
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
    ) -> tuple[float, float]:
        """
        Runs the training loop for the model and returns the final training and
        validation loss.

        Parameters:
            train_dataloader: The DataLoader for the training set.
            val_dataloader: The DataLoader for the validation set.
        Returns:
            final_train_loss: The final training loss.
            final_val_loss: The final validation loss.
        """
        self._nn.train()
        train_config = self._train_config_sequence[-1]

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

            s = f"""******** Starting Epoch {epoch}/{train_config.num_epochs} ********"""
            print(s)
            logger.info(s)

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()
                (batch_loss, _) = self._process_batch(
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

            # Update bookkeeping
            lr_trajectory[epoch] = optimizer.param_groups[0]["lr"]
            scheduler.step()
            train_loss_trajectory[epoch] = (
                total_train_batch_loss / total_train_batches
            )
            val_loss_trajectory[epoch], errors = self._validate(
                val_dataloader=val_dataloader,
                var_reg_weight=train_config.var_reg_weight,
                training_dir=self._save_dir,
                epoch=epoch,
                train_config=train_config,
            )
            _, train_errors = self._validate(
                val_dataloader=train_dataloader,
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
            s = (
                f"""Mean train batch loss: {train_loss_trajectory[epoch]}, """
                f"""Mean val batch loss: {val_loss_trajectory[epoch]}, """
                f"""Learning rate: {lr_trajectory[epoch]}\n"""
            )
            for set_name, errs in zip(
                ["training", "Validation"], [train_errors, errors]
            ):
                for suffix in ["normal", "aborted"]:
                    s += (
                        f"""On the {set_name} set, {suffix} queries:\n"""
                        f"""\tMean abs error: {errs[f"mean_abs_error_{suffix}"]}, """
                        f"""Mean q error: {errs[f"mean_q_error_{suffix}"]}\n"""
                        f"""\tp50 abs error: {errs[f"p50_abs_error_{suffix}"]}, """
                        f"""p50 q error: {errs[f"p50_q_error_{suffix}"]}\n"""
                        f"""\tp90 abs error: {errs[f"p90_abs_error_{suffix}"]}, """
                        f"""p90 q error: {errs[f"p90_q_error_{suffix}"]}\n"""
                        f"""\tp95 abs error: {errs[f"p95_abs_error_{suffix}"]}, """
                        f"""p95 q error: {errs[f"p95_q_error_{suffix}"]}\n"""
                    )

            print(s)
            logger.info(s)

            # Check for early stopping
            if val_loss_trajectory[epoch] > (
                best_val_loss["val_loss"] - train_config.min_loss_improvement
            ):
                epochs_without_improvement += 1
                if epochs_without_improvement < train_config.patience:
                    continue
                else:
                    s = (
                        f"Early stopping at epoch {epoch} after {train_config.patience} "
                        "epochs with no validation loss improvement."
                    )
                    print(s)
                    logger.info(s)
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
            s = """Saving model... """
            print(s)
            logger.info(s)

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

        return final_train_loss, final_val_loss

    def _validate(
        self,
        val_dataloader: DataLoader,
        train_config: NNModelTrainConfig,
        var_reg_weight: float = 0.0,
        training_dir: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluates the model on the validation set.

        Parameters:
            val_dataloader: The DataLoader for the validation set.
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

        total_val_batches = len(val_dataloader)
        total_val_batch_loss = 0.0
        all_pred_v_true = []
        self._nn.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                (batch_loss, batch_pred_v_true) = self._process_batch(
                    batch=batch,
                    train_config=train_config,
                    derive_individual_predictions=True,
                )
                all_pred_v_true.extend(batch_pred_v_true)
                total_val_batch_loss += batch_loss.item()

        mean_val_batch_loss = total_val_batch_loss / total_val_batches

        # Calculate error metrics
        errors: dict[str, float] = {}

        # (ModelPrediction, true_runtime, query_id, tpcds_temp_and_q_idx, run_id)
        abs_error = [
            abs(pred.overall_mean_s() - true)
            for pred, true, _, _, _ in all_pred_v_true
        ]
        q_error = [
            max(pred.overall_mean_s() / true, true / pred.overall_mean_s())
            for pred, true, _, _, _ in all_pred_v_true
        ]
        was_aborted = [
            pred.metadata.get("target_is_lower_bound", False)
            for pred, _, _, _, _ in all_pred_v_true
        ]

        for aborted in [True, False]:
            suffix = "aborted" if aborted else "normal"
            filtered_abs_error = [
                abs_err
                for abs_err, was_ab in zip(abs_error, was_aborted)
                if was_ab == aborted
            ]
            filtered_q_error = [
                q_err
                for q_err, was_ab in zip(q_error, was_aborted)
                if was_ab == aborted
            ]
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
                query_id for _, _, query_id, _, _ in all_pred_v_true
            ]
            val_df["num_other_concurrent_queries"] = [
                pred.metadata["num_other_concurrent_queries"]
                for pred, _, _, _, _ in all_pred_v_true
            ]
            val_df["y"] = [true for _, true, _, _, _ in all_pred_v_true]
            val_df["y_pred"] = [pred for pred, _, _, _, _ in all_pred_v_true]
            val_df["target_is_lower_bound"] = [
                pred.metadata["target_is_lower_bound"]
                for pred, _, _, _, _ in all_pred_v_true
            ]
            val_df["tpcds_temp_and_q_idx"] = [
                t for _, _, _, t, _ in all_pred_v_true
            ]
            val_df["abs_error"] = abs_error
            val_df["q_error"] = q_error
            val_df["run_id"] = [
                pred.metadata["run_id"] for pred, _, _, _, _ in all_pred_v_true
            ]
            if "loss" in all_pred_v_true[0][0].metadata:
                val_df["individual_loss"] = [
                    pred.metadata["loss"]
                    for pred, _, _, _, _ in all_pred_v_true
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
            query_ids,
            tpcds_temp_and_q_idxs,
            run_ids,
            y_is_lower_bound,
        ) = batch

        batch_pred_v_true = []
        # List of (ModelPrediction, true_runtime, query_id, tpcds_temp_and_q_idx, run_id)
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
                        i, x, pinch_points, query_ids, tpcds_temp_and_q_idxs,
                        run_ids, y_is_lower_bound,
                    )
                    batch_pred_v_true.append((
                        pred, y[i], query_ids[i], tpcds_temp_and_q_idxs[i], run_ids[i],
                    ))

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
                for (pred, r, q), y_, t in zip(
                    self._build_predictions_from_nn_output(
                        y_pred_mean, y_pred_logvar, y_pred_mix,
                        x_len, query_ids, tpcds_temp_and_q_idxs, run_ids, y_is_lower_bound,
                    ),
                    y_np,
                    tpcds_temp_and_q_idxs,
                ):
                    batch_pred_v_true.append((pred, y_, q, t, r))
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
                for (pred, r, q), y_, t in zip(
                    self._build_predictions_from_nn_output(
                        y_pred_mean, y_pred_logvar, y_pred_mix,
                        x_len, query_ids, tpcds_temp_and_q_idxs, run_ids, y_is_lower_bound,
                    ),
                    y_np,
                    tpcds_temp_and_q_idxs,
                ):
                    batch_pred_v_true.append((pred, y_, q, t, r))
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
                for (pred, r, q), y_, t in zip(
                    self._build_predictions_from_nn_output(
                        y_pred_mean, y_pred_logvar, y_pred_mix,
                        x_len, query_ids, tpcds_temp_and_q_idxs, run_ids, y_is_lower_bound,
                        per_item_losses=loss.detach().cpu().numpy(),
                    ),
                    y_np,
                    tpcds_temp_and_q_idxs,
                ):
                    batch_pred_v_true.append((pred, y_, q, t, r))

        return (
            batch_loss,
            batch_pred_v_true,
        )
