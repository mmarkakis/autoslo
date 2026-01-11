import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, cast

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
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

from autoslo.blueprints.cluster import Cluster

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

    learning_rate: float = 1e-3  # The learning rate for the optimizer.
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
            iconq_query_featurizer_id=self._iconq_query_featurizer_id
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
        use_stage_for_isolated_queries: bool = False,
    ) -> dict[str, ModelPrediction]:
        """
        Predicts the runtimes for the queries in the given dataset.

        Parameters:
            dataset: The dataset to predict runtimes for.
            use_stage_for_isolated_queries: Whether to just fall back to the
                underlying stage model for isolated queries (i.e., those without
                any concurrent queries).

        Returns:
            A dictionary mapping query IDs to their predicted ModelPrediction.
        """

        dataloader, _ = self._get_dataloaders(
            dataset,
            train_config=None,
            split=False,
            save_dataset=False,
        )

        predictions: dict[str, ModelPrediction] = {}
        self._nn.eval()
        with torch.no_grad():
            for (
                x,
                x_len,
                pinch_points,
                _,
                query_ids,
                tpcds_temp_and_q_idxs,
            ) in dataloader:
                x, x_len, pinch_points = (
                    x.to(self._device),
                    x_len.to(self._device),
                    pinch_points.to(self._device),
                )

                y_pred_mean, y_pred_logvar, y_pred_mix = self._nn(
                    x,
                    x_len,
                    pinch_points,
                    mdn_mix_softmax_temperature=1.0,
                )

                if self._trained_on_log_runtime:
                    y_pred_mean = torch.expm1(y_pred_mean)
                    y_pred_logvar = torch.expm1(y_pred_logvar)

                for i, query_id in enumerate(query_ids):
                    num_other_concurrent_queries = x_len[i].item() - 1
                    pred_meta = {
                        "num_other_concurrent_queries": num_other_concurrent_queries
                    }
                    if use_stage_for_isolated_queries and (
                        num_other_concurrent_queries == 0
                    ):
                        rpu = x[i][pinch_points[i]][
                            self._iconq_interaction_featurizer.rpu_dim_idx
                        ].item()
                        cluster_name = Cluster.ordered_cluster_names_per_rpu()[
                            int(rpu)
                        ][0]

                        predictions[query_id] = (
                            self.stage_model.predict_from_tpcds_temp_and_q_idx(
                                {query_id: tpcds_temp_and_q_idxs[i]},
                                cluster_name=cluster_name,
                            )[query_id]
                        )
                    elif self._nn_args["is_mdn"]:
                        predictions[query_id] = ModelPrediction(
                            mean_s=[
                                float(it) for it in y_pred_mean[i].tolist()
                            ],
                            std_dev_s=[
                                float(it)
                                for it in torch.exp(
                                    y_pred_logvar[i] / 2
                                ).tolist()
                            ],
                            mix_coeffs=[
                                float(it) for it in y_pred_mix[i].tolist()
                            ],
                            metadata=pred_meta,
                        )
                    elif self._nn_args["is_bayesian"]:
                        predictions[query_id] = ModelPrediction(
                            mean_s=[float(y_pred_mean[i].item())],
                            std_dev_s=[
                                float(torch.exp(y_pred_logvar[i] / 2).item())
                            ],
                            metadata=pred_meta,
                        )
                    else:
                        predictions[query_id] = ModelPrediction(
                            mean_s=[float(y_pred_mean[i].item())],
                            metadata=pred_meta,
                        )

        return predictions

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
        train_config: Optional[NNModelTrainConfig] = None,
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

        if save_dataset:
            dataset.save_to(os.path.join(self._save_dir, "dataset.pkl"))
        if not split:
            dataloader = DataLoader(
                dataset,
                batch_size=len(dataset),
                shuffle=True,
                collate_fn=ConcurrentQueryDataset.collate_and_pad,
            )
            return dataloader, None

        assert train_config is not None  # For linter
        train_idxs, val_idxs, test_idxs = self._get_data_splits(
            len(dataset), train_config
        )
        train_dataset = Subset(dataset, train_idxs)
        val_dataset = Subset(dataset, val_idxs)
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

            for x, x_len, pinch_points, y, _ in tqdm(train_dataloader):
                x, x_len, pinch_points, y = (
                    x.to(self._device),
                    x_len.to(self._device),
                    pinch_points.to(self._device),
                    y.to(self._device),
                )

                optimizer.zero_grad()
                y_pred_mean, y_pred_logvar, y_pred_mix = self._nn(
                    x,
                    x_len,
                    pinch_points,
                    train_config.mdn_mix_softmax_temperature,
                )
                batch_loss: torch.Tensor
                if self._loss_type == LossType.MDN_NLL:
                    batch_loss = mdn_negative_log_likelihood_loss(
                        y_pred_mean,
                        y,
                        y_pred_logvar,
                        y_pred_mix,
                        train_config.var_reg_weight,
                    )
                elif self._loss_type == LossType.NLL:
                    batch_loss = negative_log_likelihood_loss(
                        y_pred_mean,
                        y,
                        y_pred_logvar,
                        train_config.var_reg_weight,
                    )
                else:
                    batch_loss = sensitive_q_error_loss(y_pred_mean, y)
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
                val_dataloader,
                train_config.var_reg_weight,
                self._save_dir,
                epoch,
                train_config.mdn_mix_softmax_temperature,
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
                f"""On the validation set:\n"""
                f"""\tMean abs error: {errors["mean_abs_error"]}, """
                f"""Mean q error: {errors["mean_q_error"]}\n"""
                f"""\tp50 abs error: {errors["p50_abs_error"]}, """
                f"""p50 q error: {errors["p50_q_error"]}\n"""
                f"""\tp90 abs error: {errors["p90_abs_error"]}, """
                f"""p90 q error: {errors["p90_q_error"]}\n"""
                f"""\tp95 abs error: {errors["p95_abs_error"]}, """
                f"""p95 q error: {errors["p95_q_error"]}"""
                f"""\n----\n"""
                f"""\tFraction of queries with true latency under their predicted p50: """
                f"""{errors["fraction_under_p50"]:.4f} (ideal: 0.5000)\n"""
                f"""\tFraction of queries with true latency under their predicted p90: """
                f"""{errors["fraction_under_p90"]:.4f} (ideal: 0.9000)\n"""
                f"""\tFraction of queries with true latency under their predicted p95: """
                f"""{errors["fraction_under_p95"]:.4f} (ideal: 0.9500)\n"""
                f"""\tFraction of queries with true latency under their predicted p99: """
                f"""{errors["fraction_under_p99"]:.4f} (ideal: 0.9900)"""
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
        var_reg_weight: float = 0.0,
        training_dir: Optional[str] = None,
        epoch: Optional[int] = None,
        mdn_mix_softmax_temperature: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluates the model on the validation set.

        Parameters:
            val_dataloader: The DataLoader for the validation set.
            var_reg_weight: The weight for the variance regularization term for the negative
                log likelihood loss.
            training_dir: The directory for the model training. If given, will save the validation
                predictions to a CSV file.
            epoch: The epoch number. If given and training_dir is given, will include the epoch
                number in the filename of the validation predictions CSV file.
            mdn_mix_softmax_temperature: The temperature for the softmax in the MDN mixture
                coefficients.

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
            for x, x_len, pinch_points, y, query_id in val_dataloader:
                x, x_len, pinch_points, y = (
                    x.to(self._device),
                    x_len.to(self._device),
                    pinch_points.to(self._device),
                    y.to(self._device),
                )
                y_pred_mean, y_pred_logvar, y_pred_mix = self._nn(
                    x, x_len, pinch_points, mdn_mix_softmax_temperature
                )

                batch_loss: torch.Tensor
                if self._loss_type == LossType.MDN_NLL:
                    batch_loss = mdn_negative_log_likelihood_loss(
                        y_pred_mean,
                        y,
                        y_pred_logvar,
                        y_pred_mix,
                        var_reg_weight,
                    )
                    if self._trained_on_log_runtime:
                        y = torch.expm1(y)
                        y_pred_mean = torch.expm1(y_pred_mean)
                        y_pred_logvar = torch.expm1(y_pred_logvar)

                    for m, l, x, y_, q in zip(
                        y_pred_mean.detach().numpy(),
                        y_pred_logvar.detach().numpy(),
                        y_pred_mix.detach().numpy(),
                        y.numpy(),
                        query_id,
                    ):

                        all_pred_v_true.append(
                            (
                                ModelPrediction(
                                    mean_s=m,
                                    std_dev_s=[np.exp(li / 2) for li in l],
                                    mix_coeffs=x,
                                ),
                                y_,
                                q,
                            )
                        )
                elif self._loss_type == LossType.NLL:
                    batch_loss = negative_log_likelihood_loss(
                        y_pred_mean, y, y_pred_logvar, var_reg_weight
                    )
                    if self._trained_on_log_runtime:
                        y = torch.expm1(y)
                        y_pred_mean = torch.expm1(y_pred_mean)
                        y_pred_logvar = torch.expm1(y_pred_logvar)

                    for m, l, y_, q in zip(
                        y_pred_mean.detach().numpy(),
                        y_pred_logvar.detach().numpy(),
                        y.numpy(),
                        query_id,
                    ):
                        all_pred_v_true.append(
                            (
                                ModelPrediction(
                                    mean_s=m,
                                    std_dev_s=[np.exp(li / 2) for li in l],
                                ),
                                y_,
                                q,
                            )
                        )
                elif self._loss_type == LossType.SENSITIVE_Q_ERROR:
                    batch_loss = sensitive_q_error_loss(y_pred_mean, y)

                    if self._trained_on_log_runtime:
                        y = torch.expm1(y)
                        y_pred_mean = torch.expm1(y_pred_mean)

                    for m, y_, q in zip(
                        y_pred_mean.detach().numpy(), y.numpy(), query_id
                    ):
                        all_pred_v_true.append(
                            (
                                ModelPrediction(
                                    mean_s=m,
                                ),
                                y_,
                                q,
                            )
                        )
                total_val_batch_loss += batch_loss.item()

        mean_val_batch_loss = total_val_batch_loss / total_val_batches
        errors: dict[str, float] = {}

        abs_error = [
            abs(pred.overall_mean_s() - true)
            for pred, true, _ in all_pred_v_true
        ]
        q_error = [
            max(pred.overall_mean_s() / true, true / pred.overall_mean_s())
            for pred, true, _ in all_pred_v_true
        ]

        errors["mean_abs_error"] = np.mean(abs_error)
        errors["mean_q_error"] = np.mean(q_error)
        for p in [50, 90, 95]:
            errors[f"p{p}_abs_error"] = np.percentile(abs_error, p)
            errors[f"p{p}_q_error"] = np.percentile(q_error, p)

        percentiles_at_true_latencies = [
            pred.percentile_at_latency(true)
            for pred, true, _ in all_pred_v_true
        ]
        for p in [50, 90, 95, 99]:
            errors[f"fraction_under_p{p}"] = cast(
                float,
                np.mean(np.array(percentiles_at_true_latencies) <= (p / 100.0)),
            )

        # Print out a dataframe of predictions
        if training_dir is not None:
            val_df = pd.DataFrame()
            val_df["query_id"] = [
                query_id for _, _, query_id in all_pred_v_true
            ]
            val_df["y"] = [true for _, true, _ in all_pred_v_true]
            val_df["y_pred"] = [pred for pred, _, _ in all_pred_v_true]
            val_df["abs_error"] = abs_error
            val_df["q_error"] = q_error
            val_df["percentile_at_true_latency"] = percentiles_at_true_latencies
            val_df.sort_values("y", inplace=True, ascending=False)
            suffix = f"_{epoch}" if epoch is not None else ""
            val_df.to_csv(
                os.path.join(training_dir, f"val_predictions{suffix}.csv")
            )

        return mean_val_batch_loss, errors
