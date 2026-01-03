import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional, cast

import networkx as nx
import numpy as np
import pandas as pd
import torch
import xxhash
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.query_timeline import QueryTimeline
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
from autoslo.workload_execution.trace import Trace

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

        # Set up logging.

    def train(  # pylint: disable=arguments-differ,too-many-locals
        self,
        train_config: NNModelTrainConfig,
        from_scratch: bool = True,
    ) -> tuple[float, float]:
        """
        Train the model on the given workload and return the final training and validation loss.

        Parameters:
            train_config: The training configuration for the LSTM model.
            from_scratch: Whether to train the model from scratch.

        Returns:
            final_train_loss: The final training loss.
            final_val_loss: The final validation loss.
        """

        if from_scratch:
            self._train_config_sequence = []
            self._nn = RuntimeNet(**self._nn_args).to(self._device)  # type: ignore
        self._train_config_sequence.append(train_config)

        # During execution, what gets logged out is a list of query executions.
        # However, the inputs to our model are features regarding the concurrent
        # execution of queries. We need to transform the data into this format,
        # or load it from cache if it already exists.
        # FIXME: We may want to cache this, we will see.
        overlap_graph = self._get_and_enhance_overlap_graph(
            train_config.run_ids
        )
        train_dataloader, val_dataloader = self._graph_to_dataloaders(
            overlap_graph,
            train_config,
            use_log_runtime=self._trained_on_log_runtime,
        )

        # Now we are ready for the actual training loop
        return self._run_training_loop(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
        )

    def save(self) -> str:
        """
        Saves the IconqModel.

        Returns:
            The identifier of the saved IconqModel. This is a subdirectory under
                the parent_save_dir.
        """

        # Save model parameters
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

        # Save the model checkpoint
        update_checkpoint(self._nn, self._save_dir)

        return self._model_id

    @staticmethod
    def load(
        self, model_id: str, parent_load_dir: Optional[str] = None
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
            init_config=cast(IconqModelInitConfig, params["init_config"]),
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
            self._nn._state_dict = torch.load(checkpoint_path)

        return model

    def _get_and_enhance_overlap_graph(  # pylint: disable=too-many-locals
        self,
        run_ids: list[str],
    ) -> nx.Graph:
        """
        Gets the overlap graph for the given run IDs, and enhances it with
        featurizations and stage model predictions.

        Parameters:
            run_ids: The run IDs to compute the overlaps for.

        Returns:
            overlap_graph: The overlap graph with enhanced information.
        """
        overall_overlap_graph: nx.Graph = nx.Graph()

        for run_id in run_ids:
            trace = Trace(run_id)
            query_timeline = QueryTimeline(self._iconq_query_featurizer)
            query_timeline.initialize_from_trace(trace)
            overlap_graph = query_timeline.overlap_graph()

            # To each node in the graph, add its featurization and its stage
            # model prediction.
            for node, node_data in overlap_graph.nodes(data=True):
                node_data["query_featurization"] = (
                    self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                        node_data["tpcds_temp_and_q_idx"]
                    )
                )
                node_data["stage_model_prediction"] = (
                    self._stage_model.predict_from_tpcds_temp_and_q_idx(
                        {
                            node_data["query_id"]: node_data[
                                "tpcds_temp_and_q_idx"
                            ]
                        }
                    )[node_data["query_id"]]
                )
            overall_overlap_graph = nx.compose(
                overall_overlap_graph, overlap_graph
            )

        return overall_overlap_graph

    def _graph_to_dataloaders(  # pylint: disable=too-many-locals
        self,
        overlap_graph: nx.Graph,
        train_config: NNModelTrainConfig,
        use_log_runtime: bool,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Converts the given overlap graph into training and validation
        dataloaders.

        Parameters:
            overlap_graph: The overlap graph to convert.
            train_config: The training configuration for the LSTM model.
            use_log_runtime: Whether to use the log of the runtime as target.

        Returns:
            train_dataloader: The training DataLoader.
            val_dataloader: The validation DataLoader.
        """

        x = []
        y = []
        pinch_points = []
        query_id_hashes = []

        for node in overlap_graph.nodes:
            node_data = overlap_graph.nodes[node]
            interaction_featurizations: dict[
                float, IconqInteractionFeaturizer.IconqInteractionFeaturization
            ] = {}

            # Add oneself to the interaction featurizations. This helps with
            # queries that do not have any overlapping neighbors.
            interaction_featurizations[node_data["start_time_s"]] = (
                self._iconq_interaction_featurizer.featurize_from_vectors(
                    qa_features=node_data["query_featurization"],
                    qa_start_time_s=node_data["start_time_s"],
                    qa_latency_prediction=node_data[
                        "stage_model_prediction"
                    ].overall_mean_s(),
                    qb_features=node_data["query_featurization"],
                    qb_start_time_s=node_data["start_time_s"],
                    qb_latency_prediction=node_data[
                        "stage_model_prediction"
                    ].overall_mean_s(),
                )
            )

            # Collect the interaction featurizations with neighboring nodes.
            for neighbor in overlap_graph.neighbors(node):
                neighbor_data = overlap_graph.nodes[neighbor]
                interaction_featurizations[neighbor_data["start_time_s"]] = (
                    self._iconq_interaction_featurizer.featurize_from_vectors(
                        qa_features=node_data["query_featurization"],
                        qa_start_time_s=node_data["start_time_s"],
                        qa_latency_prediction=node_data[
                            "stage_model_prediction"
                        ].overall_mean_s(),
                        qb_features=neighbor_data["query_featurization"],
                        qb_start_time_s=neighbor_data["start_time_s"],
                        qb_latency_prediction=neighbor_data[
                            "stage_model_prediction"
                        ].overall_mean_s(),
                    )
                )
            neighbor_sort_order = sorted(interaction_featurizations.keys())

            # Update the tensors.
            x.append(
                torch.stack(
                    [
                        torch.tensor(
                            interaction_featurizations[neighbor_start_time_s],
                            dtype=torch.float32,
                        )
                        for neighbor_start_time_s in neighbor_sort_order
                    ]
                )
            )
            latency = node_data["end_time_s"] - node_data["start_time_s"]
            y.append(latency if not use_log_runtime else np.log1p(latency))
            pinch_points.append(
                neighbor_sort_order.index(node_data["start_time_s"])
            )
            query_id_hashes.append(
                xxhash.xxh32(node_data["query_id"]).intdigest()
            )

        # Transform lists into tensors.
        x_tensorized = x
        pinch_points_tensorized = torch.tensor(pinch_points, dtype=torch.int8)
        y_tensorized = torch.tensor(y, dtype=torch.float32)
        query_id_hashes_tensorized = torch.tensor(
            query_id_hashes, dtype=torch.int64
        )

        # Create the dataset and dataloaders.
        dataset = ConcurrentQueryDataset(
            x=x_tensorized,
            pinch_points=pinch_points_tensorized,
            y=y_tensorized,
            query_id_hashes=query_id_hashes_tensorized,
        )
        train_idxs, val_idxs, _ = self._get_data_splits(
            len(dataset), train_config
        )
        train_dataset = Subset(dataset, train_idxs)
        val_dataset = Subset(dataset, val_idxs)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
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
            for x, x_len, pinch_points, y, query_uuid in val_dataloader:
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
                        y = torch.exp(y)
                        y_pred_mean = torch.exp(y_pred_mean)
                        y_pred_logvar = torch.exp(y_pred_logvar)

                    for m, l, x, y_, q in zip(
                        y_pred_mean.detach().numpy(),
                        y_pred_logvar.detach().numpy(),
                        y_pred_mix.detach().numpy(),
                        y.numpy(),
                        query_uuid.numpy(),
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
                        y = torch.exp(y)
                        y_pred_mean = torch.exp(y_pred_mean)
                        y_pred_logvar = torch.exp(y_pred_logvar)

                    for m, l, y_, q in zip(
                        y_pred_mean.detach().numpy(),
                        y_pred_logvar.detach().numpy(),
                        y.numpy(),
                        query_uuid.numpy(),
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
                        y = torch.exp(y)
                        y_pred_mean = torch.exp(y_pred_mean)

                    for m, y_, q in zip(
                        y_pred_mean.detach().numpy(),
                        y.numpy(),
                        query_uuid.numpy(),
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
            val_df["query_uuid"] = [
                query_uuid for _, _, query_uuid in all_pred_v_true
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
