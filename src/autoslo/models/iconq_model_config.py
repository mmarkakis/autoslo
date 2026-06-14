from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class IconqModelInitConfig:
    """
    A dataclass for the configuration of the Iconq model at initialization.
    """

    schema_name: str  # The schema name.

    iconq_query_featurizer_id: Optional[str] = None
    iconq_query_featurizer_init_params: Optional[dict[str, Any]] = None
    featurizer_num_operators: int = (
        10  # The number of operators to consider in the query featurizer.
    )
    featurizer_num_tables: int = (
        10  # The number of tables to consider in the query featurizer.
    )

    stage_model_id: Optional[str] = None
    stage_model_init_params: Optional[dict[str, Any]] = None

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
    interaction_feature_version: Literal["v1", "v2"] = (
        "v1"  # Version of interaction features used by IconqInteractionFeaturizer.
    )


@dataclass
class IconqModelTrainConfig:
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

    train_stage_only_on_isolated_queries: bool = (
        False  # Whether to train stage only on isolated queries.
    )

    use_client_side_latencies: bool = (
        False  # Whether to use client-side latencies instead of server-side.
    )

    ignore_aborted_queries: bool = (
        False  # Whether to exclude aborted queries as training targets.
        # When True, aborted queries are removed from the base-query set
        # (and thus from the dataset / model training) but are still kept
        # as *neighbors* of other queries in the IconqModel dataset.
        # The same flag is forwarded to the CacheModel and XGBoostModel.
    )
