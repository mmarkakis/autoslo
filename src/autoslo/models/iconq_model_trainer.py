from autoslo.blueprint_selection.query_timeline import QueryTimeline
from autoslo.models.iconq_model import IconqModel, NNModelTrainConfig
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.nn.runtime_net import RuntimeNet
from autoslo.workload_execution.trace import Trace


def iconq_model_trainer(  # pylint: disable=arguments-differ,too-many-locals
    iconq_model: IconqModel,
    train_config: NNModelTrainConfig,
    from_scratch: bool = True,
) -> tuple[float, float]:
    """
    Train the model on the given workload and return the final training and validation loss.

    Parameters:
        iconq_model: The Iconq model to train.
        train_config: The training configuration for the LSTM model.
        from_scratch: Whether to train the model from scratch.

    Returns:
        final_train_loss: The final training loss.
        final_val_loss: The final validation loss.
    """

    if from_scratch:
        iconq_model._train_config_sequence = []
        iconq_model._nn = RuntimeNet(**iconq_model._nn_args).to(iconq_model._device)  # type: ignore
    iconq_model._train_config_sequence.append(train_config)
    iconq_model._save_params()

    use_fixed_window_radius_s = (
        iconq_model._init_config.use_fixed_window_radius_s
    )
    use_fixed_window_max_neighbors_per_side = (
        iconq_model._init_config.use_fixed_window_max_neighbors_per_side
    )
    
    datasets = []
    for run_id in train_config.run_ids:
        trace = Trace(run_id)
        query_timeline = QueryTimeline(iconq_model=iconq_model)
        query_timeline.initialize_from_trace(trace)
        dataset = query_timeline.get_dataset(
            use_log_runtime=iconq_model.trained_on_log_runtime,
            run_id=run_id,
            use_fixed_window_radius_s=use_fixed_window_radius_s,
            use_fixed_window_max_neighbors_per_side=(
                use_fixed_window_max_neighbors_per_side
            ),
        )
        datasets.append(dataset)
    overall_dataset = ConcurrentQueryDataset.concatenate(datasets)

    train_dataloader, val_dataloader = iconq_model._get_dataloaders(
        overall_dataset,
        train_config,
        split=True,
        save_dataset=True,
    )
    assert val_dataloader is not None  # For linter

    # Now we are ready for the actual training loop
    return iconq_model._run_training_loop(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
