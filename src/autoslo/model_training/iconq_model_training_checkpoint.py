from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import yaml

from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.nn.loss_functions import LossType


def plot_loss(
    save_path: Path,
    train_loss: dict[int, float],
    val_loss: dict[int, float],
    loss_type: LossType,
    mark_x: Optional[int] = None,
) -> None:
    """
    Plot the train and validation losses.

    Parameters:
        save_path: The path to save the plot.
        train_loss: A dictionary mapping epoch numbers to batch train losses.
        val_loss: A dictionary mapping epoch numbers to batch validation losses.
        loss_type: The type of loss to plot.
        mark_x: The x-coordinate to mark on the plot.
    """

    plt.plot(
        list(train_loss.keys()), list(train_loss.values()), label="Train Loss"
    )
    plt.plot(
        list(val_loss.keys()), list(val_loss.values()), label="Validation Loss"
    )
    if mark_x is not None:
        plt.axvline(mark_x, color="red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel(f"Loss ({loss_type.value})")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.clf()


def plot_learning_rate(
    save_path: Path,
    lr_trajectory: dict[int, float],
    mark_x: Optional[int] = None,
) -> None:
    """
    Plot the trajectory of the learning rate.

    Parameters:
        save_path: The path to save the plot.
        lr_trajectory: A dictionary mapping epoch numbers to learning rates.
        mark_x: The x-coordinate to mark on the plot.
    """

    plt.plot(
        list(lr_trajectory.keys()),
        list(lr_trajectory.values()),
        label="Learning Rate",
    )
    if mark_x is not None:
        plt.axvline(mark_x, color="red", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.clf()


def update_checkpoint(
    nn: torch.nn.Module,
    dir: Path,
) -> None:
    """
    Update the model checkpoint.

    Parameters:
        nn: The model to save.
        dir: The directory to save the checkpoint to.
    """
    current_ts = int(datetime.now().timestamp())
    model_save_path = dir / f"model_{current_ts}.pth"
    torch.save(nn.state_dict(), model_save_path)

    # Remove old model.
    for file in dir.iterdir():
        if file.name.startswith("model_") and file.name.endswith(".pth"):
            prev_checkpoint_ts = int(file.name[len("model_") : -len(".pth")])
            if prev_checkpoint_ts < current_ts:
                (dir / f"model_{prev_checkpoint_ts}.pth").unlink()


def update_plots(
    training_dir: Path,
    prev_checkpoint_epoch: int,
    train_loss_trajectory: dict[int, float],
    val_loss_trajectory: dict[int, float],
    lr_trajectory: dict[int, float],
    loss_type: LossType,
    mark_prev_checkpoint: bool = False,
) -> None:
    """
    Update the learning rate and loss plots.

    Parameters:
        training_dir: The directory to save the checkpoint to.
        prev_checkpoint_epoch: The previous epoch to remove the model from.
        train_loss_trajectory: A dictionary mapping epoch numbers to batch train losses.
        val_loss_trajectory: A dictionary mapping epoch numbers to batch validation losses.
        lr_trajectory: A dictionary mapping epoch numbers to learning rates.
        loss_type: The type of loss to plot.
        mark_prev_checkpoint: Whether to mark the previous checkpoint on the plot.
    """
    mark_x = prev_checkpoint_epoch if mark_prev_checkpoint else None
    loss_save_path = training_dir / "loss_trajectory.png"
    plot_loss(
        loss_save_path,
        train_loss_trajectory,
        val_loss_trajectory,
        loss_type,
        mark_x,
    )
    lr_save_path = training_dir / "lr_trajectory.png"
    plot_learning_rate(lr_save_path, lr_trajectory, mark_x)
