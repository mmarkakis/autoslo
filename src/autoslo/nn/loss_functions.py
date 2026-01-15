from math import pi

import torch
import pandas as pd

from enum import Enum

LOG_TWO_PI_CACHE = {}


def get_log_two_pi(device):
    if device not in LOG_TWO_PI_CACHE:
        LOG_TWO_PI_CACHE[device] = torch.log(
            2 * torch.tensor(pi, device=device)
        )
    return LOG_TWO_PI_CACHE[device]

class LossType(Enum):
    """
    An enumeration of the admissible types of loss functions.
    """

    NLL = "NLL"  # The negative log-likelihood loss.
    SENSITIVE_Q_ERROR = "Sensitive Q-Error Loss"  # The sensitive Q-error loss.
    MDN_NLL = "MDN NLL"  # The negative log-likelihood loss for the MDN model.



def negative_log_likelihood_loss(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    logvar: torch.Tensor,
    var_reg_weight: float = 0.01,
    return_mean: bool = True,
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of the target given the mean and 
    log-variance.

    Parameters:
        input: The tensor of mean predictions, of shape (batch_size,).
        target: The target tensor, of shape (batch_size,).
        logvar: The tensor of logvariance predictions, of shape (batch_size,).
        var_reg_weight: The variance regularization weight.
        return_mean: Whether to return the mean of the negative log-likelihood 
            across the batch, or a tensor of the negative log-likelihood for 
            each element in the batch.

    Returns:
        nll: If return_mean is True, the mean negative log-likelihood across the 
            batch, as a scalar. Otherwise, the negative log-likelihood for each 
            element in the batch, as a tensor of shape (batch_size,).
    """
    input = input.flatten()
    target = target.flatten()
    logvar = logvar.flatten()

    logvar = torch.clamp(logvar, min=-25, max=25)
    var = torch.exp(logvar)

    nll = 0.5 * (
        logvar + ((input - target) ** 2) / var + get_log_two_pi(input.device)
    )
    regularization = var_reg_weight * var

    if return_mean:
        return (nll + regularization).mean()
    else:
        return nll + regularization


def mdn_negative_log_likelihood_loss(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    logvar: torch.Tensor,
    mix_coeffs: torch.Tensor,
    var_reg_weight: float = 0.01,
    return_mean: bool = True,
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of the target given the parameters of a 
    Gaussian mixture.

    Parameters:
        input: The tensor of mean predictions, of shape 
            (batch_size, num_gaussians).
        target: The target tensor, of shape (batch_size,).
        logvar: The tensor of logvariance predictions, of shape 
            (batch_size, num_gaussians).
        mix_coeffs: The tensor of mixing coefficients, of shape 
            (batch_size, num_gaussians).
        var_reg_weight: The variance regularization weight.
        return_mean: Whether to return the mean of the negative log-likelihood 
            across the batch, or a tensor of the negative log-likelihood for 
            each element in the batch.

    Returns:
        nll: If return_mean is True, the mean negative log-likelihood across the 
            batch, as a scalar. Otherwise, the negative log-likelihood for each 
            element in the batch, as a tensor of shape (batch_size,).
    """
    logvar = torch.clamp(logvar, min=-10, max=10)
    mix_coeffs = torch.softmax(mix_coeffs, dim=1)
    var = torch.exp(logvar)

    # Expand the target to match the shape of the input
    target = target.unsqueeze(1).expand_as(input)

    # Compute the negative log-likelihood by summing the Gaussians
    log_normal = -0.5 * (
        logvar + ((input - target) ** 2) / var + get_log_two_pi(input.device)
    )
    nll = -torch.logsumexp(torch.log(mix_coeffs) + log_normal, dim=1)
    nll = torch.clamp(nll, min=0.0)

    regularization = var_reg_weight * var.sum(dim=1)

    if return_mean:
        return (nll + regularization).mean()
    else:
        return nll + regularization


def sensitive_q_error_loss(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    min_val: float | torch.Tensor = 0.001,
    small_val: float = 5.0,
    penalty_negative: float = 1e5,
    lambda_small: float = 0.1,
) -> torch.Tensor:
    """
    Compute a loss generally based on the Q-error between the input and target 
    tensors, but with special treatment of certain value ranges.

    For negative values, the loss is based on how negative the input tensor is.
    For small values, the loss is based on the L1 distance between the input and 
    target tensors.
    For all other values, the loss is based on the Q-error.

    Parameters:
        input: The input tensor.
        target: The target tensor.
        min_val: The minimum value for the input tensor. Below it, the value is 
            treated as "negative" for the purposes of selecting how to estimate 
            the loss.
        small_val: The value below which the L1 distance is used.
        penalty_negative: The penalty for negative values.
        lambda_small: The weight for the L1 distance.

    Returns:
        loss: The loss tensor.
    """

    input = input.flatten()
    target = target.flatten()

    # Penalty for negative/too small estimates. influence on loss for a negative 
    # estimate is at least `penalty_negative`.
    if isinstance(min_val, float):
        min_val = torch.full_like(input, fill_value=min_val)

    negative_mask = input < min_val
    negative_loss = (1 + torch.abs(input)) * penalty_negative

    # Use l1_loss for small values, q_loss would explode.
    small_mask = (input < small_val) & (target < small_val)
    small_loss = torch.abs(target - input) * lambda_small

    # Otherwise (logarithmic) q error.
    q_error = torch.where(
        input > target,
        torch.log(input) - torch.log(target),
        torch.log(target) - torch.log(input),
    )

    # Combine losses
    loss = torch.where(
        negative_mask,
        negative_loss,
        torch.where(small_mask, small_loss, q_error),
    )

    return loss.mean().requires_grad_(True)
