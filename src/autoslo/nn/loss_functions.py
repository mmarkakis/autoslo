from enum import Enum
from math import pi

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

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
    target_is_lower_bound: torch.Tensor,
    min_val: float | torch.Tensor = 0.001,
    small_val: float = 5.0,
    penalty_negative: float = 1e5,
    lambda_small: float = 0.1,
    sensitive_q_error_loss_version: int = 1,
    return_mean: bool = True,
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
        target_is_lower_bound: A tensor indicating whether each target value
            is an actual observation, or a lower bound.
        min_val: The minimum value for the input tensor. Below it, the value is
            treated as "negative" for the purposes of selecting how to estimate
            the loss.
        small_val: The value below which the L1 distance is used.
        penalty_negative: The penalty for negative values.
        lambda_small: The weight for the L1 distance.
        sensitive_q_error_loss_version: The version of the sensitive Q-error
            loss to use.
        return_mean: Whether to return the mean loss across the batch, or
            the elementwise loss.

    Returns:
        loss: The loss tensor.
    """
    ver_to_func = {
        1: sensitive_q_error_loss_v1,
        2: sensitive_q_error_loss_v2,
        3: sensitive_q_error_loss_v3,
        4: sensitive_q_error_loss_v4,
        5: sensitive_q_error_loss_v5,
        10: sensitive_q_error_loss_v10,
    }

    if sensitive_q_error_loss_version in ver_to_func:
        return ver_to_func[sensitive_q_error_loss_version](
            input=input,
            target=target,
            target_is_lower_bound=target_is_lower_bound,
            min_val=min_val,
            small_val=small_val,
            penalty_negative=penalty_negative,
            lambda_small=lambda_small,
            return_mean=return_mean,
        )
    else:
        raise ValueError(
            f"Unsupported sensitive Q-error loss version: "
            f"{sensitive_q_error_loss_version}"
        )


def sensitive_q_error_loss_v1(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    target_is_lower_bound: torch.Tensor,
    min_val: float | torch.Tensor = 0.001,
    small_val: float = 5.0,
    penalty_negative: float = 1e5,
    lambda_small: float = 0.1,
    return_mean: bool = True,
) -> torch.Tensor:

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

    if return_mean:
        return loss.mean().requires_grad_(True)
    else:
        return loss.requires_grad_(True)


def sensitive_q_error_loss_v2(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    target_is_lower_bound: torch.Tensor,
    min_val: float | torch.Tensor = 0.001,
    small_val: float = 5.0,
    penalty_negative: float = 1e5,
    lambda_small: float = 0.1,
    return_mean: bool = True,
) -> torch.Tensor:

    input = input.flatten()
    target = target.flatten()
    is_lb = target_is_lower_bound.flatten()
    eps = 1e-8

    # Penalty for negative/too small estimates. influence on loss for a negative
    # estimate is at least `penalty_negative`.
    if isinstance(min_val, float):
        min_val = torch.full_like(input, fill_value=min_val)

    negative_mask = input < min_val
    negative_loss = (1 + torch.abs(input)) * penalty_negative

    # Use l1_loss for small values, q_loss would explode.
    small_mask = (input < small_val) & (target < small_val)
    # If target is a lower bound, penalize only under-predictions in small regime
    small_loss = (
        torch.where(
            is_lb,
            torch.clamp(target - input, min=0.0),
            torch.abs(target - input),
        )
        * lambda_small
    )

    # Otherwise (logarithmic) q error.
    input_safe = torch.clamp(input, min=eps)
    target_safe = torch.clamp(target, min=eps)
    q_error_sym = torch.where(
        input_safe > target_safe,
        torch.log(input_safe) - torch.log(target_safe),
        torch.log(target_safe) - torch.log(input_safe),
    )
    # If target is a lower bound, penalize only when predicting under the target
    q_error = torch.where(
        is_lb,
        torch.clamp(torch.log(target_safe) - torch.log(input_safe), min=0.0),
        q_error_sym,
    )

    # Combine losses
    loss = torch.where(
        negative_mask,
        negative_loss,
        torch.where(small_mask, small_loss, q_error),
    )

    if return_mean:
        return loss.mean().requires_grad_(True)
    else:
        return loss.requires_grad_(True)


def sensitive_q_error_loss_v3(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    target_is_lower_bound: torch.Tensor,
    min_val: float | torch.Tensor = 0.001,
    penalty_negative: float = 1e5,
    return_mean: bool = True,
    **kwargs,  # To ignore unused parameters.
) -> torch.Tensor:

    input = input.flatten()
    target = target.flatten()
    if isinstance(min_val, float):
        min_val = torch.full_like(input, fill_value=min_val)
    is_lb = target_is_lower_bound.flatten()

    # Assumption: target is always >= min_val.

    # Regime 1: Prediction (input) is larger than the target.
    # Apply Q-error penalty, or no penalty if target is a lower bound.
    regime_1_mask = input >= target
    regime_1_loss = torch.where(
        is_lb,
        torch.zeros_like(input),
        torch.abs(torch.log(input) - torch.log(target)),
    )

    # Regime 2: Prediction (input) is between min_val and target. Apply Q-error
    # penalty.
    regime_2_mask = (input < target) & (input >= min_val)
    regime_2_loss = torch.abs(torch.log(target) - torch.log(input))

    # Regime 3: Prediction (input) is below min_val. Apply a linearly increasing
    # penalty, which passes through the following two points:
    #  (min_val, torch.abs(torch.log(min_val) - torch.log(target)))
    #  (0, torch.abs(torch.log(2*target) - torch.log(target)))
    regime_3_mask = input < min_val
    error_at_min_val = torch.abs(torch.log(target) - torch.log(min_val))
    error_at_2_target = torch.abs(torch.log(2 * target) - torch.log(target))
    regime_3_slope = (error_at_min_val - error_at_2_target) / (min_val)
    regime_3_loss = regime_3_slope * input + error_at_2_target

    loss = torch.where(
        regime_1_mask,
        regime_1_loss,
        torch.where(
            regime_2_mask,
            regime_2_loss,
            regime_3_loss,
        ),
    )

    if return_mean:
        return loss.mean().requires_grad_(True)
    else:
        return loss.requires_grad_(True)


def sensitive_q_error_loss_v4(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    target_is_lower_bound: torch.Tensor,
    return_mean: bool = True,
    **kwargs,  # To ignore unused parameters.
) -> torch.Tensor:

    input = input.flatten()
    target = target.flatten()
    is_lb = target_is_lower_bound.flatten()

    # Regime 1: Prediction (input) is larger than the target and the target is a
    # lower bound.
    regime_1_mask = (input >= target) & is_lb
    regime_1_loss = torch.zeros_like(input)

    # Regime 2: Prediction (input) is larger than an exact target, or smaller
    # than the target (regardless of whether it's a lower bound or exact).
    # Apply Q-error penalty.
    regime_2_loss = torch.abs(torch.log(target) - torch.log(input))

    loss = torch.where(
        regime_1_mask,
        regime_1_loss,
        regime_2_loss,
    )

    if return_mean:
        return loss.mean().requires_grad_(True)
    else:
        return loss.requires_grad_(True)

def sensitive_q_error_loss_v5(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    target_is_lower_bound: torch.Tensor,
    return_mean: bool = True,
    **kwargs,  # To ignore unused parameters.
) -> torch.Tensor:

    input = input.flatten()
    target = target.flatten()
    is_lb = target_is_lower_bound.flatten()

    # Regime 1: Prediction (input) is larger than the target and the target is a
    # lower bound.
    regime_1_mask = (input >= target) & is_lb
    regime_1_loss = torch.zeros_like(input)

    # Regime 2: Prediction (input) is larger than an exact target, or smaller
    # than the target (regardless of whether it's a lower bound or exact).
    # Apply Q-error penalty with Huber loss.
    r = torch.log(input) - torch.log(target)
    beta = np.log(1.5)
    regime_2_loss = F.smooth_l1_loss(r, torch.zeros_like(r), beta=beta)

    loss = torch.where(
        regime_1_mask,
        regime_1_loss,
        regime_2_loss,
    )

    if return_mean:
        return loss.mean().requires_grad_(True)
    else:
        return loss.requires_grad_(True)



def sensitive_q_error_loss_v10(
    input: torch.Tensor,  # pylint: disable=redefined-builtin
    target: torch.Tensor,
    target_is_lower_bound: torch.Tensor,
    min_val: float | torch.Tensor = 0.001,
    small_val: float = 5.0,
    penalty_negative: float = 1e5,
    lambda_small: float = 0.1,
    return_mean: bool = True,
) -> torch.Tensor:
    """
    Smooth version that matches v1 behavior per regime but uses soft gates at
    the transition points to avoid discontinuities.

    Regimes (same as v1):
    1. Negative/too-small inputs: penalty scaled by `penalty_negative`.
    2. Small values: L1 distance scaled by `lambda_small` (only when both input
       and target are below `small_val`).
    3. Otherwise: Q-error (absolute log difference).

    Differences vs v1:
    - Uses sigmoid-based gates to smoothly mix regimes instead of hard masks.
    - Small-loss gate is suppressed when the target is already large, matching
      v1's condition (target < small_val).

    Returns:
        Mean loss over the batch.
    """

    input = input.flatten()
    target = target.flatten()

    if isinstance(min_val, float):
        min_val = torch.full_like(input, fill_value=min_val)

    eps = 1e-8

    # Loss components match v1 behaviors
    negative_loss = (1 + torch.abs(input)) * penalty_negative
    small_loss = torch.abs(target - input) * lambda_small
    input_safe = torch.clamp(input, min=eps)
    target_safe = torch.clamp(target, min=eps)
    q_error = torch.abs(torch.log(input_safe) - torch.log(target_safe))

    # Smooth gates for transitions
    small_val_t = torch.as_tensor(
        small_val, device=input.device, dtype=input.dtype
    )
    neg_width = torch.clamp(
        0.2 * torch.abs(min_val) + 0.02 * small_val_t, min=1e-3
    )
    small_width = torch.clamp(0.02 * small_val_t, min=1e-3)
    target_width = torch.clamp(0.2 * small_val_t, min=1e-3)

    neg_gate = torch.sigmoid((min_val - input) / neg_width)
    small_gate_input = torch.sigmoid((small_val_t - input) / small_width)
    small_gate_target = torch.sigmoid((small_val_t - target) / target_width)
    small_gate = small_gate_input * small_gate_target

    raw_q_gate = 1.0 - neg_gate - small_gate
    q_gate = torch.clamp(raw_q_gate, min=0.0)

    total = neg_gate + small_gate + q_gate + eps
    neg_w = neg_gate / total
    small_w = small_gate / total
    q_w = q_gate / total

    loss = neg_w * negative_loss + small_w * small_loss + q_w * q_error

    if return_mean:
        return loss.mean().requires_grad_(True)
    else:
        return loss.requires_grad_(True)
