import torch

from autoslo.nn.loss_functions import (
    get_log_two_pi,
    negative_log_likelihood_loss,
    mdn_negative_log_likelihood_loss,
    sensitive_q_error_loss,
)


def test_negative_log_likelihood_loss_matches_manual():
    """Test mean NLL matches manual computation."""
    inputs = torch.tensor([0.0, 1.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 1.5], dtype=torch.float32)
    logvar = torch.tensor([0.0, 0.0], dtype=torch.float32)
    loss = negative_log_likelihood_loss(inputs, targets, logvar)
    logvar_clamped = torch.clamp(logvar, min=-25, max=25)
    var = torch.exp(logvar_clamped)
    log_two_pi = get_log_two_pi(inputs.device)
    nll = 0.5 * (
        logvar_clamped + ((inputs - targets) ** 2) / var + log_two_pi
    )
    expected = (nll + 0.01 * var).mean()
    assert torch.allclose(loss, expected)


def test_negative_log_likelihood_loss_per_sample():
    """Test NLL returns per-sample losses when return_mean is False."""
    inputs = torch.tensor([0.5, 2.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 1.5], dtype=torch.float32)
    logvar = torch.tensor([-0.5, 0.5], dtype=torch.float32)
    loss = negative_log_likelihood_loss(
        inputs, targets, logvar, return_mean=False
    )
    logvar_clamped = torch.clamp(logvar, min=-25, max=25)
    var = torch.exp(logvar_clamped)
    log_two_pi = get_log_two_pi(inputs.device)
    nll = 0.5 * (
        logvar_clamped + ((inputs - targets) ** 2) / var + log_two_pi
    )
    expected = nll + 0.01 * var
    assert loss.shape == expected.shape
    assert torch.allclose(loss, expected)


def test_mdn_negative_log_likelihood_loss_matches_manual():
    """Test MDN NLL matches manual computation."""
    means = torch.tensor([[0.0, 2.0], [1.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([1.0, 3.0], dtype=torch.float32)
    base_var = torch.tensor(
        [[1.0, 2.0], [1.5, 0.5]], dtype=torch.float32
    )
    logvar = torch.log(base_var)
    mix_coeffs = torch.tensor(
        [[0.3, 0.7], [1.2, -0.3]], dtype=torch.float32
    )
    loss = mdn_negative_log_likelihood_loss(
        means, targets, logvar, mix_coeffs
    )
    logvar_clamped = torch.clamp(logvar, min=-10, max=10)
    var = torch.exp(logvar_clamped)
    mix = torch.softmax(mix_coeffs, dim=1)
    expanded_targets = targets.unsqueeze(1).expand_as(means)
    log_two_pi = get_log_two_pi(means.device)
    log_normal = -0.5 * (
        logvar_clamped
        + ((means - expanded_targets) ** 2) / var
        + log_two_pi
    )
    nll = -torch.logsumexp(torch.log(mix) + log_normal, dim=1)
    nll = torch.clamp(nll, min=0.0)
    expected = (nll + 0.01 * var.sum(dim=1)).mean()
    assert torch.allclose(loss, expected)


def test_sensitive_q_error_loss_branch_behaviour():
    """Test sensitive Q loss blends negative, small, and q-error terms."""
    inputs = torch.tensor(
        [0.0005, 2.0, 10.0], dtype=torch.float32
    )
    targets = torch.tensor(
        [0.01, 3.0, 5.0], dtype=torch.float32
    )
    loss = sensitive_q_error_loss(inputs, targets)
    min_val = 0.001
    small_val = 5.0
    penalty = 1e5
    lambda_small = 0.1
    negative_mask = inputs < min_val
    negative_loss = (1 - inputs) * penalty
    small_mask = (inputs < small_val) & (targets < small_val)
    small_loss = torch.abs(targets - inputs) * lambda_small
    q_error = torch.where(
        inputs > targets,
        torch.log(inputs) - torch.log(targets),
        torch.log(targets) - torch.log(inputs),
    )
    combined = torch.where(
        negative_mask,
        negative_loss,
        torch.where(small_mask, small_loss, q_error),
    )
    expected = combined.mean()
    assert torch.allclose(loss, expected)
    assert loss.requires_grad
