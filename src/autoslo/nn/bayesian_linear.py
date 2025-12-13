import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


class BayesianLinear(nn.Module):
    """
    Define a Bayesian Linear layer with normal prior and normal posterior.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Bayesian Linear layer with normal prior and normal
        posterior.

        Parameters:
            in_features: The number of input features.
            out_features: The number of output features.
            device: The device to use for the model.
        """

        super().__init__()
        self._in_features = in_features
        self._out_features = out_features

        self._weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).to(device)
        )
        self._weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).to(device)
        )
        self._bias_mu = nn.Parameter(torch.Tensor(out_features).to(device))
        self._bias_rho = nn.Parameter(torch.Tensor(out_features).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the Bayesian Linear layer.
        """

        nn.init.xavier_normal_(self._weight_mu)
        nn.init.constant_(self._weight_rho, -3.0)
        nn.init.uniform_(self._bias_mu, -0.1, 0.1)
        nn.init.constant_(self._bias_rho, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the Bayesian Linear layer.

        Parameters:
            x: The input tensor, of shape (batch_size, in_features).

        Returns:
            The output tensor.
        """

        weight_dist = Normal(self._weight_mu, F.softplus(self._weight_rho))
        bias_dist = Normal(self._bias_mu, F.softplus(self._bias_rho))

        weight = weight_dist.rsample()
        bias = bias_dist.rsample()
        return F.linear(x, weight, bias)

    def forward_mc(self, x: torch.Tensor, mc_samples: int = 1) -> torch.Tensor:
        """
        Perform the forward pass through the Bayesian Linear layer, possibly
        with multiple Monte Carlo samples.

        Parameters:
            x: The input tensor, of shape (batch_size, in_features).
            mc_samples: The number of Monte Carlo samples to use.

        Returns:
            The output tensor. If mc_samples == 1, the output has shape
                (batch_size, out_features).
            If mc_samples > 1, the output has shape
                (mc_samples, batch_size, out_features).
        """
        weight_sigma = F.softplus(self._weight_rho)
        bias_sigma = F.softplus(self._bias_rho)

        if mc_samples == 1:
            weight = self._weight_mu + weight_sigma * torch.randn_like(
                self._weight_mu
            )
            bias = self._bias_mu + bias_sigma * torch.randn_like(self._bias_mu)
            return F.linear(x, weight, bias)

        # Expand x to include a Monte Carlo sample dimension.
        # x has shape (batch_size, in_features) and becomes
        # (mc_samples, batch_size, in_features)
        x_expanded = x.unsqueeze(0).expand(mc_samples, -1, -1)

        # Sample weights and biases independently for each Monte Carlo sample.
        # The resulting shapes are:
        #   weight: (mc_samples, out_features, in_features)
        #   bias: (mc_samples, out_features)
        weight = self._weight_mu + weight_sigma * torch.randn(
            mc_samples, *self._weight_mu.shape, device=x.device
        )
        bias = self._bias_mu + bias_sigma * torch.randn(
            mc_samples, *self._bias_mu.shape, device=x.device
        )

        # Perform batched linear transformation:
        #   For each sample, we need to compute x * weight^T.
        # x_expanded: (mc_samples, batch_size, in_features)
        # weight.transpose(1, 2): (mc_samples, in_features, out_features)
        # torch.bmm returns (mc_samples, batch_size, out_features)
        out = torch.bmm(x_expanded, weight.transpose(1, 2))

        # Add the bias to each output: bias is (mc_samples, out_features) and is
        # unsqueezed to (mc_samples, 1, out_features)
        out = out + bias.unsqueeze(1)

        return out
