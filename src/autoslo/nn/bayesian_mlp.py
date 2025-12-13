import torch
from torch import nn

from autoslo.nn.bayesian_linear import BayesianLinear


class BayesianMLP(nn.Module):
    """
    Define a (possibly) Bayesian MLP.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        is_bayesian: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Bayesian MLP. This is the same as a standard MLP, but
        with Bayesian linear layers.

        Parameters:
            input_size: The number of input features.
            hidden_size: The number of hidden features.
            output_size: The number of output features.
            num_layers: The number of layers in the MLP.
            is_bayesian: Whether to use Bayesian linear layers.
            device: The device to use for the model.
        """
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._device = device

        linear_layer = BayesianLinear if is_bayesian else nn.Linear

        layers: list[nn.Module] = []

        # Input layer
        layers.append(linear_layer(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(linear_layer(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(linear_layer(hidden_size, output_size))

        self._mlp = nn.Sequential(*layers)

        self._mlp.to(self._device)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """
        Perform the forward pass through the Bayesian MLP. This is the same as a
        forward pass through a standardMLP, but with Bayesian linear layers.

        Parameters:
            x: The input tensor, of shape (batch_size, input_size).

        Returns:
            The output tensor.
        """
        _, input_size = x.size()
        assert input_size == self._input_size

        return self._mlp(x)
