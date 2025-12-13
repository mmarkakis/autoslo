from typing import Optional

import torch
from torch import nn

from autoslo.nn.bayesian_linear import BayesianLinear


class BayesianLSTMCell(nn.Module):
    """
    Define a (possibly) Bayesian LSTM cell.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        is_bayesian: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Bayesian LSTM cell. This is the same as a standard LSTM 
        cell, but with Bayesian linear layers.

        Parameters:
            input_size: The number of input features.
            hidden_size: The number of hidden features.
            is_bayesian: Whether to use Bayesian linear layers.
            device: The device to use for the model.
        """
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._device = device

        linear_layer = BayesianLinear if is_bayesian else nn.Linear

        self._input_gate = linear_layer(
            input_size + hidden_size, hidden_size, device=self._device
        ).to(self._device)
        self._forget_gate = linear_layer(
            input_size + hidden_size, hidden_size, device=self._device
        ).to(self._device)
        self._output_gate = linear_layer(
            input_size + hidden_size, hidden_size, device=self._device
        ).to(self._device)
        self._cell_gate = linear_layer(
            input_size + hidden_size, hidden_size, device=self._device
        ).to(self._device)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass through the Bayesian LSTM cell. This is the 
        same as a forward pass through a standard LSTM cell, but with Bayesian 
        linear layers.

        Parameters:
            x: The input tensor, of shape (batch_size, input_size).
            hidden: The hidden state tuple, containing the hidden state and the 
                cell state. Each of these tensors is of shape 
                (batch_size, hidden_size). If `hidden` is `None`, the hidden 
                state is initialized to zeros.

        Returns:
            h: The hidden state tensor.
            c: The cell state tensor
        """
        batch_size, input_size = x.size()
        assert input_size == self._input_size

        if hidden is None:
            h_prev = x.new_zeros((batch_size, self._hidden_size)).to(
                self._device
            )
            c_prev = x.new_zeros((batch_size, self._hidden_size)).to(
                self._device
            )
            hidden = (h_prev, c_prev)

        h_prev, c_prev = hidden
        full_input = torch.cat((x, h_prev), 1)

        # Calculate gate outputs
        i = torch.sigmoid(self._input_gate(full_input))
        f = torch.sigmoid(self._forget_gate(full_input))
        o = torch.sigmoid(self._output_gate(full_input))
        g = torch.tanh(self._cell_gate(full_input))

        # Calculate new cell and hidden states
        c = f * c_prev.clone() + i * g
        h = o * torch.tanh(c)

        return h, c
