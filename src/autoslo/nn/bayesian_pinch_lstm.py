import torch
from torch import nn

from autoslo.nn.bayesian_lstm_cell import BayesianLSTMCell


class BayesianPinchLSTM(nn.Module):
    """
    Define a (possibly) Bayesian Bidirectional LSTM where each input sequence is
    split into two parts, according to a `pinch point`. The first part is
    processed by a forward LSTM, and the second part is processed by a reverse
    LSTM. The outputs of the two LSTMs are concatenated and returned.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers_forward: int,
        num_layers_reverse: int,
        dropout: float = 0.0,
        is_bayesian: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the (possibly) Bayesian LSTM.

        Parameters:
            input_size: The number of input features.
            hidden_size: The number of hidden features.
            num_layers_forward: The number of layers in the forward LSTM.
            num_layers_reverse: The number of layers in the reverse LSTM.
            dropout: The dropout probability.
            is_bayesian: Whether to use Bayesian linear layers.
            device: The device to use for the model.
        """

        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._1d_hidden_size = hidden_size // 2
        self._num_layers_forward = num_layers_forward
        self._num_layers_reverse = num_layers_reverse
        self._dropout = nn.Dropout(dropout)
        self._num_layers_total = num_layers_forward + num_layers_reverse
        self._device = device
        self._is_bayesian = is_bayesian

        if not is_bayesian:
            # Use PyTorch's fused nn.LSTM kernel — far faster than a Python
            # timestep loop over individual BayesianLSTMCells.
            self._forward_lstm = nn.LSTM(
                input_size=self._input_size,
                hidden_size=self._1d_hidden_size,
                num_layers=self._num_layers_forward,
                batch_first=True,
                dropout=dropout if num_layers_forward > 1 else 0.0,
            ).to(device)
            self._reverse_lstm = nn.LSTM(
                input_size=self._input_size,
                hidden_size=self._1d_hidden_size,
                num_layers=self._num_layers_reverse,
                batch_first=True,
                dropout=dropout if num_layers_reverse > 1 else 0.0,
            ).to(device)
        else:
            self._cells = nn.ModuleList(
                [
                    BayesianLSTMCell(
                        input_size=(
                            self._input_size
                            if (
                                i in [0, num_layers_forward]
                            )  # First layer per direction.
                            else self._1d_hidden_size
                        ),
                        hidden_size=self._1d_hidden_size,
                        is_bayesian=is_bayesian,
                        device=self._device,
                    ).to(self._device)
                    for i in range(self._num_layers_total)
                ]
            )
        # N.B: The first `num_layers_forward` cells are for the forward LSTM, 
        # and the remaining `num_layers_reverse` cells are for the reverse LSTM.

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        pinch_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass through the (possibly) Bayesian LSTM. This is 
        "forward" in the deep learning sense - note that it involves both the 
        forward and reverse pass of the pinch LSTM, as described above.

        Parameters:
            x: The input tensor, of shape (batch_size, seq_len, input_size).
            x_len: The actual lengths of the sequences in the batch, even though 
                they are 0-padded to seq_len in `x`. This tensor has shape 
                (batch_size,).
            pinch_points: The indices of the pinch points in each of the 
                sequences in the batch. This tensor has shape (batch_size,).

        Returns:
            outputs: The output tensor, of shape (batch_size, hidden_size).
        """
        batch_size, seq_len, input_size = x.size()
        assert input_size == self._input_size
        assert x_len.size() == (batch_size,)

        if not self._is_bayesian:
            return self._forward_fused(x, x_len, pinch_points)
        return self._forward_cell(x, x_len, pinch_points)

    def _build_x_rev(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse each sequence within its valid length, keeping padding at
        the end."""
        batch_size, seq_len, _ = x.size()
        x_rev = x.clone()
        idx = torch.arange(seq_len, device=self._device).unsqueeze(0).expand(
            batch_size, -1
        )  # Shape (batch_size, seq_len)
        mask = idx < x_len.unsqueeze(1)  # Shape (batch_size, seq_len)
        for i in range(batch_size):
            x_rev[i, mask[i]] = x[i, mask[i]].flip(0)
        return x_rev

    def _forward_fused(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        pinch_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fast path using PyTorch's fused nn.LSTM kernel (no Python timestep
        loop). Used when is_bayesian=False.
        """
        batch_size = x.size(0)

        # Forward direction: run full sequence, pick hidden state at pinch point
        outputs_f, _ = self._forward_lstm(x)
        # outputs_f shape: (batch_size, seq_len, 1d_hidden_size)
        outputs_f_final = outputs_f[
            torch.arange(batch_size, device=self._device), pinch_points
        ]  # Shape: (batch_size, 1d_hidden_size)

        # Reverse direction: reverse each sequence, run, pick at reverse index
        x_rev = self._build_x_rev(x, x_len)
        outputs_r, _ = self._reverse_lstm(x_rev)
        # outputs_r shape: (batch_size, seq_len, 1d_hidden_size)
        outputs_r_final = outputs_r[
            torch.arange(batch_size, device=self._device),
            x_len - 1 - pinch_points,
        ]  # Shape: (batch_size, 1d_hidden_size)

        return torch.cat(
            (outputs_f_final, outputs_r_final), dim=1
        )  # Shape: (batch_size, hidden_size)

    def _forward_cell(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        pinch_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bayesian path: Python timestep loop over BayesianLSTMCells. Used when
        is_bayesian=True.
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden and cell states
        h = x.new_zeros(
            (self._num_layers_total, batch_size, self._1d_hidden_size)
        ).to(self._device)
        c = x.new_zeros(
            (self._num_layers_total, batch_size, self._1d_hidden_size)
        ).to(self._device)

        # Forward pass - The final effect should be to create a tensor of shape 
        # (batch_size, 1d_hidden_size), where each element is the hidden state 
        # of the forward LSTM after processing the pinch point (for each 
        # batch/sequence).
        outputs_f = []

        for t in range(seq_len):
            x_t = x[:, t, :].squeeze(
                1
            )  # x_t has shape (batch_size, input_size)
            for layer in range(self._num_layers_forward):
                h[layer], c[layer] = self._cells[layer](
                    x_t, (h[layer], c[layer])
                )
                x_t = self._dropout(h[layer])

            outputs_f.append(
                h[self._num_layers_forward - 1]
            )  # Each appended element has shape (batch_size, 1d_hidden_size)

        outputs_f_stacked = torch.stack(
            outputs_f, dim=1
        )  # `outputs_f_stacked` has shape (batch_size, seq_len, 1d_hidden_size)
        outputs_f_idx = (torch.arange(batch_size), pinch_points)
        outputs_f_final = outputs_f_stacked[
            outputs_f_idx
        ]  # `outputs_f_final` has shape (batch_size, 1d_hidden_size)

        # Before the reverse pass, we need some work. The problem is that the 
        # sequences are zero-padded so we can't just start processing from the 
        # back. The easiest way is to unpad, reverse, repad the sequences, and 
        # then process them "in order".
        x_rev = self._build_x_rev(x, x_len)

        # Reverse pass - The final effect should be to create a tensor of shape 
        # (batch_size, 1d_hidden_size), where each element is the hidden state 
        # of the reverse LSTM after processing the pinch point (for each 
        # batch/sequence).
        outputs_r = []

        for t in range(seq_len):
            x_rev_t = x_rev[:, t, :].squeeze(
                1
            )  # x_rev_t has shape (batch_size, input_size)
            for layer in range(
                self._num_layers_forward, self._num_layers_total
            ):
                h[layer], c[layer] = self._cells[layer](
                    x_rev_t, (h[layer], c[layer])
                )
                x_rev_t = self._dropout(h[layer])

            outputs_r.append(
                h[-1]
            )  # Each appended element has shape (batch_size, 1d_hidden_size)

        outputs_r_stacked = torch.stack(
            outputs_r, dim=1
        )  # `outputs_r_stacked` has shape (batch_size, seq_len, 1d_hidden_size)
        outputs_r_idx = (torch.arange(batch_size), x_len - 1 - pinch_points)
        outputs_r_final = outputs_r_stacked[
            outputs_r_idx
        ]  # `outputs_r_final` has shape (batch_size, 1d_hidden_size)

        return torch.cat(
            (outputs_f_final, outputs_r_final), dim=1
        )  # Shape (batch_size, hidden_size)
