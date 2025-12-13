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
        x_rev = x.clone()
        idx = (torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)).to(
            self._device
        )  # Shape (batch_size, seq_len)
        mask = idx < x_len.unsqueeze(1)  # Shape (batch_size, seq_len)

        for i in range(batch_size):
            x_rev[i, mask[i]] = x[i, mask[i]].flip(0)

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

        outputs = torch.cat(
            (outputs_f_final, outputs_r_final), dim=1
        )  # Shape (batch_size, hidden_size)

        return outputs
