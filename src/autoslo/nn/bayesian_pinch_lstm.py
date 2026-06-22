import torch
from torch import nn

from autoslo.nn.bayesian_lstm_cell import BayesianLSTMCell


class BayesianPinchLSTM(nn.Module):
    """
    Define a (possibly) Bayesian Bidirectional LSTM where each input sequence is
    split into two parts, according to a `pinch point`. The first part is
    processed by a forward LSTM, and the second part is processed by an
    after-direction LSTM.

    When ``forward_after=False`` (default): the after-direction LSTM processes
    the sequence in reverse chronological order (original behaviour).

    When ``forward_after=True``: the after-direction LSTM processes the sequence
    in forward chronological order, starting from the pinch point.  Positions
    before the pinch are zero-filled.  This mode enables O(1) incremental
    hidden-state resumption via ``get_forward_pinch_out``,
    ``get_after_final_state``, and ``step_after``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers_forward: int,
        num_layers_reverse: int,
        dropout: float = 0.0,
        is_bayesian: bool = True,
        forward_after: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the (possibly) Bayesian LSTM.

        Parameters:
            input_size: The number of input features.
            hidden_size: The number of hidden features.
            num_layers_forward: The number of layers in the forward LSTM.
            num_layers_reverse: The number of layers in the after-direction LSTM.
            dropout: The dropout probability.
            is_bayesian: Whether to use Bayesian linear layers.
            forward_after: When True, the after-direction LSTM processes future
                queries in forward chronological order instead of reverse.  See
                class docstring for details.
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
        self._forward_after = forward_after

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
            self._after_lstm = nn.LSTM(
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
        # and the remaining `num_layers_reverse` cells are for the after LSTM.

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        pinch_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass through the (possibly) Bayesian LSTM. This is
        "forward" in the deep learning sense — it involves both the forward-
        direction LSTM and the after-direction LSTM.

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

    def _build_x_after(
        self,
        x: torch.Tensor,
        pinch_points: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-fill positions before each sequence's pinch point.

        For batch element i, positions 0..pinch_points[i]-1 are set to zero so
        that the after-LSTM only attends to the base query (at the pinch) and
        its future neighbors.  Positions at and after the pinch are unchanged.
        """
        B, T, F = x.shape
        pos = (
            torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        )  # (B, T)
        before_pinch = pos < pinch_points.unsqueeze(1)  # (B, T)
        return x.masked_fill(before_pinch.unsqueeze(-1), 0.0)

    def _build_x_rev(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse each sequence within its valid length, keeping padding at
        the end.

        Vectorized implementation: a single gather + masked_fill rather than a
        Python loop over the batch dimension.  For a batch of size B this
        reduces kernel launches from O(B) to O(1).
        """
        B, T, F = x.shape
        lengths = x_len.to(x.device)  # (B,)

        # pos[i, j] = j  — position index broadcast across the batch
        pos = (
            torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        )  # (B, T)

        # rev_pos[i, j] = x_len[i] - 1 - j  for valid positions,
        #                 0                   for padding positions (clamped)
        rev_pos = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0)  # (B, T)

        # Single gather kernel: x_rev[i, j, k] = x[i, rev_pos[i, j], k]
        # At padding positions (j >= x_len[i]) this writes x[i, 0, k], which is
        # corrected to zero by the masked_fill below.
        x_rev = x.gather(1, rev_pos.unsqueeze(-1).expand(B, T, F))  # (B, T, F)

        # Zero out padding positions (j >= x_len[i]) across all features.
        pad_mask = pos >= lengths.unsqueeze(1)  # (B, T)
        x_rev = x_rev.masked_fill(pad_mask.unsqueeze(-1), 0.0)

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
            torch.arange(x.size(0), device=self._device), pinch_points
        ]  # Shape: (batch_size, 1d_hidden_size)

        # After direction
        outputs_a_final = torch.empty(
            (batch_size, self._1d_hidden_size), device=self._device
        )
        if self._forward_after:
            # Zero-fill before pinch, run in forward order, pick at last valid pos
            x_after = self._build_x_after(x, pinch_points)
            outputs_a, _ = self._after_lstm(x_after)
            outputs_a_final = outputs_a[
                torch.arange(batch_size, device=self._device),
                x_len - 1,
            ]  # Shape: (batch_size, 1d_hidden_size)
        else:
            # Reverse the full sequence, run, pick at the reverse-pinch index
            x_rev = self._build_x_rev(x, x_len)
            outputs_r, _ = self._after_lstm(x_rev)
            outputs_a_final = outputs_r[
                torch.arange(batch_size, device=self._device),
                x_len - 1 - pinch_points,
            ]  # Shape: (batch_size, 1d_hidden_size)

        return torch.cat(
            (outputs_f_final, outputs_a_final), dim=1
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

        # Before the after pass, build the input sequence for the after-LSTM.
        # forward_after=True: zero-fill before pinch, pick at x_len-1.
        # forward_after=False: reverse the full sequence, pick at L-1-pinch.
        if self._forward_after:
            x_proc = self._build_x_after(x, pinch_points)
            after_pick_idx = x_len - 1
        else:
            x_proc = self._build_x_rev(x, x_len)
            after_pick_idx = x_len - 1 - pinch_points

        # After/reverse pass.
        outputs_r = []

        for t in range(seq_len):
            x_proc_t = x_proc[:, t, :].squeeze(
                1
            )  # x_proc_t has shape (batch_size, input_size)
            for layer in range(
                self._num_layers_forward, self._num_layers_total
            ):
                h[layer], c[layer] = self._cells[layer](
                    x_proc_t, (h[layer], c[layer])
                )
                x_proc_t = self._dropout(h[layer])

            outputs_r.append(
                h[-1]
            )  # Each appended element has shape (batch_size, 1d_hidden_size)

        outputs_r_stacked = torch.stack(
            outputs_r, dim=1
        )  # `outputs_r_stacked` has shape (batch_size, seq_len, 1d_hidden_size)
        outputs_r_idx = (torch.arange(batch_size), after_pick_idx)
        outputs_r_final = outputs_r_stacked[
            outputs_r_idx
        ]  # `outputs_r_final` has shape (batch_size, 1d_hidden_size)

        return torch.cat(
            (outputs_f_final, outputs_r_final), dim=1
        )  # Shape (batch_size, hidden_size)

    # ── Stateful inference helpers (forward_after=True only) ──────────────────
    def pass_through_forward_lstm(
        self,
        x: torch.Tensor,
        pinch_points: torch.Tensor,
    ) -> torch.Tensor:
        """Run the forward-LSTM and return the hidden output at each pinch point.

        Parameters:
            x: Embedded input of shape (batch_size, seq_len, embedding_size).
            pinch_points: Shape (batch_size,).

        Returns:
            fwd_out: (batch_size, 1d_hidden_size) — forward output at pinch.

        Only valid when is_bayesian=False and forward_after=True.
        """
        s = (
            "pass_through_forward_lstm requires is_bayesian=False and "
            "forward_after=True"
        )
        assert not self._is_bayesian and self._forward_after, s
        outputs_f, _ = self._forward_lstm(x)
        return outputs_f[
            torch.arange(x.size(0), device=x.device), pinch_points
        ]  # (batch_size, 1d_hidden_size)

    def pass_through_after_lstm(
        self,
        x: torch.Tensor,
        pinch_points: torch.Tensor,
        x_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the after-LSTM (zero-filled before pinch) and return the state
        at the last position.

        Parameters:
            x: Embedded input of shape (batch_size, seq_len, embedding_size).
                May be zero-padded when *x_len* is provided.
            pinch_points: Shape (batch_size,).
            x_len: True sequence lengths, shape (batch_size,).  When provided,
                ``pack_padded_sequence`` is used so that the returned states
                correspond to the last *valid* timestep of each sequence,
                enabling batched calls over variable-length inputs.  When
                ``None`` the input is assumed to be unpadded (original
                single-query behaviour).

        Returns:
            after_out: (batch_size, 1d_hidden_size) — output at the last position.
            after_h:   (num_layers, batch_size, 1d_hidden_size) — hidden state.
            after_c:   (num_layers, batch_size, 1d_hidden_size) — cell state.

        Only valid when is_bayesian=False and forward_after=True.
        """
        s = (
            "pass_through_after_lstm requires is_bayesian=False and "
            "forward_after=True"
        )
        assert not self._is_bayesian and self._forward_after, s
        x_after = self._build_x_after(x, pinch_points)
        if x_len is not None:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_after, x_len.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (after_h, after_c) = self._after_lstm(packed)
        else:
            _, (after_h, after_c) = self._after_lstm(x_after)
        # after_h[-1] is correct whether the input was packed (variable-length)
        # or unpadded (single-query): PyTorch returns the state at each
        # sequence's last valid timestep.
        after_out = after_h[-1]  # (batch_size, 1d_hidden_size)
        return after_out, after_h, after_c

    def step_after(
        self,
        x_token: torch.Tensor,
        after_h: torch.Tensor,
        after_c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single step of the after-LSTM from a saved state.

        Parameters:
            x_token: Embedded token of shape (batch_size, embedding_size).
            after_h: Saved hidden state, shape (num_layers, batch_size, 1d_hidden_size).
            after_c: Saved cell state,   shape (num_layers, batch_size, 1d_hidden_size).

        Returns:
            after_out: (batch_size, 1d_hidden_size) — output at this step.
            new_h:     (num_layers, batch_size, 1d_hidden_size) — updated hidden state.
            new_c:     (num_layers, batch_size, 1d_hidden_size) — updated cell state.

        Only valid when is_bayesian=False and forward_after=True.
        """
        s = (
            "step_after requires is_bayesian=False and "
            "forward_after=True"
        )
        assert not self._is_bayesian and self._forward_after, s
        output, (new_h, new_c) = self._after_lstm(
            x_token.unsqueeze(1),  # (batch_size, 1, embedding_size)
            (after_h, after_c),
        )
        return output[:, 0, :], new_h, new_c
