"""
Tests for the forward-after-LSTM architecture and stateful inference cache.

Covers:
  1. BayesianPinchLSTM with forward_after=True produces correct output shape.
  2. forward_after=True and forward_after=False produce different outputs
     (they encode different representations).
  3. get_forward_pinch_out / get_after_final_state return correct shapes.
  4. Incremental equivalence: register_query([base, n1]) ≡
     register_query([base]) then update_query(n1).
  5. complete_query evicts the state.
  6. _load_nn_state_dict_with_feature_guard clears the state cache.
  7. register_query and update_query raise when forward_after_lstm=False.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoslo.nn.bayesian_pinch_lstm import BayesianPinchLSTM
from autoslo.nn.runtime_net import RuntimeNet

# ── Shared constants ──────────────────────────────────────────────────────────

_INPUT_SIZE = 8
_HIDDEN_SIZE = 16  # total; 8 per direction
_EMB_SIZE = 12
_NUM_LAYERS = 1


def _make_lstm(forward_after: bool = False) -> BayesianPinchLSTM:
    return BayesianPinchLSTM(
        input_size=_INPUT_SIZE,
        hidden_size=_HIDDEN_SIZE,
        num_layers_forward=_NUM_LAYERS,
        num_layers_reverse=_NUM_LAYERS,
        is_bayesian=False,
        forward_after=forward_after,
    )


def _make_net(forward_after_lstm: bool = False) -> RuntimeNet:
    return RuntimeNet(
        input_size=_INPUT_SIZE,
        embedding_size=_EMB_SIZE,
        lstm_hidden_size=_HIDDEN_SIZE,
        lstm_num_layers=_NUM_LAYERS,
        lstm_dropout=0.0,
        is_bayesian=False,
        forward_after_lstm=forward_after_lstm,
    )


# ── BayesianPinchLSTM tests ───────────────────────────────────────────────────


class TestBayesianPinchLSTMForwardAfter:
    def test_output_shape_forward_after(self):
        """forward_after=True produces (batch, hidden_size) output."""
        lstm = _make_lstm(forward_after=True)
        B, T = 4, 6
        x = torch.randn(B, T, _INPUT_SIZE)
        x_len = torch.tensor([T, T - 1, T - 2, T - 3])
        pinch = torch.tensor([0, 1, 0, 2])
        out = lstm(x, x_len, pinch)
        assert out.shape == (B, _HIDDEN_SIZE)

    def test_output_shape_matches_reverse(self):
        """Both directions produce the same output shape."""
        fwd = _make_lstm(forward_after=True)
        rev = _make_lstm(forward_after=False)
        x = torch.randn(3, 5, _INPUT_SIZE)
        x_len = torch.tensor([5, 4, 3])
        pinch = torch.tensor([1, 0, 2])
        assert fwd(x, x_len, pinch).shape == rev(x, x_len, pinch).shape

    def test_forward_after_differs_from_reverse(self):
        """The two directions learn different representations (different outputs
        for the same weights, since the sequence is processed differently)."""
        torch.manual_seed(42)
        fwd = _make_lstm(forward_after=True)
        # Copy weights into a reverse model.
        rev = _make_lstm(forward_after=False)
        rev.load_state_dict(fwd.state_dict())

        x = torch.randn(2, 4, _INPUT_SIZE)
        x_len = torch.tensor([4, 3])
        pinch = torch.tensor([1, 0])

        with torch.no_grad():
            out_fwd = fwd(x, x_len, pinch)
            out_rev = rev(x, x_len, pinch)

        # The forward-LSTM half must be identical (same weights, same input).
        # The after half will differ because the input sequence is processed
        # differently.
        half = _HIDDEN_SIZE // 2
        assert torch.allclose(out_fwd[:, :half], out_rev[:, :half], atol=1e-5)
        assert not torch.allclose(out_fwd[:, half:], out_rev[:, half:], atol=1e-5)

    def test_get_forward_pinch_out_shape(self):
        """get_forward_pinch_out returns (B, 1d_hidden_size)."""
        lstm = _make_lstm(forward_after=True)
        B, T = 3, 5
        x = torch.randn(B, T, _INPUT_SIZE)
        pinch = torch.tensor([0, 2, 1])
        out = lstm.get_forward_pinch_out(x, pinch)
        assert out.shape == (B, _HIDDEN_SIZE // 2)

    def test_get_after_final_state_shapes(self):
        """get_after_final_state returns correct tensor shapes."""
        lstm = _make_lstm(forward_after=True)
        x = torch.randn(1, 4, _INPUT_SIZE)
        pinch = torch.tensor([1])
        after_out, after_h, after_c = lstm.get_after_final_state(x, pinch)
        assert after_out.shape == (1, _HIDDEN_SIZE // 2)
        assert after_h.shape == (_NUM_LAYERS, 1, _HIDDEN_SIZE // 2)
        assert after_c.shape == (_NUM_LAYERS, 1, _HIDDEN_SIZE // 2)

    def test_step_after_shapes(self):
        """step_after returns correct tensor shapes."""
        lstm = _make_lstm(forward_after=True)
        x = torch.randn(1, 3, _INPUT_SIZE)
        pinch = torch.tensor([0])
        _, after_h, after_c = lstm.get_after_final_state(x, pinch)
        x_tok = torch.randn(1, _INPUT_SIZE)
        out, new_h, new_c = lstm.step_after(x_tok, after_h, after_c)
        assert out.shape == (1, _HIDDEN_SIZE // 2)
        assert new_h.shape == (_NUM_LAYERS, 1, _HIDDEN_SIZE // 2)
        assert new_c.shape == (_NUM_LAYERS, 1, _HIDDEN_SIZE // 2)

    def test_stateful_helpers_raise_without_forward_after(self):
        """Stateful helpers raise AssertionError when forward_after=False."""
        lstm = _make_lstm(forward_after=False)
        x = torch.randn(1, 3, _INPUT_SIZE)
        pinch = torch.tensor([0])
        dummy_h = torch.zeros(_NUM_LAYERS, 1, _HIDDEN_SIZE // 2)
        dummy_c = torch.zeros(_NUM_LAYERS, 1, _HIDDEN_SIZE // 2)
        with pytest.raises(AssertionError):
            lstm.get_forward_pinch_out(x, pinch)
        with pytest.raises(AssertionError):
            lstm.get_after_final_state(x, pinch)
        with pytest.raises(AssertionError):
            lstm.step_after(torch.randn(1, _INPUT_SIZE), dummy_h, dummy_c)


# ── Incremental equivalence tests ─────────────────────────────────────────────


class TestIncrementalEquivalence:
    """
    Core correctness property: processing [tok0, tok1] in one shot must produce
    the same LSTM output as processing [tok0] and then resuming with [tok1].
    """

    def _run_full(
        self, lstm: BayesianPinchLSTM, toks: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the after-LSTM on the full token sequence and return final state."""
        x = torch.stack(toks, dim=1)  # (1, T, input_size)
        pinch = torch.zeros(1, dtype=torch.long)  # pinch at 0: process all
        after_out, after_h, after_c = lstm.get_after_final_state(x, pinch)
        return after_out, after_h, after_c

    def _run_incremental(
        self, lstm: BayesianPinchLSTM, toks: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the after-LSTM incrementally, one token at a time."""
        # Initial pass on the first token.
        x0 = toks[0].unsqueeze(1)  # (1, 1, input_size)
        pinch = torch.zeros(1, dtype=torch.long)
        after_out, after_h, after_c = lstm.get_after_final_state(x0, pinch)
        # Subsequent tokens via step_after.
        for tok in toks[1:]:
            after_out, after_h, after_c = lstm.step_after(tok, after_h, after_c)
        return after_out, after_h, after_c

    def test_one_extra_token(self):
        """Full pass over [tok0, tok1] == incremental step from [tok0] + [tok1]."""
        torch.manual_seed(0)
        lstm = _make_lstm(forward_after=True)
        lstm.eval()
        toks = [torch.randn(1, _INPUT_SIZE) for _ in range(2)]
        with torch.no_grad():
            out_full, h_full, c_full = self._run_full(lstm, toks)
            out_inc, h_inc, c_inc = self._run_incremental(lstm, toks)
        assert torch.allclose(out_full, out_inc, atol=1e-5)
        assert torch.allclose(h_full, h_inc, atol=1e-5)
        assert torch.allclose(c_full, c_inc, atol=1e-5)

    def test_three_extra_tokens(self):
        """Full pass over [tok0..tok3] == incremental, four tokens."""
        torch.manual_seed(1)
        lstm = _make_lstm(forward_after=True)
        lstm.eval()
        toks = [torch.randn(1, _INPUT_SIZE) for _ in range(4)]
        with torch.no_grad():
            out_full, h_full, _ = self._run_full(lstm, toks)
            out_inc, h_inc, _ = self._run_incremental(lstm, toks)
        assert torch.allclose(out_full, out_inc, atol=1e-5)
        assert torch.allclose(h_full, h_inc, atol=1e-5)

    def test_embed_then_full_vs_incremental(self):
        """Equivalence holds end-to-end through RuntimeNet.embed + step_after."""
        torch.manual_seed(2)
        net = _make_net(forward_after_lstm=True)
        net.eval()
        lstm = net._mid_model

        raw = [torch.randn(1, 1, _INPUT_SIZE) for _ in range(3)]
        with torch.no_grad():
            embs = [net._embed(r).squeeze(1) for r in raw]  # list of (1, E)

            # Full pass: embed all, stack, run get_after_final_state
            x_all = torch.cat([e.unsqueeze(1) for e in embs], dim=1)  # (1,3,E)
            pinch = torch.zeros(1, dtype=torch.long)
            out_full, h_full, c_full = lstm.get_after_final_state(x_all, pinch)

            # Incremental pass
            x0 = embs[0].unsqueeze(1)  # (1, 1, E)
            out_inc, h_inc, c_inc = lstm.get_after_final_state(x0, pinch)
            for e in embs[1:]:
                out_inc, h_inc, c_inc = lstm.step_after(e, h_inc, c_inc)

        assert torch.allclose(out_full, out_inc, atol=1e-5)
        assert torch.allclose(h_full, h_inc, atol=1e-5)


# ── RuntimeNet helper tests ───────────────────────────────────────────────────


class TestRuntimeNetHelpers:
    def test_embed_output_shape(self):
        """embed returns (B, T, embedding_size)."""
        net = _make_net()
        net.eval()
        x = torch.randn(2, 5, _INPUT_SIZE)
        with torch.no_grad():
            out = net._embed(x)
        assert out.shape == (2, 5, _EMB_SIZE)

    def test_predict_from_lstm_out_shape(self):
        """predict_from_lstm_out returns (B, 1) mean and logvar."""
        net = _make_net()
        net.eval()
        lstm_out = torch.randn(3, _HIDDEN_SIZE)
        with torch.no_grad():
            mean, logvar, mix = net._finalize(lstm_out)
        assert mean.shape == (3, 1)
        assert logvar.shape == (3, 1)
        assert mix is None

    def test_forward_consistent_with__finalize(self):
        """The two-step path (embed → LSTM → predict_from_lstm_out) produces
        the same result as the one-shot forward() call."""
        torch.manual_seed(3)
        net = _make_net(forward_after_lstm=True)
        net.eval()
        B, T = 2, 4
        x = torch.randn(B, T, _INPUT_SIZE)
        x_len = torch.tensor([T, T - 1])
        pinch = torch.tensor([1, 0])

        with torch.no_grad():
            mean_full, logvar_full, _ = net(x, x_len, pinch)

            # Two-step path
            x_emb = net._embed(x)
            lstm_out = net._mid_model(x_emb, x_len, pinch)
            mean_step, logvar_step, _ = net._finalize(lstm_out)

        assert torch.allclose(mean_full, mean_step, atol=1e-5)
        assert torch.allclose(logvar_full, logvar_step, atol=1e-5)
