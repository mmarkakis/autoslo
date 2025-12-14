import random

import torch

from autoslo.nn.bayesian_pinch_lstm import BayesianPinchLSTM

# pylint: disable=protected-access

# FIXME: These have not been updated after the codebase migration.


class TestBayesianPinchLSTM:
    """
    Test the BayesianPinchLSTM class.
    """

    def test_initialization_shapes(self):
        """
        Test the shapes related to the initialization of the BayesianPinchLSTM
        class.
        """

        lstm = BayesianPinchLSTM(10, 5, 2, 2)
        assert lstm._cells[0]._input_size == 10
        assert lstm._hidden_size == 5
        assert lstm._num_layers_forward == 2
        assert lstm._num_layers_reverse == 2

    def test_forward_shapes(self):
        """
        Test the shapes related to the forward pass of the BayesianPinchLSTM
        class.
        """

        input_size = 10
        hidden_size = 6
        num_layers = 2
        batch_size = 3
        seq_len = 4
        x_len = torch.tensor(
            [random.randint(1, seq_len) for _ in range(batch_size)],
            dtype=torch.long,
        )
        pinch_points = torch.tensor(
            [random.randint(0, x_len[i] - 1) for i in range(batch_size)],
            dtype=torch.long,
        )

        lstm = BayesianPinchLSTM(
            input_size, hidden_size, num_layers, num_layers
        )
        input_tensor = torch.randn(batch_size, seq_len, input_size)
        output_tensor = lstm(input_tensor, x_len, pinch_points)
        assert output_tensor.shape == (batch_size, hidden_size)
