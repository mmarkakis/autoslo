import torch

from autoslo.nn.bayesian_lstm_cell import BayesianLSTMCell

# pylint: disable=protected-access

# FIXME: These have not been updated after the codebase migration.


class TestBayesianLSTMCell:
    """
    Test the BayesianLSTMCell class.
    """

    def test_initialization_shapes(self):
        """
        Test the shapes related to the initialization of the BayesianLSTMCell
        class.
        """

        cell = BayesianLSTMCell(10, 5)
        assert cell._input_size == 10
        assert cell._hidden_size == 5

    def test_forward_shapes(self):
        """
        Test the shapes related to the forward pass of the BayesianLSTMCell
        class.
        """

        cell = BayesianLSTMCell(10, 5)
        input_tensor = torch.randn(2, 10)
        h = torch.randn(2, 5)
        c = torch.randn(2, 5)
        h_next, c_next = cell(input_tensor, (h, c))
        assert h_next.shape == (2, 5)
        assert c_next.shape == (2, 5)
