import random

import torch

# from src.squire.models.lstm_model import ConcurrentQueryDataset, LSTMModel
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from autoslo.nn.runtime_net import RuntimeNet  # negative_log_likelihood_loss,

# pylint: disable=protected-access

# FIXME: These have not been updated after the codebase migration.


class TestBayesianRuntimeNet:
    """
    Test the BayesianRuntimeNet class.
    """

    def test_initialization_shapes(self):
        """
        Test the shapes related to the initialization of the BayesianRuntimeNet
        class.
        """

        input_size = 53
        embedding_size = 256
        lstm_hidden_size = 512
        lstm_num_layers = 3
        lstm_dropout = 0.3

        model = RuntimeNet(
            input_size=input_size,
            embedding_size=embedding_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
        )

        assert model._input_size == input_size
        assert model._embedding_size == embedding_size
        assert model._lstm_hidden_size == lstm_hidden_size
        assert model._lstm_num_layers == lstm_num_layers
        assert model._lstm_dropout == lstm_dropout

        assert model._bn.num_features == input_size
        assert model._in_model[0]._in_features == input_size
        assert model._in_model[0]._out_features == embedding_size
        assert model._in_model[1]._in_features == embedding_size
        assert model._in_model[1]._out_features == embedding_size
        assert model._mid_model._input_size == embedding_size
        assert model._mid_model._hidden_size == lstm_hidden_size
        assert model._mid_model._num_layers_forward == lstm_num_layers
        assert model._mid_model._num_layers_reverse == lstm_num_layers
        assert model._mid_model._dropout.p == lstm_dropout
        assert model._mean_out_model[0]._in_features == lstm_hidden_size
        assert model._mean_out_model[0]._out_features == lstm_hidden_size // 2
        assert model._mean_out_model[1]._in_features == lstm_hidden_size // 2
        assert model._mean_out_model[1]._out_features == 1
        assert model._logvar_out_model[0]._in_features == lstm_hidden_size
        assert model._logvar_out_model[0]._out_features == lstm_hidden_size // 2
        assert model._logvar_out_model[1]._in_features == lstm_hidden_size // 2
        assert model._logvar_out_model[1]._out_features == 1

    def test_forward_shapes(self):
        """
        Test the shapes related to the forward pass of the BayesianRuntimeNet
        class.
        """

        input_size = 50
        embedding_size = 128
        lstm_hidden_size = 256
        lstm_num_layers = 2
        lstm_dropout = 0.2

        model = BayesianRuntimeNet(
            input_size=input_size,
            embedding_size=embedding_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
        )

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

        input_tensor = torch.randn(batch_size, seq_len, input_size)
        output_mean, output_std = model(input_tensor, x_len, pinch_points)
        assert output_mean.shape == (batch_size, 1)
        assert output_std.shape == (batch_size, 1)

    # def test_training(self):  # pylint: disable=too-many-locals
    #     """
    #     Test the training of the BayesianRuntimeNet model.
    #     """

    #     torch.autograd.set_detect_anomaly(True)

    #     input_size = 53
    #     embedding_size = 256
    #     lstm_hidden_size = 512
    #     lstm_num_layers = 3
    #     lstm_dropout = 0.3

    #     model = BayesianRuntimeNet(
    #         input_size=input_size,
    #         embedding_size=embedding_size,
    #         lstm_hidden_size=lstm_hidden_size,
    #         lstm_num_layers=lstm_num_layers,
    #         lstm_dropout=lstm_dropout,
    #     )

    #     # Set up dummy data
    #     batch_size = 16
    #     num_samples = 100
    #     max_seq_len = 10
    #     output_size = 1

    #     # Random input and target data
    #     x_data = torch.randn(num_samples, max_seq_len, input_size)
    #     pinch_points_data = torch.randint(0, max_seq_len, (num_samples,))
    #     y_data = torch.randn(num_samples, output_size)
    #     query_uuids = list(range(num_samples))

    #     # Create DataLoader
    #     dataset = ConcurrentQueryDataset(x_data, pinch_points_data, y_data, query_uuids)
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=True,
    #         collate_fn=LSTMModel._collate_and_pad,
    #     )

    #     # Define a simple optimizer and loss function
    #     optimizer = optim.Adam(model.parameters(), lr=0.01)

    #     # Training loop for a few epochs
    #     epochs = 3
    #     for _ in range(epochs):
    #         for x_batch, x_len_batch, pinch_points_batch, y_batch, _ in dataloader:
    #             optimizer.zero_grad()
    #             mean, logvar = model(x_batch, x_len_batch, pinch_points_batch)
    #             loss = negative_log_likelihood_loss(mean, y_batch, logvar)
    #             loss.backward()
    #             optimizer.step()
