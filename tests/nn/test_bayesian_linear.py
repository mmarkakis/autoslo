import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from autoslo.nn.bayesian_linear import BayesianLinear

# pylint: disable=protected-access


# FIXME: These have not been updated after the codebase migration.


class TestBayesianLinear:
    """
    Test the BayesianLinear class.
    """

    def test_initialization_shapes(self):
        """
        Test the shapes related to the initialization of the BayesianLinear
        class.
        """

        layer = BayesianLinear(10, 5)
        assert layer._in_features == 10
        assert layer._out_features == 5

    def test_forward_shapes(self):
        """
        Test the shapes related to the forward pass of the BayesianLinear class.
        """

        layer = BayesianLinear(10, 5)
        input_tensor = torch.randn(3, 10)
        output_tensor = layer(input_tensor)
        assert output_tensor.shape == (3, 5)

    def test_training(self):  # pylint: disable=too-many-locals
        """
        Test the training of the BayesianLinear model.
        """

        # Set up dummy data
        batch_size = 16
        in_features = 10
        out_features = 5
        num_samples = 100

        # Random input and target data
        x_data = torch.randn(num_samples, in_features)
        y_data = torch.randn(num_samples, out_features)

        # Create DataLoader
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize the BayesianLinear model
        model = BayesianLinear(
            in_features=in_features, out_features=out_features
        )

        # Define a simple optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Training loop for a few epochs
        epochs = 3
        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
