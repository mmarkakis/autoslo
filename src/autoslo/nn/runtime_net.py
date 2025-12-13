from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from autoslo.nn.bayesian_linear import BayesianLinear
from autoslo.nn.bayesian_pinch_lstm import BayesianPinchLSTM
import logging

logger = logging.getLogger(__name__)


def xavier_init(m: nn.Module):
    """
    Apply Xavier initialization to the weights of a module.
    """

    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            nn.init.constant_(param.data, 0.0)


class RuntimeNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    A model sandwich with linear layers on the input and output sides and an
    LSTM in the middle.

    This model can be configured in three ways:
    - As a deterministic model, with regular linear layers and LSTM. This model
        predicts mean latency.
    - As a Bayesian model, with Bayesian linear layers and a Bayesian LSTM. This
        model predicts both the mean and the log variance of the latency.
    - As a Mixture Density Network (MDN), with regular linear layers and LSTM,
        and a final layer that outputs the parameters of a Gaussian mixture
        model. This model predicts the parameters of the Gaussian mixture model.
        The number of Gaussian components is specified by the
        `mdn_num_gaussians` parameter. For each component, the model outputs the
        mixing coefficient,mean, and standard deviation.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_size: float = 50,
        embedding_size: float = 128,
        lstm_hidden_size: float = 256,
        lstm_num_layers: float = 2,
        lstm_dropout: float = 0.2,
        is_bayesian: bool = True,
        bayesian_samples: int = 5,
        is_mdn: bool = False,
        mdn_num_gaussians: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the (possibly) Bayesian model "sandwich".

        Parameters:
            input_size: The number of input features.
            embedding_size: The number of embedding layer output features.
            lstm_hidden_size: The number of hidden features in the LSTM.
            lstm_num_layers: The number of layers in the LSTM.
            lstm_dropout: The dropout probability in the LSTM.
            is_bayesian: Whether to use Bayesian linear layers.
            bayesian_samples: The number of samples to draw from the model at
                each inference, to account for model uncertainty.
            is_mdn: Whether to use the model as a Mixture Density Network (MDN).
            mdn_num_gaussians: The number of MDN Gaussian mixture components.
            device: The device to use for the model.
        """
        super().__init__()
        self._input_size = int(input_size)
        self._embedding_size = int(embedding_size)
        self._lstm_hidden_size = int(lstm_hidden_size)
        self._lstm_num_layers = int(lstm_num_layers)
        self._lstm_dropout = lstm_dropout
        self._is_bayesian = is_bayesian
        self._bayesian_samples = bayesian_samples
        self._is_mdn = is_mdn
        self._mdn_num_gaussians = mdn_num_gaussians

        assert not (
            self._is_bayesian and self._is_mdn
        ), "The model cannot be both Bayesian and an MDN."

        self._bn = nn.BatchNorm1d(self._input_size).to(device)

        # Input model: two linear layers
        self._in_model = nn.Sequential(
            nn.Linear(
                self._input_size,
                self._embedding_size,
                device=device,
            ),
            nn.Linear(
                self._embedding_size, self._embedding_size, device=device
            ),
        ).to(device)

        # Middle model: Bayesian LSTM
        self._mid_model = BayesianPinchLSTM(
            input_size=self._embedding_size,
            hidden_size=self._lstm_hidden_size,
            num_layers_forward=self._lstm_num_layers,
            num_layers_reverse=self._lstm_num_layers,
            dropout=self._lstm_dropout,
            is_bayesian=False,  # self._is_bayesian,
            device=device,
        ).to(device)
        xavier_init(self._mid_model)

        # Output model(s):
        #
        # - If the model is not an MDN, we have two single-output models with 
        #   two linear layers each, which output the mean and log variance of 
        #   the runtime, respectively.
        # - If the model is a MDN, we have three, M-output models with two 
        #   linear layers each, where M is the number of MDN Gaussian mixture 
        #   components. The models output the means, log variances, and mixing 
        #   coefficients of the Gaussian components, respectively.
        linear_layer = BayesianLinear if self._is_bayesian else nn.Linear
        mean_dims = 1 if not self._is_mdn else self._mdn_num_gaussians
        logvar_dims = 1 if not self._is_mdn else self._mdn_num_gaussians
        self._out_model_mean = nn.Sequential(
            linear_layer(
                self._lstm_hidden_size,
                self._lstm_hidden_size // 2,
                device=device,
            ),
            linear_layer(self._lstm_hidden_size // 2, mean_dims, device=device),
        ).to(device)
        self._out_model_logvar = nn.Sequential(
            linear_layer(
                self._lstm_hidden_size,
                self._lstm_hidden_size // 2,
                device=device,
            ),
            linear_layer(
                self._lstm_hidden_size // 2, logvar_dims, device=device
            ),
        ).to(device)
        self._out_model_mix = (
            None
            if (not self._is_mdn)
            else (
                nn.Sequential(
                    linear_layer(
                        self._lstm_hidden_size,
                        self._lstm_hidden_size // 2,
                        device=device,
                    ),
                    linear_layer(
                        self._lstm_hidden_size // 2,
                        self._mdn_num_gaussians,
                        device=device,
                    ),
                ).to(device)
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        pinch_points: torch.Tensor,
        mdn_mix_softmax_temperature: float = 1.0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform the forward pass.

        Parameters:
            x: The input tensor, of shape (batch_size, max_seq_len, input_size).
            x_len: The actual lengths of the sequences in the batch, even though 
                they are 0-padded to max_seq_len in `x`.
            pinch_points: The indices of the pinch points in each of the 
                sequences in the batch. This tensor has shape (batch_size,).
            mdn_mix_softmax_temperature: The temperature parameter for the 
                softmax function used to compute the mixing coefficients in the 
                MDN. Only used if the model is an MDN.

        Returns:
            output_mean: The output tensor of mean values, of shape 
                (batch_size, N). N is 1 if the model is not an MDN, and equal to 
                the number of Gaussian components if it is.
            output_logvar: The output tensor of log variances, of shape 
                (batch_size, N). N is 1 if the model is not an MDN, and equal 
                to the number of Gaussian components if it is.
            output_mix: The output tensor of Gaussian mixture parameters, of 
                shape (batch_size, num_gaussians). Only returned if the model is 
                an MDN.
        """
        # Apply batch normalization. BatchNorm1d expects an input shape of
        # (batch_size, num_features, seq_len), so we permute the input tensor.
        logger.info(f"Input shape: {x.shape}")

        x = x.permute(0, 2, 1)
        x_norm = self._bn(x)
        x_norm = x_norm.permute(0, 2, 1)

        if not self._is_mdn:
            return self._forward_impl_bayesian(x_norm, x_len, pinch_points) + (
                None,
            )
        else:  # self._is_mdn
            return self._forward_impl_mdn(
                x_norm, x_len, pinch_points, mdn_mix_softmax_temperature
            )

    def _forward_impl_bayesian(
        self,
        x_norm: torch.Tensor,
        x_len: torch.Tensor,
        pinch_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        An implementation of the forward pass when the model is not an MDN.

        Parameters:
            x_norm: The input tensor, which after batch normalization is of 
                shape (batch_size, max_seq_len, input_size).
            x_len: The actual lengths of the sequences in the batch, even though 
                they are 0-padded to max_seq_len in `x`.
            pinch_points: The indices of the pinch points in each of the 
                sequences in the batch. This tensor has shape (batch_size,).

        Returns:
            output_mean: The output tensor of mean values, of shape 
                (batch_size, 1).
            output_logvar: The output tensor of standard deviations, of shape 
                (batch_size, 1).
        """

        mean_predictions = []
        variance_predictions = []

        # Pass through the layers
        in_model_output = self._in_model(
            x_norm
        )  # Output shape: (batch_size, seq_len, embedding_size)
        mid_model_output = self._mid_model(
            in_model_output, x_len, pinch_points
        )  # Output shape: (batch_size, lstm_hidden_size)

        for _ in range(self._bayesian_samples):
            out_model_mean_output = self._out_model_mean(
                mid_model_output
            )  # Output shape: (batch_size, 1)
            out_model_logvar_output = self._out_model_logvar(
                mid_model_output
            )  # Output shape: (batch_size, 1)

            if not self._is_bayesian:
                # If the model is not Bayesian, we only need to sample once, 
                # since the predictions will be deterministic.
                return out_model_mean_output, out_model_logvar_output

            mean_predictions.append(out_model_mean_output)
            variance_predictions.append(torch.exp(out_model_logvar_output))

        # Stack lists of samples to tensors
        mean_predictions_tensor = torch.stack(
            mean_predictions, dim=0
        )  # Shape: (num_samples, batch_size, 1)
        variance_predictions_tensor = torch.stack(
            variance_predictions, dim=0
        )  # Shape: (num_samples, batch_size, 1)

        # Compute the mean and variance of the predictions
        out_model_mean_output = mean_predictions_tensor.mean(
            dim=0
        )  # Shape: (batch_size, 1)
        squared_diffs = (
            mean_predictions_tensor - out_model_mean_output
        ) ** 2  # Shape: (num_samples, batch_size, 1)
        variance_from_mean = squared_diffs.mean(dim=0)  # Shape: (batch_size, 1)
        predictive_variance = (
            variance_predictions_tensor.mean(dim=0) + variance_from_mean
        )  # Shape: (batch_size, 1)
        out_model_logvar_output = torch.log(
            predictive_variance + 1e-6
        )  # For numerical stability

        return out_model_mean_output, out_model_logvar_output

    def _forward_impl_mdn(
        self,
        x_norm: torch.Tensor,
        x_len: torch.Tensor,
        pinch_points: torch.Tensor,
        mdn_mix_softmax_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        An implementation of the forward pass when the model is an MDN.

        Parameters:
            x_norm: The input tensor, which after batch normalization is of 
                shape (batch_size, max_seq_len, input_size).
            x_len: The actual lengths of the sequences in the batch, even though 
                they are 0-padded to max_seq_len in `x`.
            pinch_points: The indices of the pinch points in each of the 
                sequences in the batch. This tensor has shape (batch_size,).
            mdn_mix_softmax_temperature: The temperature parameter for the 
                softmax function used to compute the mixing coefficients in the 
                MDN.

        Returns:
            output_mean: The output tensor of mean values, of shape 
                (batch_size, num_gaussians).
            output_logvar: The output tensor of standard deviations, of shape 
                (batch_size, num_gaussians).
            output_mix: The output tensor of mixing coefficients, of shape 
                (batch_size, num_gaussians).
        """

        if self._out_model_mix is None:
            raise ValueError("The model must be an MDN to use this method.")

        # Pass through the layers
        in_model_output = self._in_model(
            x_norm
        )  # Output shape: (batch_size, seq_len, embedding_size)
        mid_model_output = self._mid_model(
            in_model_output, x_len, pinch_points
        )  # Output shape: (batch_size, lstm_hidden_size)
        out_model_mean_output = F.softplus(
            self._out_model_mean(mid_model_output), beta=100
        )  # Output shape: (batch_size, num_gaussians), use softplus to ensure
        # mean is positive
        out_model_logvar_output = self._out_model_logvar(
            mid_model_output
        )  # Output shape: (batch_size, num_gaussians).
        out_model_mix_output = F.softmax(
            self._out_model_mix(mid_model_output) / mdn_mix_softmax_temperature,
            dim=1,
        )  # Output shape: (batch_size, num_gaussians), use softmax to
        # ensure mixing coefficients sum to 1

        return (
            out_model_mean_output,
            out_model_logvar_output,
            out_model_mix_output,
        )
