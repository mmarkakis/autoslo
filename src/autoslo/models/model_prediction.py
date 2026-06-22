from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
from scipy.optimize import root_scalar  # type: ignore[import]
from scipy.stats import norm  # type: ignore[import]


class ModelPredictionKind(Enum):
    """Discriminant for ModelPrediction computed once at construction time."""

    CONSTANT = auto()  # len(mean)==1, no std, no mix
    NORM = auto()  # len(mean)==1, std present, no mix
    CONSTANT_MIX = auto()  # len(mean)>1, no std, mix present
    NORM_MIX = auto()  # len(mean)>1, std and mix present
    INVALID = auto()  # anything else


@dataclass(frozen=True)
class ModelPrediction:
    """
    A prediction for the latency of a query, given the concurrent state it encounters. We assume
    that the latency is a mixture of Gaussians, and we predict the mean and standard deviation of
    each Gaussian, as well as the mixing coefficients.
    """

    MINIMUM_OUTPUT_LATENCY_S = (
        0.001  # Minimum output latency in seconds to avoid zero or negative
    )

    mean_s: Optional[list[float]] = None
    std_dev_s: Optional[list[float]] = None
    mix_coeffs: Optional[list[float]] = None
    metadata: Optional[dict[str, Any]] = None
    # Computed once at construction; excluded from __eq__, __hash__, __repr__.
    _kind: ModelPredictionKind = field(
        default=ModelPredictionKind.INVALID,
        init=False,
        compare=False,
        repr=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        # frozen=True prevents normal attribute assignment, so we use
        # object.__setattr__ to set the derived field.
        if self.mean_s is None:
            return  # _kind stays INVALID
        n = len(self.mean_s)
        msg = "Could not determine the kind of the model prediction."
        if n == 1:
            if self.std_dev_s is None and self.mix_coeffs is None:
                object.__setattr__(self, "_kind", ModelPredictionKind.CONSTANT)
            elif (
                self.std_dev_s is not None
                and len(self.std_dev_s) == 1
                and self.mix_coeffs is None
            ):
                object.__setattr__(self, "_kind", ModelPredictionKind.NORM)
            else:
                raise ValueError(msg)
        elif n > 1:
            if (
                self.std_dev_s is None
                and self.mix_coeffs is not None
                and len(self.mix_coeffs) == n
            ):
                object.__setattr__(
                    self, "_kind", ModelPredictionKind.CONSTANT_MIX
                )
            elif (
                self.std_dev_s is not None
                and len(self.std_dev_s) == n
                and self.mix_coeffs is not None
                and len(self.mix_coeffs) == n
            ):
                object.__setattr__(self, "_kind", ModelPredictionKind.NORM_MIX)
            else:
                raise ValueError(msg)
        else:
            raise ValueError(msg)

    def overall_mean_s(self) -> float:
        """
        Based on the contents of the model prediction, builds a mixture of Gaussians and computes
        the mean latency.

        Returns:
            The mean latency.

        Raises:
            ValueError: If the model prediction is not well-formed.
        """
        if (
            self._kind is ModelPredictionKind.CONSTANT
            or self._kind is ModelPredictionKind.NORM
        ):
            return max(self.mean_s[0], self.MINIMUM_OUTPUT_LATENCY_S)  # type: ignore

        if (
            self._kind is ModelPredictionKind.CONSTANT_MIX
            or self._kind is ModelPredictionKind.NORM_MIX
        ):
            unnormalized_mean = sum(
                mu * w for mu, w in zip(self.mean_s, self.mix_coeffs)  # type: ignore
            )
            return max(unnormalized_mean / sum(self.mix_coeffs), self.MINIMUM_OUTPUT_LATENCY_S)  # type: ignore

        raise ValueError("Model prediction is not well-formed.")

    def overall_std_dev_s(self) -> float:
        """
        Based on the contents of the model prediction, builds a mixture of Gaussians and computes
        the standard deviation of the latency.

        Returns:
            The standard deviation of the latency.

        Raises:
            ValueError: If the model prediction is not well-formed.
        """

        if self._kind is ModelPredictionKind.CONSTANT:
            return 0.0

        if self._kind is ModelPredictionKind.NORM:
            return self.std_dev_s[0]  # type: ignore

        if self._kind is ModelPredictionKind.CONSTANT_MIX:
            overall_mean = self.overall_mean_s()
            unnormalized_variance = sum(
                ((mean - overall_mean) ** 2) * mix_coeff  # type: ignore
                for mean, mix_coeff in zip(self.mean_s, self.mix_coeffs)  # type: ignore
            )
            variance = unnormalized_variance / sum(self.mix_coeffs)  # type: ignore
            return np.sqrt(variance)

        if self._kind is ModelPredictionKind.NORM_MIX:
            overall_mean = self.overall_mean_s()
            variance = (
                sum(
                    (std_dev**2 + mean**2) * mix_coeff  # type: ignore
                    for std_dev, mean, mix_coeff in zip(
                        self.std_dev_s, self.mean_s, self.mix_coeffs  # type: ignore
                    )
                )
                / sum(self.mix_coeffs)  # type: ignore
            ) - overall_mean**2
            return np.sqrt(variance)

        raise ValueError("Model prediction is not well-formed.")

    def overall_likelihood(self, latency: float) -> float:
        """
        Based on the contents of the model prediction, builds a mixture of Gaussians and computes
        the likelihood of the given latency.

        Parameters:
            latency: The latency for which to compute the likelihood.

        Returns:
            The likelihood of the given latency.

        Raises:
            ValueError: If the model prediction is not well-formed.
        """

        # Case 1: Single constant latency.
        if self._kind is ModelPredictionKind.CONSTANT:
            return 1.0 if self.mean_s[0] == latency else 0.0  # type: ignore

        # Case 2: Single Gaussian.
        if self._kind is ModelPredictionKind.NORM:
            return norm.pdf(latency, loc=self.mean_s[0], scale=self.std_dev_s[0])  # type: ignore

        # Case 3: Mix of constant latencies.
        if self._kind is ModelPredictionKind.CONSTANT_MIX:
            for mean, mix_coeff in zip(self.mean_s, self.mix_coeffs):  # type: ignore
                if mean == latency:
                    return mix_coeff / sum(self.mix_coeffs)  # type: ignore
            return 0.0

        # Case 4: Multiple means, multiple standard deviations, correct mixing coefficients.
        if self._kind is ModelPredictionKind.NORM_MIX:
            unnomralized_likelihood = sum(
                mix_coeff * norm.pdf(latency, loc=mean, scale=std_dev)
                for mean, std_dev, mix_coeff in zip(
                    self.mean_s, self.std_dev_s, self.mix_coeffs  # type: ignore
                )
            )
            return unnomralized_likelihood / sum(self.mix_coeffs)  # type: ignore

        # If we got here, the prediction is not well-formed.
        raise ValueError("Model prediction is not well-formed.")

    def q_error_at_mean(
        self, latency: float, min_allowed_predicted_mean: float = 0.01
    ) -> float:
        """
        Based on the contents of the model prediction, builds a mixture of Gaussians and computes
        the q-error of the given latency with respect to the mean.

        Parameters:
            latency: The latency for which to compute the q-error.
            min_allowed_predicted_mean: The minimum allowed predicted mean latency. If the predicted
                mean is below this value, it is replaced with this value.

        Returns:
            The q-error of the given latency.

        Raises:
            ValueError: If the model prediction is not well-formed.
        """

        predicted_mean = max(self.overall_mean_s(), min_allowed_predicted_mean)
        q_error = max(latency / predicted_mean, predicted_mean / latency)
        return q_error

    def latency_at_percentile(self, percentile: float) -> float:
        """
        Based on the contents of the model prediction, builds a mixture of Gaussians and computes
        the latency at the given percentile of the mixture.

        Parameters:
            percentile: The percentile at which to compute the latency, between 0 and 100.

        Returns:
            The latency at the given percentile.

        Raises:
            ValueError: If the model prediction is not well-formed, or if the percentile is out of
                bounds.
        """

        # Correctness checking.
        if (percentile < 0) or (percentile > 100):
            raise ValueError(f"Percentile {percentile} is out of bounds.")

        # Case 1: Single constant latency.
        if self._kind is ModelPredictionKind.CONSTANT:
            return self.mean_s[0]  # type: ignore

        # Case 2: Single Gaussian.
        if self._kind is ModelPredictionKind.NORM:
            return norm.ppf(
                q=(percentile / 100), loc=self.mean_s[0], scale=self.std_dev_s[0]  # type: ignore
            )

        # Case 3: Mix of constant latencies.
        if self._kind is ModelPredictionKind.CONSTANT_MIX:
            sort_order = np.argsort(self.mean_s)  # type: ignore
            sorted_means = [self.mean_s[i] for i in sort_order]  # type: ignore
            sorted_mix_coeffs = [self.mix_coeffs[i] for i in sort_order]  # type: ignore
            cumulative_mix_coeffs = np.cumsum(sorted_mix_coeffs)
            for mean, cum_mix_coeff in zip(sorted_means, cumulative_mix_coeffs):
                if cum_mix_coeff >= (percentile / 100):
                    return mean

        # Case 4: Mixture of distributions with standard deviations.
        if self._kind is ModelPredictionKind.NORM_MIX:
            func = lambda x: np.sum(
                self.mix_coeffs
                * norm.cdf(x, loc=self.mean_s, scale=self.std_dev_s)
            ) - (percentile / 100)
            x_min = min(
                [mu - 3 * sigma for mu, sigma in zip(self.mean_s, self.std_dev_s)]  # type: ignore
            )
            x_max = max(
                [mu + 3 * sigma for mu, sigma in zip(self.mean_s, self.std_dev_s)]  # type: ignore
            )
            result = root_scalar(func, bracket=[x_min, x_max], method="brentq")
            if not result.converged:
                raise ValueError("Failed to find the percentile.")

            return result.root

        # If we got here, the prediction is not well-formed.
        raise ValueError("Model prediction is not well-formed.")

    def percentile_at_latency(self, latency: float) -> float:
        """
        Based on the contents of the model prediction, builds a mixture of Gaussians and computes
        the percentile at which the given latency occurs. That is, it computes the fraction of the
        probability mass that is below the given latency.

        Parameters:
            latency: The latency for which to compute the percentile.

        Returns:
            The percentile at which the given latency occurs.

        Raises:
            ValueError: If the model prediction is not well-formed.
        """

        # Case 1: Single constant latency.
        if self._kind is ModelPredictionKind.CONSTANT:
            return 0.0 if latency < self.mean_s[0] else 1.0  # type: ignore

        # Case 2: Single Gaussian.
        if self._kind is ModelPredictionKind.NORM:
            return norm.cdf(latency, loc=self.mean_s[0], scale=self.std_dev_s[0])  # type: ignore

        # Case 3: Mix of constant latencies.
        if self._kind is ModelPredictionKind.CONSTANT_MIX:
            sort_order = np.argsort(self.mean_s)  # type: ignore
            sorted_means = [self.mean_s[i] for i in sort_order]  # type: ignore
            sorted_mix_coeffs = [self.mix_coeffs[i] for i in sort_order]  # type: ignore

            running_cumulative = 0.0
            for mean, mix_coeff in zip(sorted_means, sorted_mix_coeffs):
                if latency < mean:
                    return running_cumulative
                running_cumulative += mix_coeff
            return 1.0

        # Case 4: Mixture of distributions with standard deviations.
        if self._kind is ModelPredictionKind.NORM_MIX:
            running_cumulative = 0.0
            for mean, std_dev, mix_coeff in zip(self.mean_s, self.std_dev_s, self.mix_coeffs):  # type: ignore
                running_cumulative += mix_coeff * norm.cdf(
                    latency, loc=mean, scale=std_dev
                )
            return running_cumulative

        # If we got here, the prediction is not well-formed.
        raise ValueError("Model prediction is not well-formed.")
