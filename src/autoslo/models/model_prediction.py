from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from autoslo.models.residual_calibrator import ResidualCalibrator

import numpy as np
from scipy.optimize import root_scalar  # type: ignore[import]
from scipy.stats import norm  # type: ignore[import]


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
    # Shared reference to the owning IconqModel's calibrator.
    calibrator: Optional["ResidualCalibrator"] = field(
        default=None, compare=False, repr=False, hash=False
    )

    def _c_constant(self) -> bool:
        """
        True if the model represents a single constant latency, False otherwise.
        """
        return (
            (self.mean_s is not None)
            and (len(self.mean_s) == 1)
            and (self.std_dev_s is None)
            and (self.mix_coeffs is None)
        )

    def _c_norm(self) -> bool:
        """
        True if the model represents a single Gaussian, False otherwise.
        """
        return (
            (self.mean_s is not None)
            and (len(self.mean_s) == 1)
            and (self.std_dev_s is not None)
            and (len(self.std_dev_s) == 1)
            and (self.mix_coeffs is None)
        )

    def _c_constant_mix(self) -> bool:
        """
        True if the model represents a mixture of discrete latencies, False otherwise.
        """
        return (
            (self.mean_s is not None)
            and (len(self.mean_s) > 1)
            and (self.std_dev_s is None)
            and (self.mix_coeffs is not None)
            and (len(self.mean_s) == len(self.mix_coeffs))
        )

    def _c_norm_mix(self) -> bool:
        """
        True if the model represents a mixture of Gaussians, False otherwise.
        """
        return (
            (self.mean_s is not None)
            and (len(self.mean_s) > 1)
            and (self.std_dev_s is not None)
            and (len(self.std_dev_s) == len(self.mean_s))
            and (self.mix_coeffs is not None)
            and (len(self.mean_s) == len(self.mix_coeffs))
        )

    def overall_mean_s(self, percentile: Optional[float] = None) -> float:
        """
        Based on the contents of the model prediction, builds a mixture of Gaussians and computes
        the mean latency.

        When ``percentile`` is provided and the prediction carries a residual
        calibrator (set by IconqModel.predict_from_dataset), a post-hoc
        multiplicative correction is applied in log-space.  Calibration is
        restricted to point-estimate predictions (``_c_constant()``) that
        originate from the LSTM (``metadata['model_source'] == 'lstm'``).

        Parameters:
            percentile: If given, apply residual calibration at this quantile
                level (0–1).  ``None`` returns the raw prediction.

        Returns:
            The (optionally calibrated) mean latency in seconds.

        Raises:
            ValueError: If the model prediction is not well-formed.
        """

        if self._c_constant():
            raw = max(self.mean_s[0], self.MINIMUM_OUTPUT_LATENCY_S)  # type: ignore
            if (
                percentile is not None
                and self.calibrator is not None
                and self.metadata is not None
                and self.metadata.get("model_source") == "lstm"
            ):
                rpu = int(self.metadata["rpu"])
                concurrency = int(
                    self.metadata.get("num_other_concurrent_queries", 0)
                )
                return self.calibrator.correct_scalar(
                    raw, rpu, concurrency, percentile
                )

            return raw
        if self._c_norm():
            return max(self.mean_s[0], self.MINIMUM_OUTPUT_LATENCY_S)  # type: ignore

        if self._c_constant_mix() or self._c_norm_mix():
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

        if self._c_constant():
            return 0.0

        if self._c_norm():
            return self.std_dev_s[0]  # type: ignore

        if self._c_constant_mix():
            overall_mean = self.overall_mean_s()
            unnormalized_variance = sum(
                ((mean - overall_mean) ** 2) * mix_coeff  # type: ignore
                for mean, mix_coeff in zip(self.mean_s, self.mix_coeffs)  # type: ignore
            )
            variance = unnormalized_variance / sum(self.mix_coeffs)  # type: ignore
            return np.sqrt(variance)

        if self._c_norm_mix():
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
        if self._c_constant():
            return 1.0 if self.mean_s[0] == latency else 0.0  # type: ignore

        # Case 2: Single Gaussian.
        if self._c_norm():
            return norm.pdf(latency, loc=self.mean_s[0], scale=self.std_dev_s[0])  # type: ignore

        # Case 3: Mix of constant latencies.
        if self._c_constant_mix():
            for mean, mix_coeff in zip(self.mean_s, self.mix_coeffs):  # type: ignore
                if mean == latency:
                    return mix_coeff / sum(self.mix_coeffs)  # type: ignore
            return 0.0

        # Case 4: Multiple means, multiple standard deviations, correct mixing coefficients.
        if self._c_norm_mix():
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
        if self._c_constant():
            return self.mean_s[0]  # type: ignore

        # Case 2: Single Gaussian.
        if self._c_norm():
            return norm.ppf(
                q=(percentile / 100), loc=self.mean_s[0], scale=self.std_dev_s[0]  # type: ignore
            )

        # Case 3: Mix of constant latencies.
        if self._c_constant_mix():
            sort_order = np.argsort(self.mean_s)  # type: ignore
            sorted_means = [self.mean_s[i] for i in sort_order]  # type: ignore
            sorted_mix_coeffs = [self.mix_coeffs[i] for i in sort_order]  # type: ignore
            cumulative_mix_coeffs = np.cumsum(sorted_mix_coeffs)
            for mean, cum_mix_coeff in zip(sorted_means, cumulative_mix_coeffs):
                if cum_mix_coeff >= (percentile / 100):
                    return mean

        # Case 4: Mixture of distributions with standard deviations.
        if self._c_norm_mix():
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
        if self._c_constant():
            return 0.0 if latency < self.mean_s[0] else 1.0  # type: ignore

        # Case 2: Single Gaussian.
        if self._c_norm():
            return norm.cdf(latency, loc=self.mean_s[0], scale=self.std_dev_s[0])  # type: ignore

        # Case 3: Mix of constant latencies.
        if self._c_constant_mix():
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
        if self._c_norm_mix():
            running_cumulative = 0.0
            for mean, std_dev, mix_coeff in zip(self.mean_s, self.std_dev_s, self.mix_coeffs):  # type: ignore
                running_cumulative += mix_coeff * norm.cdf(
                    latency, loc=mean, scale=std_dev
                )
            return running_cumulative

        # If we got here, the prediction is not well-formed.
        raise ValueError("Model prediction is not well-formed.")
