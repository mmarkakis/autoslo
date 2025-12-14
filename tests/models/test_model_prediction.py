import math

import pytest
from pytest import approx
from scipy.stats import norm

from autoslo.models.model_prediction import ModelPrediction


@pytest.mark.unit
def test_constant_latency_properties():
    """Test metrics for constant latency prediction."""
    prediction = ModelPrediction(mean_s=[5.0])

    assert prediction.overall_mean_s() == 5.0
    assert prediction.overall_std_dev_s() == 0.0
    assert prediction.overall_likelihood(5.0) == 1.0
    assert prediction.overall_likelihood(4.0) == 0.0
    assert prediction.latency_at_percentile(50.0) == 5.0
    assert prediction.percentile_at_latency(5.0) == 1.0
    assert prediction.percentile_at_latency(4.0) == 0.0
    assert prediction.q_error_at_mean(5.0) == 1.0


@pytest.mark.unit
def test_normal_latency_properties():
    """Test Gaussian latency metrics."""
    prediction = ModelPrediction(mean_s=[10.0], std_dev_s=[2.0])

    assert prediction.overall_mean_s() == approx(10.0)
    assert prediction.overall_std_dev_s() == approx(2.0)
    assert prediction.latency_at_percentile(50.0) == approx(10.0)
    assert prediction.percentile_at_latency(10.0) == approx(0.5)
    assert prediction.q_error_at_mean(8.0) == approx(1.25)


@pytest.mark.unit
def test_constant_mixture_metrics():
    """Test discrete mixture metrics."""
    prediction = ModelPrediction(
        mean_s=[1.0, 3.0],
        mix_coeffs=[0.25, 0.75],
    )

    assert prediction.overall_mean_s() == approx(2.5)
    assert prediction.overall_std_dev_s() == approx(0.8660254038)
    assert prediction.latency_at_percentile(10.0) == 1.0
    assert prediction.latency_at_percentile(90.0) == 3.0
    assert prediction.percentile_at_latency(1.0) == 0.25
    assert prediction.percentile_at_latency(3.0) == 1.0


@pytest.mark.unit
def test_normal_mixture_percentile_inverse():
    """Test percentile inversion for Gaussian mixture."""
    means = [5.0, 15.0]
    stds = [1.0, 2.0]
    weights = [0.4, 0.6]
    prediction = ModelPrediction(
        mean_s=means,
        std_dev_s=stds,
        mix_coeffs=weights,
    )

    expected_mean = sum(m * w for m, w in zip(means, weights))
    expected_variance = (
        sum(
            (s ** 2 + m ** 2) * w
            for m, s, w in zip(means, stds, weights)
        )
        - expected_mean ** 2
    )
    expected_std = math.sqrt(expected_variance)
    expected_likelihood = sum(
        w * norm.pdf(12.0, loc=m, scale=s)
        for m, s, w in zip(means, stds, weights)
    )

    assert prediction.overall_mean_s() == approx(expected_mean)
    assert prediction.overall_std_dev_s() == approx(expected_std)
    assert prediction.overall_likelihood(12.0) == approx(expected_likelihood)
    latency = prediction.latency_at_percentile(60.0)
    assert prediction.percentile_at_latency(latency) == approx(0.6, abs=1e-3)
