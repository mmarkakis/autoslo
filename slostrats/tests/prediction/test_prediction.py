import pytest

from slostrats.prediction.prediction import Prediction


def test_cannot_instantiate_abstract_prediction():
    """
    Cannot instantiate Prediction abstract base class directly.
    """
    with pytest.raises(TypeError):
        Prediction()


def test_validate_threshold_accepts_bounds():
    """
    Thresholds 0.0 and 1.0 are accepted after validation.
    """

    class Dummy(Prediction):
        def has_lower_predicted_slo_violation_rate(self, other) -> bool:
            return False

        def has_lower_predicted_cost(self, other) -> bool:
            return False

        def _has_predicted_slo_violation_rate_under(
            self, threshold: float
        ) -> bool:
            return True

    d = Dummy()
    assert isinstance(d.has_predicted_slo_violation_rate_under(0.0), bool)
    assert isinstance(d.has_predicted_slo_violation_rate_under(1.0), bool)


def test_validate_threshold_rejects_out_of_range():
    """
    Thresholds outside [0,1] raise ValueError.
    """

    class Dummy(Prediction):
        def has_lower_predicted_slo_violation_rate(self, other) -> bool:
            return False

        def has_lower_predicted_cost(self, other) -> bool:
            return False

        def _has_predicted_slo_violation_rate_under(
            self, threshold: float
        ) -> bool:
            return False

    d = Dummy()
    with pytest.raises(ValueError):
        d.has_predicted_slo_violation_rate_under(-0.01)
    with pytest.raises(ValueError):
        d.has_predicted_slo_violation_rate_under(1.01)


def test_has_predicted_slo_violation_rate_under_delegates():
    """
    Method delegates to subclass implementation with the threshold arg.
    """

    class Dummy(Prediction):
        def __init__(self):
            self.called_with = None

        def has_lower_predicted_slo_violation_rate(self, other) -> bool:
            return False

        def has_lower_predicted_cost(self, other) -> bool:
            return False

        def _has_predicted_slo_violation_rate_under(
            self, threshold: float
        ) -> bool:
            self.called_with = threshold
            return threshold < 0.5

    d = Dummy()
    assert d.has_predicted_slo_violation_rate_under(0.4) is True
    assert d.called_with == 0.4
    assert d.has_predicted_slo_violation_rate_under(0.6) is False
    assert d.called_with == 0.6
