import pytest

from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.prediction.p_exact import PExact
from autoslo.prediction.prediction import Prediction
from autoslo.strategies_prediction.prediction_strategy import (
    PredictionStrategy,
)


def test_cannot_instantiate_abstract_prediction_strategy():
    """
    Cannot instantiate PredictionStrategy abstract base class.
    """
    with pytest.raises(TypeError):
        PredictionStrategy()


def test_predict_returns_prediction_instance():
    """
    Predict must return a Prediction (here DummyPrediction) instance.
    """

    class DummyPrediction(Prediction):
        def has_lower_predicted_slo_violation_rate(self, other) -> bool:
            return False

        def has_lower_predicted_cost(self, other) -> bool:
            return False

        def _has_predicted_slo_violation_rate_under(
            self, threshold: float
        ) -> bool:
            return False

    class DummyStrategy(PredictionStrategy):
        def predict(self, blueprint, latency_slo_s, *args, **kwargs):
            return DummyPrediction()

    bp = Blueprint([Cluster(rpu=4)])
    d = DummyStrategy()
    pred = d.predict(bp, 0.5)
    assert isinstance(pred, Prediction)
    assert isinstance(pred, DummyPrediction)


def test_predict_receives_args_and_kwargs():
    """
    predict() should receive blueprint, latency and any extra args/kwargs.
    """

    class Recorder(PredictionStrategy):
        def __init__(self):
            self.called = None

        def predict(self, blueprint, latency_slo_s, *args, **kwargs):
            self.called = (blueprint, latency_slo_s, args, kwargs)
            return PExact(slo_violation_rate=0.0, cost=0.0)

    bp = Blueprint([Cluster(rpu=8)])
    r = Recorder()
    res = r.predict(bp, 1.0, 42, "x", flag=True)
    assert isinstance(res, PExact)
    assert r.called is not None
    assert r.called[0] is bp
    assert r.called[1] == 1.0
    assert r.called[2] == (42, "x")
    assert r.called[3] == {"flag": True}
