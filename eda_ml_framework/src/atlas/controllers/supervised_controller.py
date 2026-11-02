"""Explicit workflow adapter; scientific calculations remain in models."""

from typing import Any

from atlas.models import SupervisedLearning

from .eda_controller import EDAController


class SupervisedController(EDAController):
    model_type = SupervisedLearning

    def identify_problem(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.identify_problem(*args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.train(*args, **kwargs)

    def create_estimator(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.create_estimator(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.evaluate(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(*args, **kwargs)

    def feature_importance(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.feature_importance(*args, **kwargs)

    def coefficients(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.coefficients(*args, **kwargs)

    def learning_curve(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.learning_curve(*args, **kwargs)

    def compare(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.compare(*args, **kwargs)
