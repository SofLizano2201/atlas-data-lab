"""Explicit workflow adapter; scientific calculations remain in models."""

from typing import Any

from atlas.models import Regression

from .supervised_controller import SupervisedController


class RegressionController(SupervisedController):
    model_type = Regression

    @classmethod
    def algorithms(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.model_type.algorithms(*args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.train(*args, **kwargs)

    def create_estimator(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.create_estimator(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.evaluate(*args, **kwargs)

    def plot_result(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.plot_result(*args, **kwargs)

    def compare_lasso(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.compare_lasso(*args, **kwargs)

    def compare_ridge(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.compare_ridge(*args, **kwargs)

    def compare_svm(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.compare_svm(*args, **kwargs)
