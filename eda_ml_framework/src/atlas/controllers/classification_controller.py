"""Explicit workflow adapter; scientific calculations remain in models."""

from typing import Any

from atlas.models import Classification

from .supervised_controller import SupervisedController


class ClassificationController(SupervisedController):
    model_type = Classification

    @classmethod
    def algorithms(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.model_type.algorithms(*args, **kwargs)

    def create_estimator(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.create_estimator(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.evaluate(*args, **kwargs)

    def plot_result(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.plot_result(*args, **kwargs)
