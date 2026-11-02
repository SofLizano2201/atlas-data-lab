"""Explicit workflow adapter; scientific calculations remain in models."""

from typing import Any

from atlas.models import UnsupervisedLearning

from .eda_controller import EDAController


class UnsupervisedController(EDAController):
    model_type = UnsupervisedLearning

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.prepare(*args, **kwargs)

    def reduce_dimensions(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.reduce_dimensions(*args, **kwargs)

    def distances(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.distances(*args, **kwargs)

    def plot_representation(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.plot_representation(*args, **kwargs)

    def compare_representations(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.compare_representations(*args, **kwargs)
