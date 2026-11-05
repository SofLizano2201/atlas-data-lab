"""Explicit workflow adapter; scientific calculations remain in models."""

from typing import Any

from atlas.models import Clustering

from .unsupervised_controller import UnsupervisedController


class ClusteringController(UnsupervisedController):
    @staticmethod
    def sample_for_analysis(df, size, random_state=42):
        return Clustering.sample_for_analysis(df, size, random_state)

    def cluster_sizes(self, result):
        return self._model.cluster_sizes(result)

    def profile_heatmap(self, result):
        return self._model.profile_heatmap(result)

    def inspect_cluster(self, result, label):
        return self._model.inspect_cluster(result, label)

    model_type = Clustering

    @classmethod
    def algorithms(cls, *args: Any, **kwargs: Any) -> Any:
        return cls.model_type.algorithms(*args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.train(*args, **kwargs)

    def elbow(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.elbow(*args, **kwargs)

    def silhouette_plot(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.silhouette_plot(*args, **kwargs)

    def plot_elbow(self, table):
        return self._model.plot_elbow(table)

    def dendrogram(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.dendrogram(*args, **kwargs)

    def characterize(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.characterize(*args, **kwargs)

    def representatives(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.representatives(*args, **kwargs)

    def compare(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.compare(*args, **kwargs)
