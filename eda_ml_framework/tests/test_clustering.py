"""Algoritmos y manejo de ruido y grupos incompatibles."""

import numpy as np
import pytest

from framework import Clustering
from framework.exceptions import FrameworkError
from framework.metrics import clustering_metrics


@pytest.mark.parametrize("algorithm", [a for a in Clustering.algoritmos() if a != "kmedoids"])
def test_clustering(blobs, algorithm):
    engine = Clustering(blobs)
    result = engine.entrenar(algorithm, list(blobs))
    assert len(result.predictions) == len(blobs)
    assert "ruido" in result.metrics
    assert not engine.caracterizar(result).empty
    if algorithm in {"kmeans", "minibatch"}:
        assert result.metrics["Silhouette"] > 0.4
        assert engine.silhouette_plot(result) is not None


def test_noise_metrics():
    x = np.arange(20).reshape(10, 2)
    for labels in [np.full(10, -1), np.zeros(10), np.arange(10)]:
        assert "Silhouette" not in clustering_metrics(x, labels)


def test_representations(blobs):
    engine = Clustering(blobs)
    for method in ["pca", "tsne"]:
        result = engine.reducir(list(blobs), method)
        assert result["datos"].shape == (70, 2)
        assert engine.visualizar_representacion(result["datos"]) is not None
    assert len(engine.comparar_representaciones()) == 2
    assert engine.distancias(list(blobs), "cityblock").shape == (70, 70)
    assert len(engine.codo(list(blobs), 4)) == 4
    assert engine.dendrograma(list(blobs)) is not None
    with pytest.raises(FrameworkError):
        engine.entrenar("kmeans", list(blobs), n_clusters=100)
