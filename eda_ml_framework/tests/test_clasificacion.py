"""Clasificadores y curvas con/sin probabilidades."""

import numpy as np
import pandas as pd
import pytest

from framework import Clasificacion
from framework.exceptions import FrameworkError


@pytest.mark.parametrize("algorithm", [a for a in Clasificacion.algoritmos() if a != "xgboost"])
def test_classifiers(iris, algorithm):
    engine = Clasificacion(iris)
    result = engine.entrenar(
        algorithm,
        "target",
        list(iris.columns[:-1]),
        cv=2,
        parametros={"hidden_layer_sizes": (8,), "max_iter": 20} if algorithm == "mlp" else {},
    )
    assert 0 <= result.metrics["Accuracy"] <= 1
    assert result.details["confusion"].shape == (3, 3)
    for kind in ["confusion", "confusion_normalized", "roc", "pr", "correctness"]:
        assert engine.grafico_resultado(result, kind) is not None


def test_binary_no_proba(iris):
    df = iris[iris.target < 2].copy()
    df["target"] = df.target.map({0: "a", 1: "b"})
    engine = Clasificacion(df)
    result = engine.entrenar("svm_linear", "target", list(df.columns[:-1]), cv=2)
    assert "ROC_AUC" in result.metrics
    assert "Log_loss" not in result.metrics
    assert result.details["classes"].tolist() == ["a", "b"]


def test_insufficient_classes(iris):
    iris["target"] = 0
    with pytest.raises(FrameworkError):
        Clasificacion(iris).entrenar("logistic", "target", list(iris.columns[:-1]))
    df = pd.DataFrame({"x": range(12), "target": [0] * 10 + [1] * 2})
    with pytest.raises(FrameworkError):
        Clasificacion(df).entrenar("logistic", "target", ["x"], cv=5)


def test_mixed_missing(iris):
    iris["category"] = np.where(iris.index % 2, "a", "b")
    iris.iloc[0, 0] = np.nan
    engine = Clasificacion(iris)
    result = engine.entrenar("logistic", "target", [c for c in iris if c != "target"], cv=2)
    unseen = iris.drop(columns="target").copy()
    unseen["category"] = "unseen"
    assert len(engine.predecir(result, unseen)) == len(iris)
