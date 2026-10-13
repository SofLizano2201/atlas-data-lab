"""Regresores, errores académicos y ajuste exclusivo con entrenamiento."""

import numpy as np
import pytest

from framework import Regresion
from framework.exceptions import FrameworkError
from framework.metrics import regression_metrics


@pytest.mark.parametrize("algorithm", [a for a in Regresion.algoritmos() if a != "xgboost"])
def test_regressors(regression, algorithm):
    engine = Regresion(regression)
    features = ["a"] if algorithm == "linear_simple" else ["a", "b", "c"]
    result = engine.entrenar(
        algorithm,
        "target",
        features,
        cv=2,
        parametros={"hidden_layer_sizes": (8,), "max_iter": 20} if algorithm == "mlp" else {},
    )
    assert np.isfinite(result.metrics["RMSE"])
    assert len(result.predictions) == 16
    assert engine.predecir(result, regression).shape[0] == 80


def test_academic_error():
    values = regression_metrics(np.array([2.0, 4.0]), np.array([1.0, 5.0]), 1)
    assert values["ER"] == pytest.approx(1 / 3)
    assert values["MAE"] == 1
    zero = regression_metrics(np.zeros(2), np.zeros(2), 3)
    assert np.isnan(zero["ER"]) and np.isnan(zero["MAPE"])
    assert zero["sMAPE"] == 0


def test_leakage_and_comparison(regression):
    engine = Regresion(regression)
    first = engine.entrenar("linear_multiple", "target", ["a", "b", "c"], cv=2)
    scaler = (
        first.model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    )
    np.testing.assert_allclose(scaler.mean_, first.details["x_train"].mean())
    engine.entrenar("ridge", "target", ["a", "b", "c"], cv=2)
    assert len(engine.comparar()) == 2
    assert len(first.model_bytes()) > 100
    assert engine.coeficientes(first).shape[0] == 3
    assert engine.importancia(first).shape[0] == 3
    assert engine.curva_aprendizaje(first).shape[0] == 3
    for plot in ["actual_predicted", "residuals", "histogram", "qq"]:
        assert engine.grafico_resultado(first, plot) is not None


def test_invalid_targets(regression):
    engine = Regresion(regression)
    with pytest.raises(FrameworkError):
        engine.entrenar("ridge", "missing", ["a"])
    with pytest.raises(FrameworkError):
        engine.entrenar("ridge", "target", ["target"])
    with pytest.raises(FrameworkError):
        engine.entrenar("linear_simple", "target", ["a", "b"])
    regression.loc[0, "target"] = np.nan
    with pytest.raises(FrameworkError):
        Regresion(regression).entrenar("ridge", "target", ["a"])


def test_search(regression):
    for kind in ["grid", "random"]:
        result = Regresion(regression).entrenar(
            "ridge", "target", ["a", "b"], cv=2, busqueda={"alpha": [0.1, 1.0]}, tipo_busqueda=kind
        )
        assert result.config["mejores_parametros"]["model__alpha"] in [0.1, 1.0]
