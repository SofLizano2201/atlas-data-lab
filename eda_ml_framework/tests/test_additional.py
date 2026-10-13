"""Casos límite, adaptadores e integraciones opcionales sin dependencias obligatorias."""

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from framework import AnalisisDatosExploratorio as EDA
from framework import Clasificacion, Clustering, Regresion
from framework.estimators import EncodedClassifier
from framework.exceptions import FrameworkError, OptionalDependencyError
from framework.results import compare, export_frame, html_report


def test_optional_errors(monkeypatch, blobs, regression, iris):
    for module in ["umap", "xgboost", "sklearn_extra.cluster"]:
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(OptionalDependencyError, match="umap"):
        Clustering(blobs).reducir(list(blobs), "umap")
    with pytest.raises(OptionalDependencyError, match="K-Medoids"):
        Clustering(blobs).entrenar("kmedoids", list(blobs))
    with pytest.raises(OptionalDependencyError, match="xgboost"):
        Regresion(regression).crear_estimador("xgboost", {})
    with pytest.raises(OptionalDependencyError, match="xgboost"):
        Clasificacion(iris).crear_estimador("xgboost", {})


def test_encoded_classifier(iris):
    x = iris.drop(columns="target")
    y = iris.target.map({0: "a", 1: "b", 2: "c"})
    estimator = clone(EncodedClassifier(LogisticRegression(max_iter=500))).fit(x, y)
    assert set(estimator.predict(x)) == set(y)
    assert estimator.predict_proba(x).shape == (150, 3)


def test_incompatible_comparison(regression):
    engine = Regresion(regression)
    first = engine.entrenar("ridge", "target", ["a"], cv=2)
    second = engine.entrenar("ridge", "target", ["b"], cv=2)
    with pytest.raises(FrameworkError):
        compare([first, second])
    with pytest.raises(FrameworkError):
        compare([])


def test_safe_exports_and_html():
    data = pd.DataFrame({"text": ["=SUM(1,2)", "<script>alert(1)</script>"], "x": [1, 2]})
    exported = export_frame(data)
    assert "'=SUM" in exported.decode("utf-8-sig")
    report = html_report("<script>", {"test": data})
    assert "<script>" not in report
    assert "&lt;script&gt;" in report


def test_further_eda_cases():
    data = pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [1.0, 2.0, np.nan], "empty": [np.nan] * 3})
    eda = EDA(data)
    assert "empty" not in eda.transformar("drop_null_columns", ["empty"], umbral=0.5)
    with pytest.raises(FrameworkError):
        eda.transformar("mean", ["empty"])
    duplicate = EDA(pd.DataFrame({"x": [1, 1, 2]}))
    assert len(duplicate.transformar("duplicates")) == 2
    with pytest.raises(FrameworkError):
        eda.transformar("drop_columns", list(data))
    with pytest.raises(FrameworkError):
        eda.transformar("mean", ["missing"])
    with pytest.raises(FrameworkError):
        EDA(pd.DataFrame({"x": [-1.0, 2.0], "y": [np.nan, 2.0]})).transformar("log")
    assert eda.transformar_personalizado(lambda df: df.fillna(0)).isna().sum().sum() == 0
    with pytest.raises(FrameworkError):
        EDA(pd.DataFrame({"A B": [1], "a_b": [2]})).transformar("normalize_names")


def test_statistics_and_plots():
    rng = np.random.default_rng(4)
    data = pd.DataFrame(
        {"x": rng.normal(size=100), "y": rng.normal(size=100), "c": ["a"] * 50 + ["b"] * 50}
    )
    eda = EDA(data)
    for method in [
        "media",
        "mediana",
        "minimo",
        "maximo",
        "rango",
        "varianza",
        "desviacion_estandar",
        "rango_intercuartilico",
        "coeficiente_variacion",
        "asimetria",
        "curtosis",
    ]:
        assert np.isfinite(getattr(eda, method)("x"))
    assert eda.moda("c").size == 2
    assert eda.cuartiles("x").size == 3
    assert eda.estadisticas_agrupadas("c", "x").shape[0] == 2
    assert eda.categorias_raras("c", 0.6).shape[0] == 2
    for kind in ["scatter", "line"]:
        assert eda.grafico(kind, "x", "y") is not None
    assert eda.grafico("pairplot") is not None
    for test in ["shapiro", "normaltest"]:
        assert 0 <= eda.prueba(test, ["x"])["p_value"] <= 1
    for test in ["levene", "mannwhitney"]:
        assert 0 <= eda.prueba(test, ["x", "y"])["p_value"] <= 1
    from scipy.stats import norm

    quantiles = norm.ppf((np.arange(100) + 0.5) / 100)
    normal = EDA(pd.DataFrame({"x": quantiles, "y": rng.permutation(quantiles) + 0.1}))
    for test in ["ttest", "anova"]:
        assert 0 <= normal.prueba(test, ["x", "y"])["p_value"] <= 1
    with pytest.raises(FrameworkError, match="Normalidad"):
        EDA(pd.DataFrame({"x": np.exp(quantiles), "y": quantiles})).prueba("ttest", ["x", "y"])
    table = pd.DataFrame({"a": ["a", "b"] * 50, "b": ["a", "a", "b", "b"] * 25})
    assert EDA(table).prueba("chi2", ["a", "b"])["p_value"] == 1


def test_backend_errors(regression):
    engine = Regresion(regression.head(10))
    with pytest.raises(FrameworkError):
        engine.entrenar("ridge", "target", ["a"], cv=10)
    with pytest.raises(FrameworkError):
        Regresion(regression).entrenar("forest", "target", ["a"], parametros={"n_estimators": 1001})
    result = Regresion(regression).entrenar("ridge", "target", ["a"], cv=2)
    with pytest.raises(FrameworkError):
        engine.predecir(result, regression[["b"]])


def test_cluster_export_bundle(blobs):
    from io import BytesIO

    import joblib

    result = Clustering(blobs).entrenar("kmeans", list(blobs))
    bundle = joblib.load(BytesIO(result.model_bytes()))
    assert "preprocessor" in bundle and "modelo" in bundle
    x = bundle["preprocessor"].transform(blobs)
    np.testing.assert_array_equal(bundle["modelo"].predict(x), result.details["labels"])
