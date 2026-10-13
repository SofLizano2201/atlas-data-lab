"""Carga, herencia, estadísticas y transformaciones con casos límite."""

import numpy as np
import pandas as pd
import pytest

from framework import AnalisisDatosExploratorio as EDA
from framework import Clasificacion, Clustering, NoSupervisado, Regresion, Supervisado
from framework.exceptions import FrameworkError


@pytest.mark.parametrize("fmt", ["csv", "xlsx", "json", "parquet"])
def test_roundtrip(mixed, fmt):
    eda = EDA(mixed)
    result = EDA.cargar(eda.exportar(fmt), formato=fmt)
    assert result.dimensiones() == mixed.shape
    assert result.columnas() == list(mixed)


def test_csv_config_and_invalid():
    eda = EDA.cargar(
        "nombre;valor\nJosé;2".encode("latin-1"),
        formato="csv",
        separador=";",
        codificacion="latin-1",
    )
    assert eda.df.nombre.iloc[0] == "José"
    for fmt in ["exe", "pickle", "joblib"]:
        with pytest.raises(FrameworkError):
            EDA.cargar(b"abc", formato=fmt)
    with pytest.raises(FrameworkError):
        EDA.cargar(b"", formato="csv")
    with pytest.raises(FrameworkError):
        EDA.cargar(b"x" * (25 * 1024**2 + 1), formato="csv")


def test_inheritance(mixed):
    assert issubclass(Clustering, NoSupervisado)
    assert issubclass(Regresion, Supervisado)
    assert issubclass(Clasificacion, Supervisado)
    for cls in [EDA, NoSupervisado, Supervisado, Clustering, Clasificacion, Regresion]:
        assert cls(mixed).dimensiones() == mixed.shape
        assert cls(mixed).nulos().loc["numero", "cantidad"] == 1


def test_empty_and_columns():
    with pytest.raises(FrameworkError):
        EDA(pd.DataFrame())
    assert EDA(pd.DataFrame({"x": [1]})).dimensiones() == (1, 1)
    with pytest.raises(FrameworkError):
        EDA(pd.DataFrame([[1, 2]], columns=["x", "x"]))


def test_stats_and_encapsulation(mixed):
    eda = EDA(mixed)
    copied = eda.df
    copied.loc[0, "numero"] = 999
    assert eda.df.numero.iloc[0] == 1
    s = eda.estadisticas(["numero"])
    assert s.loc["numero", "media"] == pytest.approx(26.75)
    assert "constante" in eda.constantes()
    assert eda.percentiles("numero", [50]).iloc[0] == 3
    assert eda.frecuencias("grupo").absoluta.sum() == 5
    with pytest.raises(FrameworkError):
        eda.estadisticas(["grupo"])


@pytest.mark.parametrize(
    "operation",
    [
        "mean",
        "median",
        "knn",
        "standard",
        "minmax",
        "robust",
        "clip",
        "winsor",
        "drop_outliers",
        "drop_rows",
    ],
)
def test_numeric_transformations(mixed, operation):
    eda = EDA(mixed)
    out = eda.transformar(operation, ["numero"])
    assert isinstance(out, pd.DataFrame)
    assert eda.df.numero.isna().sum() == 1
    assert eda.historial.iloc[-1].estado == "propuesta"


@pytest.mark.parametrize("operation", ["onehot", "ordinal", "mode", "constant"])
def test_category_transformations(mixed, operation):
    out = EDA(mixed).transformar(operation, ["grupo"], valor="desconocido")
    assert len(out) == len(mixed)


def test_history(mixed):
    eda = EDA(mixed)
    out = eda.transformar("median", ["numero"])
    with pytest.raises(FrameworkError):
        eda.aplicar(out, "median")
    eda.aplicar(out, "median", confirmar=True)
    assert not eda.df.numero.isna().any()
    eda.restaurar(0, confirmar=True)
    assert eda.df.numero.isna().any()


def test_outlier_and_stats(mixed):
    eda = EDA(mixed)
    assert eda.outliers(["numero"]).iloc[-1, 0]
    assert not eda.outliers(["constante"], "zscore", 3).any().any()
    for method in ["pearson", "spearman", "kendall"]:
        assert eda.correlacion(method).shape == (2, 2)
    with pytest.raises(FrameworkError):
        eda.prueba("normaltest", ["numero"])
    sample = EDA(pd.DataFrame({"x": np.random.default_rng(1).normal(size=60)}))
    assert 0 <= sample.prueba("shapiro", ["x"])["p_value"] <= 1


def test_other_transformations():
    eda = EDA(pd.DataFrame({"A B": [1.0, 2.0, 3.0, 4.0, 5.0], "group": ["a", "b", "a", "b", "c"]}))
    for operation in ["log", "boxcox", "yeojohnson", "binning", "normalizer"]:
        assert eda.transformar(operation, ["A B"]).shape == (5, 2)
    assert "a_b" in eda.transformar("normalize_names")
    assert eda.transformar("select", ["group"]).shape[1] == 1
    assert eda.transformar("drop_columns", ["group"]).shape[1] == 1
    assert eda.transformar("label", ["group"]).group.nunique() == 3
    assert eda.transformar("convert", ["A B"], tipo="string")["A B"].dtype.name == "string"
    assert EDA(eda.df.assign(z=[1, 2, 1, 2, 1])).vif(["A B", "z"]).shape[0] == 2


@pytest.mark.parametrize(
    "kind",
    [
        "histogram",
        "box",
        "violin",
        "bar",
        "pie",
        "count",
        "kde",
        "qq",
        "missing",
        "correlation",
        "outliers",
    ],
)
def test_plots(kind):
    eda = EDA(pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.0, 4.0]}))
    assert eda.grafico(kind, "x") is not None
    assert "<html" in eda.reporte()
