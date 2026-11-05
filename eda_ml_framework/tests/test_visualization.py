"""Regresiones visuales semánticas: colores, rangos, orden y datos realmente representados."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from framework import Clasificacion, NoSupervisado, Regresion
from framework.exceptions import FrameworkError
from framework.visualization import PALETTE, before_after, categorical_colors, plot


def test_correlation_has_signed_scale_and_preserves_data():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    original = frame.copy()
    fig = plot(frame, "correlation")
    assert fig.layout.coloraxis.cmin == -1 and fig.layout.coloraxis.cmax == 1
    assert fig.layout.coloraxis.cmid == 0
    assert np.asarray(fig.data[0].z)[0, 1] == pytest.approx(-1)
    pd.testing.assert_frame_equal(frame, original)


def test_missing_scale_is_binary_even_without_missing_values():
    fig = plot(pd.DataFrame({"a": [1, 2]}), "missing")
    assert fig.layout.coloraxis.cmin == 0 and fig.layout.coloraxis.cmax == 1
    assert tuple(fig.layout.coloraxis.colorbar.ticktext) == ("Presente", "Faltante")
    assert not np.asarray(fig.data[0].z).any()


def test_undefined_correlations_are_marked_instead_of_looking_like_zero():
    fig = plot(
        pd.DataFrame({"variable": [1.0, 2.0, 3.0], "constante": [4.0, 4.0, 4.0]}), "correlation"
    )
    assert len(fig.data) == 2
    assert "N/D" in np.asarray(fig.data[1].text)
    assert np.isnan(np.asarray(fig.data[0].z)[1, 1])


def test_groups_are_stable_and_not_background_colors():
    assert categorical_colors(["b", "a"]) == categorical_colors(["a", "b"])
    assert "#F1ECCE" not in PALETTE
    frame = pd.DataFrame({"x": range(12), "y": range(12), "grupo": ["a", "b"] * 6})
    fig = plot(frame, "scatter", "x", "y", "grupo")
    assert len({trace.marker.color for trace in fig.data}) == 2
    assert len({trace.marker.symbol for trace in fig.data}) == 2


def test_continuous_color_has_scale_instead_of_thousands_of_groups():
    frame = pd.DataFrame({"x": range(30), "y": range(30), "valor": range(30)})
    fig = plot(frame, "scatter", "x", "y", "valor")
    assert len(fig.data) == 1
    assert list(fig.data[0].marker.color) == list(frame.valor)
    assert fig.layout.coloraxis.colorscale


def test_histogram_does_not_sum_unrelated_y_values_or_change_x_type():
    frame = pd.DataFrame({"x": [1, 2, 1, 2], "y": [100, 200, 300, 400]})
    fig = plot(frame, "histogram", "x", "y", "x")
    assert all(trace.y is None for trace in fig.data)
    assert all(isinstance(value, (int, np.integer)) for trace in fig.data for value in trace.x)


def test_line_is_sorted_by_x_without_mutating_input():
    frame = pd.DataFrame({"x": [3, 1, 2], "y": [30, 10, 20]})
    fig = plot(frame, "line", "x", "y")
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(frame.x) == [3, 1, 2]


def test_counts_include_missing_and_long_category_names():
    frame = pd.DataFrame(
        {"_atlas_count": ["Categoría con nombre largo", None, "Categoría con nombre largo"]}
    )
    fig = plot(frame, "count", "_atlas_count")
    assert sum(sum(trace.x) for trace in fig.data) == 3
    assert all(trace.orientation == "h" for trace in fig.data)


def test_before_after_uses_comparable_normalization_and_bins():
    fig = before_after(pd.DataFrame({"x": [1, 2, 3]}), pd.DataFrame({"x": [1, 2]}), "x")
    assert len(fig.data) == 2
    assert all(trace.histnorm == "percent" for trace in fig.data)
    assert len({trace.bingroup for trace in fig.data}) == 1


def test_noise_is_explicit_and_visually_distinct():
    coordinates = pd.DataFrame({"PC1": [0, 1, 2, 3], "PC2": [3, 2, 1, 0]})
    fig = NoSupervisado(coordinates).visualizar_representacion(coordinates, [-1, 0, 0, 1])
    noise = next(trace for trace in fig.data if trace.name == "Ruido")
    assert noise.marker.symbol == "x"
    assert all(
        trace.marker.color != noise.marker.color for trace in fig.data if trace.name != "Ruido"
    )


def test_classification_ranges_and_reference_are_correct():
    frame = load_iris(as_frame=True).frame
    model = Clasificacion(frame)
    result = model.entrenar("logistic", "target", list(frame.columns[:-1]), cv=3)
    normalized = model.grafico_resultado(result, "confusion_normalized")
    assert normalized.layout.coloraxis.cmin == 0 and normalized.layout.coloraxis.cmax == 1
    np.testing.assert_allclose(np.asarray(normalized.data[0].z).sum(axis=1), 1)
    roc = model.grafico_resultado(result, "roc")
    reference = next(trace for trace in roc.data if trace.name == "Referencia aleatoria")
    assert list(reference.x) == [0, 1] and list(reference.y) == [0, 1]
    assert list(roc.layout.xaxis.range) == [0, 1]


def test_regression_has_common_ranges_and_zero_residual_reference():
    frame = load_diabetes(as_frame=True).frame
    model = Regresion(frame)
    result = model.entrenar("linear_multiple", "target", list(frame.columns[:-1]), cv=3)
    fig = model.grafico_resultado(result)
    assert fig.layout.xaxis.range == fig.layout.yaxis.range
    residuals = model.grafico_resultado(result, "residuals")
    assert any(shape.y0 == 0 and shape.y1 == 0 for shape in residuals.layout.shapes)


def test_excessive_color_categories_fail_clearly():
    with pytest.raises(FrameworkError, match="categorías"):
        plot(
            pd.DataFrame({"x": range(13), "y": range(13), "g": list(map(str, range(13)))}),
            "scatter",
            "x",
            "y",
            "g",
        )


def test_statistical_graphs_have_labeled_axes():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "g": ["a"] * 3 + ["b"] * 3})
    for kind in ["qq", "kde"]:
        fig = plot(frame, kind, "x", color="g" if kind == "kde" else None)
        assert fig.layout.xaxis.title.text and fig.layout.yaxis.title.text
        assert fig.data[0].hovertemplate
