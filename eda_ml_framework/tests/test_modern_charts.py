"""Verify chart data, transformations, interactive contracts and real model diagnostics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.datasets import load_diabetes, load_iris, make_blobs
from test_integration import active_app, navigate

from atlas.models import Classification, Clustering, Regression
from atlas.models.comparison import ExperimentComparison
from atlas.models.visualization import chart_frame, diagnostic_bars, plot


def test_correlation_honors_method_and_column_filter():
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 4, 9, 100], "z": [2, 1, 2, 1]})
    fig = plot(frame, "correlation", method="spearman", columns=["x", "y"])
    np.testing.assert_allclose(fig.data[0].z, frame[["x", "y"]].corr(method="spearman"))
    assert not np.isclose(np.asarray(fig.data[0].z)[0, 1], frame.x.corr(frame.y))
    assert list(fig.data[0].x) == ["x", "y"]


def test_missing_summary_percentages_and_download_match_full_data():
    frame = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, 4]})
    fig = plot(frame, "missing_summary")
    actual = dict(zip(fig.data[0].y, fig.data[0].x))
    assert actual == {"a": 50.0, "b": 0.0}
    downloaded = chart_frame(fig)
    assert set(downloaded.X) == {0.0, 50.0}
    assert "Faltantes" in fig.data[0].hovertemplate


def test_visual_sampling_preserves_source_indices_and_does_not_modify_metrics():
    frame = pd.DataFrame(
        {"x": range(1200), "y": range(1200)}, index=[f"row-{i}" for i in range(1200)]
    )
    first = plot(frame, "scatter", "x", "y", max_points=1000)
    second = plot(frame, "scatter", "x", "y", max_points=1000)
    assert len(first.data[0].x) == 1000
    np.testing.assert_array_equal(first.data[0].x, second.data[0].x)
    for value, custom in zip(first.data[0].x, first.data[0].customdata):
        assert custom[0] == f"row-{value}"
    assert "1,000" in first.layout.meta["atlas_caption"]
    histogram = plot(frame, "histogram", "x")
    assert len(histogram.data[0].x) == len(frame)
    assert len(frame) == 1200


def test_trace_export_respects_selected_series():
    fig = plot(
        pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1], "g": ["a", "b", "a", "b"]}),
        "scatter",
        "x",
        "y",
        "g",
    )
    fig.data[1].visible = "legendonly"
    exported = chart_frame(fig)
    assert set(exported.Serie) == {fig.data[0].name}
    assert len(exported) == 2


def test_statistical_charts_are_interactive_with_valid_density():
    frame = pd.DataFrame({"x": np.linspace(-3, 3, 100)})
    density = plot(frame, "kde", "x")
    assert isinstance(density, go.Figure)
    assert np.trapezoid(density.data[0].y, density.data[0].x) == pytest.approx(1, abs=0.01)
    qq = plot(frame, "qq", "x")
    np.testing.assert_allclose(qq.data[0].y, np.sort(frame.x))
    pairs = plot(frame.assign(y=frame.x * 2), "pairplot")
    assert any(trace.type == "histogram" for trace in pairs.data)
    assert any(trace.type == "scattergl" for trace in pairs.data)


def test_tree_and_dendrogram_use_trained_structure():
    iris = load_iris(as_frame=True).frame
    classifier = Classification(iris)
    result = classifier.train("tree", "target", list(iris.columns[:-1]), cv=3)
    tree = classifier.plot_result(result, "tree")
    assert isinstance(tree, go.Figure)
    assert any(
        "Registros de entrenamiento" in value
        for trace in tree.data
        if trace.customdata is not None
        for value in trace.customdata
    )
    values, _ = make_blobs(n_samples=20, random_state=42)
    cluster = Clustering(pd.DataFrame(values, columns=["x", "y"]))
    dendrogram = cluster.dendrogram(["x", "y"])
    assert isinstance(dendrogram, go.Figure)
    assert len(dendrogram.data) == 19


def test_confusion_and_regression_hover_preserve_calculated_values():
    iris = load_iris(as_frame=True).frame
    classifier = Classification(iris)
    classified = classifier.train("logistic", "target", list(iris.columns[:-1]), cv=3)
    normalized = classifier.plot_result(classified, "confusion_normalized")
    np.testing.assert_array_equal(normalized.data[0].customdata, classified.details["confusion"])
    roc = classifier.plot_result(classified, "roc")
    assert any("AUC" in trace.name for trace in roc.data)
    diabetes = load_diabetes(as_frame=True).frame
    regressor = Regression(diabetes)
    result = regressor.train("ridge", "target", list(diabetes.columns[:-1]), cv=3)
    scatter = regressor.plot_result(result, "actual_predicted")
    for real, predicted, extra in zip(
        scatter.data[0].x, scatter.data[0].y, scatter.data[0].customdata
    ):
        assert float(extra[0]) == pytest.approx(real - predicted)
        assert str(extra[1]) in set(map(str, result.predictions.index))
    np.testing.assert_allclose(np.sort(scatter.data[0].x), np.sort(result.predictions.real))


def test_coefficient_signs_and_metric_comparison_preserve_values():
    coefficients = pd.DataFrame({"weight": [-3.0, 2.0]}, index=["negative", "positive"])
    fig = diagnostic_bars(coefficients, "coefficients")
    assert {trace.name.split(",")[0] for trace in fig.data} == {"Positivo", "Negativo"}
    frame = load_diabetes(as_frame=True).frame
    model = Regression(frame)
    first = model.train("ridge", "target", list(frame.columns[:-1]), cv=3)
    second = model.train("linear_multiple", "target", list(frame.columns[:-1]), cv=3)
    ranking = ExperimentComparison.chart([first, second], "RMSE", direction="min")
    assert any(trace.name == "Favorecido según el criterio" for trace in ranking.data)
    comparison = ExperimentComparison.chart(
        [first, second], "RMSE", "metrics", metrics=["MAE", "RMSE"]
    )
    assert len(comparison.data) == 4


def test_chart_controls_and_fullscreen_do_not_retrain_models():
    app = active_app()
    navigate(app, "🔎 Exploración")
    next(item for item in app.selectbox if item.label == "Gráfico").set_value("scatter").run()
    next(item for item in app.selectbox if item.label == "Variable Y").set_value("target").run()
    app.checkbox(key="chart_inspect_eda").set_value(True).run()
    assert not app.exception and not app.session_state["experiments"]
    app.button(key="chart_fullscreen_eda").click().run()
    assert not app.exception
    app.checkbox(key="chart_zoom_eda").set_value(True).run()
    navigate(app, "🏠 Inicio")
    navigate(app, "🔎 Exploración")
    assert app.checkbox(key="chart_zoom_eda").value


def test_undefined_correlation_mask_is_not_exported_as_an_actual_value():
    frame = pd.DataFrame({"a": [1, 2, 3], "constant": [1, 1, 1]})
    exported = chart_frame(plot(frame, "correlation"))
    assert len(exported) == 4
    assert exported.Valor.isna().sum() == 3
