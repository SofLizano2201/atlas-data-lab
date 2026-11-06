"""Presentation preserves underlying data and interactive chart semantics."""

import ast
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from atlas.models.visualization import plot, style
from atlas.views.components.tables import prepare_table


def test_table_stripes_preserve_values_and_precision():
    frame = pd.DataFrame({"Categoría": ["A", "B"], "Valor": [1.23456789, None]})
    before = frame.copy(deep=True)
    styled = prepare_table(frame)
    pd.testing.assert_frame_equal(styled.data, before)
    html = styled.to_html()
    assert "#FFFFFF" in html and "#F1F5F9" in html
    pd.testing.assert_frame_equal(frame, before)


def test_large_and_duplicate_index_tables_keep_native_rendering():
    large = pd.DataFrame({str(i): range(3000) for i in range(8)})
    assert prepare_table(large) is large
    duplicated = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
    assert prepare_table(duplicated) is duplicated


def test_chart_theme_has_distinct_series_and_no_white_template():
    data = pd.DataFrame({"Valor": range(10)})
    histogram = plot(data, "histogram", "Valor")
    assert isinstance(histogram, go.Figure)
    assert histogram.data[0].marker.color == "#694D75"
    assert histogram.layout.plot_bgcolor == "#F5F2F8"
    assert list(histogram.data[0].x) == list(data.Valor)
    restyled = style(go.Figure(histogram))
    assert "<br><br>" not in restyled.layout.title.text


def test_application_does_not_render_static_chart_backends():
    root = Path(__file__).parents[1] / "src" / "atlas" / "views"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"pyplot", "bar_chart", "line_chart", "scatter_chart"}
            for node in ast.walk(tree)
        ), path


def test_old_saved_histograms_receive_current_theme():
    old = go.Figure(go.Histogram(x=[1, 2, 3], marker_color="#1B5299"))
    updated = style(old, "Distribución")
    assert updated.data[0].marker.color == "#694D75"
    assert list(updated.data[0].x) == [1, 2, 3]


def test_session_visual_upgrade_preserves_trained_experiments(monkeypatch):
    from sklearn.datasets import load_iris

    from atlas.controllers.autonomous_controller import AutonomousController
    from atlas.controllers.classification_controller import ClassificationController
    from atlas.services.dataset_service import DatasetService
    from atlas.services.presentation_service import upgrade_session_presentation

    frame = load_iris(as_frame=True).frame
    frame.insert(0, "id", range(len(frame)))
    state = {}
    DatasetService.activate(state, frame, "Iris.csv")
    run = AutonomousController.run(state)
    run.figures["chart:id"] = go.Figure(go.Histogram(x=frame.id))
    experiments = list(state["experiments"])
    state.pop("presentation_revision", None)
    monkeypatch.setattr(
        ClassificationController,
        "train",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Do not retrain")),
    )
    upgrade_session_presentation(state)
    assert "chart:id" not in run.figures
    assert all(before is after for before, after in zip(experiments, state["experiments"]))
    assert len(state["experiments"]) == 7
    pd.testing.assert_frame_equal(state["eda"].df, frame)
    assert run.report
