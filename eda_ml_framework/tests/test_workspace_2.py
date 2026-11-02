"""Real offline workflow, interactive chart contracts and session-state regression tests."""

import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.datasets import load_iris
from test_integration import active_app, navigate

from atlas.controllers.workspace_controller import WorkspaceController
from atlas.models import Classification, Clustering, ExploratoryDataAnalysis
from atlas.models.comparison import ExperimentComparison
from atlas.models.visualization import PALETTE, categorical_colors, plot
from atlas.services.dataset_service import DatasetService


def test_interactive_bins_log_facets_and_no_input_mutation():
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8], "group": ["a", "a", "b", "b"]})
    original = frame.copy()
    figure = plot(frame, "histogram", "x", nbins=15, log_x=True, facet="group")
    assert all(trace.nbinsx == 15 for trace in figure.data)
    assert figure.layout.xaxis.type == "log"
    assert len(figure.data) == 2
    assert figure.layout.uirevision
    assert PALETTE[:3] == ["#1B5299", "#694D75", "#9FC2CC"]
    pd.testing.assert_frame_equal(frame, original)
    for options in [{"nbins": 1}, {"facet": "not-present"}]:
        with pytest.raises(ValueError):
            plot(frame, "histogram", "x", **options)
    with pytest.raises(ValueError):
        plot(pd.DataFrame({"x": [0, 1, 2]}), "histogram", "x", log_x=True)


def test_quality_impact_is_computed_in_model_without_mutation():
    model = ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, None, 3.0]}))
    candidate = model.transform("median", ["x"])
    impact = model.transformation_impact(candidate)
    assert impact["missing_before"] == 1 and impact["missing_after"] == 0
    assert impact["changed_columns"] == ["x"]
    assert model.version_count == 1
    assert model.profile()["missing_percentage"] == pytest.approx(100 / 3)


def test_progress_never_marks_unperformed_steps_completed():
    state = {}
    assert not any(WorkspaceController.progress(state).values())
    DatasetService.activate(state, load_iris(as_frame=True).frame, "Iris")
    progress = WorkspaceController.progress(state)
    assert progress["Datos"]
    assert not progress["Exploración"] and not progress["Preparación"] and not progress["Modelado"]
    assert WorkspaceController.context(state, "EDA")["rows"] == 150


def test_new_cluster_diagnostics_and_bounded_inspection():
    frame = load_iris(as_frame=True).frame.drop(columns="target")
    model = Clustering(frame)
    result = model.train("kmeans", list(frame), n_clusters=3)
    assert isinstance(model.cluster_sizes(result), go.Figure)
    assert isinstance(model.profile_heatmap(result), go.Figure)
    group = int(result.details["labels"][0])
    rows = model.inspect_cluster(result, group)
    assert (rows.cluster == group).all() and len(rows) <= 200
    assert len(set(categorical_colors(range(20), max_categories=24).values())) == 20


def test_comparison_views_use_recorded_metrics_and_validate_cohorts():
    frame = load_iris(as_frame=True).frame
    model = Classification(frame)
    results = [
        model.train(name, "target", list(frame.columns[:-1]), cv=2) for name in ["logistic", "tree"]
    ]
    for kind in ["ranking", "runtime", "cost", "cv"]:
        assert isinstance(ExperimentComparison.chart(results, "Accuracy", kind), go.Figure)
    assert len(ExperimentComparison.groups(results)) == 1
    assert results[0].details["cv_scores"]
    assert isinstance(model.plot_result(results[0], "probabilities"), go.Figure)
    results[1].cohort = "different"
    with pytest.raises(ValueError):
        ExperimentComparison.chart(results, "Accuracy")


def test_manual_end_to_end_without_ai(monkeypatch):
    import atlas.agents.agent as agent_module

    def forbidden():
        raise AssertionError("Manual workflow must not construct an LLM")

    monkeypatch.setattr(agent_module, "configured_provider", forbidden)
    app = active_app()
    navigate(app, "🔎 Exploración")
    assert app.session_state["eda_reviewed"]
    navigate(app, "🧹 Preparación")
    next(b for b in app.button if b.label == "Previsualizar transformación").click().run()
    next(b for b in app.button if b.label == "Confirmar estos cambios").click().run()
    assert app.session_state["eda"].version_count == 2
    navigate(app, "🎯 Clasificación")
    app.multiselect(key="atlas_algorithms_clasificacion").set_value(["logistic", "tree"]).run()
    next(b for b in app.button if b.label == "Entrenar y evaluar").click().run(timeout=60)
    assert not app.exception and not app.error
    assert len(app.session_state["experiments"]) == 2
    app.run()
    assert len(app.session_state["experiments"]) == 2
    navigate(app, "experiment_history")
    assert not app.exception and not app.error
    navigate(app, "⚖️ Experimentos")
    next(s for s in app.multiselect if s.label == "Experimentos").set_value(
        app.session_state["experiments"]
    ).run()
    assert not app.exception and not app.error
    navigate(app, "📤 Reportes")
    next(b for b in app.button if b.label == "Generar reporte").click().run()
    assert not app.exception and not app.error
    assert app.session_state["reports"][0].data


def test_offline_panel_guidance_and_history_navigation():
    app = active_app()
    navigate(app, "🔎 Exploración")
    app.button(key="open_EDAAgent").click().run()
    app.button(key="guided_EDAAgent").click().run()
    assert not app.exception and not app.error
    response = app.session_state["agent_last_EDAAgent"]
    assert response.status == "success" and response.provider is None
    app.button(key="dismiss_EDAAgent").click().run()
    assert app.session_state["agent_last_EDAAgent"] is None
    navigate(app, "experiment_history")
    assert not app.exception
