import pandas as pd
import pytest

from atlas.controllers.methodological_controller import MethodologicalController
from atlas.core.exceptions import FrameworkError
from atlas.models.eda import ExploratoryDataAnalysis
from atlas.models.methodological_validation import MethodologicalValidation as Guardian


def engine():
    return ExploratoryDataAnalysis(pd.DataFrame({"x": range(30), "y": [0, 1] * 15}))


def test_clean_data_has_no_findings():
    assert Guardian.validate(engine(), {"target": "y", "predictors": ["x"]}) == []


def test_quality_and_severity():
    eda = ExploratoryDataAnalysis(pd.DataFrame({"x": [1, None, 3], "constant": [1] * 3}))
    findings = {f.finding_id: f for f in Guardian.validate(eda)}
    assert findings["missing"].severity == "warning"
    assert findings["constants"].severity == "suggestion"
    Guardian.enforce(list(findings.values()))


@pytest.mark.parametrize(
    "config,code",
    [
        ({"target": "absent"}, "invalid_target"),
        ({"target": "y", "predictors": ["y"]}, "target_leakage"),
        ({"algorithm": "kmeans", "metric": "cosine"}, "distance"),
    ],
)
def test_blocking(config, code):
    findings = Guardian.validate(engine(), config)
    assert any(f.finding_id == code and f.blocking for f in findings)
    with pytest.raises(FrameworkError):
        Guardian.enforce(findings)


@pytest.mark.parametrize("operation", ["preprocessing", "feature_selection", "resampling"])
def test_explicit_leakage(operation):
    assert Guardian.validate(engine(), metadata={operation + "_fit_scope": "train"}) == []
    findings = Guardian.validate(engine(), metadata={operation + "_fit_scope": "full_dataset"})
    assert findings[0].blocking


def test_group_and_time_leakage_require_metadata():
    train = pd.DataFrame({"group": [1, 2], "time": [2, 4]})
    test = pd.DataFrame({"group": [1, 3], "time": [3, 5]})
    assert Guardian.validate(engine(), train=train, test=test) == []
    findings = Guardian.validate(
        engine(),
        train=train,
        test=test,
        metadata={"group_column": "group", "time_column": "time", "future_prediction": True},
    )
    assert {f.finding_id: f.blocking for f in findings} == {
        "group_leakage": False,
        "temporal_leakage": True,
    }


def test_silhouette_and_cache():
    assert not Guardian.validate(engine(), labels=[0, 0, 1, 1])
    assert Guardian.validate(engine(), labels=[0, 0, 0])[0].severity == "warning"
    state = {"eda": engine()}
    first = MethodologicalController.synchronize(state, {})
    assert MethodologicalController.synchronize(state, {}) is first
    assert len(state["methodological_history"]) == 1


@pytest.mark.parametrize(
    "name", ["eda", "classification", "regression", "clustering", "comparison"]
)
def test_all_specialists_interpret_guardian(name):
    from types import SimpleNamespace

    from atlas.agents.factory import AgentFactory
    from atlas.core.session import AgentContext

    eda = engine()
    context = AgentContext(eda)
    context.application_state = {
        "methodological_findings": Guardian.validate(eda, {"target": "y", "predictors": ["y"]})
    }
    feedback = AgentFactory.get_agent(name, context).on_event(
        SimpleNamespace(id="review", type="FEATURES_CHANGED")
    )
    assert feedback.priority == "critical"
    assert "objetivo" in feedback.title
    assert not context.experiments


def test_invalid_cv_and_imbalance():
    eda = ExploratoryDataAnalysis(pd.DataFrame({"x": range(30), "y": [0] * 28 + [1] * 2}))
    findings = Guardian.validate(eda, {"target": "y", "problem": "classification", "cv": 5})
    assert {f.finding_id for f in findings} >= {"imbalance", "invalid_validation"}


def test_invalid_k():
    assert Guardian.validate(engine(), {"algorithm": "kmeans", "n_clusters": 50})[0].blocking


def test_existing_confirmation_is_preserved():
    eda = engine()
    before = eda.df
    with pytest.raises(FrameworkError):
        eda.apply(before.iloc[:10], "remove rows", confirm=False)
    pd.testing.assert_frame_equal(eda.df, before)


def test_comparison_protocols():
    from types import SimpleNamespace

    a = SimpleNamespace(
        problem="regresion", dataset_id="v1", cohort="same", config={"cv": 3}, details={}, id="a"
    )
    b = SimpleNamespace(
        problem="regresion", dataset_id="v1", cohort="same", config={"cv": 5}, details={}, id="b"
    )
    assert Guardian.comparison([a, a]) == []
    assert Guardian.comparison([a, b])[0].blocking


def test_panel_acknowledgement_and_blocking_controls():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string("""
import streamlit as st
import pandas as pd
from atlas.models.eda import ExploratoryDataAnalysis
from atlas.models.methodological_validation import MethodologicalValidation
from atlas.views.components.methodological_feedback import render_methodological_feedback
st.session_state.methodological_key = "test"
st.session_state.methodological_findings = MethodologicalValidation.validate(
    ExploratoryDataAnalysis(pd.DataFrame({"x": [1, None, 3], "y": [0, 1, 0]})),
    {"target": "y", "predictors": ["y"]})
render_methodological_feedback()
""").run()
    assert not app.exception
    assert sum(b.label == "Continuar de todos modos" for b in app.button) == 1
    next(b for b in app.button if b.label == "Continuar de todos modos").click().run()
    assert not app.exception
    assert not any(b.label == "Continuar de todos modos" for b in app.button)
    assert any(b.label == "Aplicar corrección" for b in app.button)


def test_guardian_correction_needs_confirmation():
    state = {
        "eda": engine(),
        "experiments": [],
        "models": {},
        "analysis_selection": {"problem": "clasificacion", "target": "y", "predictors": ["x", "y"]},
    }
    with pytest.raises(ValueError):
        MethodologicalController.correct_target_selection(state)
    MethodologicalController.correct_target_selection(state, confirm=True)
    assert state["analysis_selection"]["predictors"] == ["x"]
    assert not state["experiments"]
