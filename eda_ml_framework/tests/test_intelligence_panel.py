"""Real Streamlit interactions preserve panel state and explicit execution grants."""

from test_integration import active_app, navigate


def test_panel_is_visible_by_default_and_collapse_survives_navigation():
    app = active_app()
    assert not app.exception
    assert app.button(key="intelligence_collapse")
    key = app.session_state["workflow_dataset_key"]
    execution = app.session_state["automatic_eda"][key]
    app.button(key="intelligence_collapse").click().run()
    navigate(app, "🔎 Exploración")
    assert not app.session_state["intelligence_visible"]
    app.button(key="intelligence_expand").click().run()
    assert app.button(key="intelligence_collapse")
    assert app.session_state["automatic_eda"][key].seconds == execution.seconds
    assert not app.exception


def test_panel_actions_do_not_train_without_confirmation_and_show_right_agent():
    app = active_app()
    navigate(app, "🎯 Clasificación")
    assert "ClassificationAgent" in app.session_state["intelligence_feedback"]
    before = len(app.session_state["experiments"])
    app.button(key="intelligence_action_0").click().run()
    assert not app.exception
    assert app.session_state["agent_open_ClassificationAgent"]
    assert len(app.session_state["experiments"]) == before


def test_completed_training_updates_panel_without_retraining_on_rerun():
    app = active_app()
    navigate(app, "🎯 Clasificación")
    app.button(key="train_clasificacion").click().run(timeout=60)
    assert not app.exception and len(app.session_state["experiments"]) == 1
    feedback = app.session_state["intelligence_feedback"]["ClassificationAgent"]
    assert "evaluado" in feedback.title
    assert any("Accuracy" in evidence for evidence in feedback.evidence)
    app.run()
    assert len(app.session_state["experiments"]) == 1
    navigate(app, "experiment_history")
    assert "ModelComparisonAgent" in app.session_state["intelligence_feedback"]


def test_automatic_results_are_available_after_import_without_eda_click():
    app = active_app()
    assert app.session_state["automatic_eda"]
    navigate(app, "🔎 Exploración")
    assert app.selectbox(key="automatic_eda_result")
    assert not app.exception
