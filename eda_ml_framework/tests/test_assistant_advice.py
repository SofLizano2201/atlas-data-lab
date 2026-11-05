"""Advice is actionable, tied to actual context and does not execute changes."""

import pandas as pd
import pytest
from sklearn.datasets import load_iris
from test_integration import active_app, navigate

from atlas.agents.agent import AgentFactory, NoProvider
from atlas.controllers.agent_controller import AgentController
from atlas.core.session import AgentContext
from atlas.models import Classification, ExploratoryDataAnalysis


@pytest.mark.parametrize(
    "values,expected",
    [([1.0, None, 3.0], "faltan"), ([1.0, 1.0, 3.0], "repetidas"), ([1.0, 2.0, 3.0], "variables")],
)
def test_eda_advice_uses_actual_quality_without_mutation(values, expected):
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": values})))
    advice = AgentFactory.get_agent("eda", context, NoProvider()).advice()
    assert expected in advice.title
    assert advice.reason and advice.next_step and advice.action_label
    assert context.eda.version_count == 1 and not context.pending
    assert not context.experiments and not context.conversation


def test_preparation_advice_does_not_link_back_to_the_same_page():
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, None]})))
    context.selections = {"workspace_page": "preparation"}
    assert AgentFactory.get_agent("eda", context).advice().destination is None


def test_comparison_without_models_has_no_runnable_dead_end():
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1, 2, 3]})))
    advice = AgentFactory.get_agent("comparison", context).advice()
    assert not advice.runnable and advice.destination == "home"


def test_new_target_does_not_recommend_errors_from_an_old_target():
    frame = load_iris(as_frame=True).frame
    result = Classification(frame).train("logistic", "target", list(frame.columns[:-1]), cv=2)
    context = AgentContext(ExploratoryDataAnalysis(frame), experiments=[result])
    agent = AgentFactory.get_agent("classification", context)
    context.selections = {"analysis_selection": {"target": "target"}}
    assert agent.advice().goal == "errors"
    context.selections = {"analysis_selection": {"target": "sepal length (cm)"}}
    assert agent.advice().goal == "classes"
    assert not any(idea.goal in {"errors", "evaluate"} for idea in agent.suggestions().ideas)


def test_context_reads_current_target_widget_before_form_rendering():
    frame = load_iris(as_frame=True).frame
    state = {
        "eda": ExploratoryDataAnalysis(frame),
        "experiments": [],
        "models": {},
        "navigation": "📈 Regresión",
        "analysis_selection": {"target": "old"},
        "analysis_target_regresion": "target",
    }
    context = AgentController.context_for(state)
    assert context.selections["analysis_selection"]["target"] == "target"


def test_main_advice_navigates_to_preparation_without_applying_changes():
    app = active_app()
    navigate(app, "🔎 Exploración")
    app.button(key="open_EDAAgent").click().run()
    assert not app.exception
    app.button(key="idea_EDAAgent_diagnose").click().run()
    app.button(key="continue_page_EDAAgent").click().run()
    assert app.session_state["navigation"] == "🧹 Preparación"
    assert app.session_state["eda"].version_count == 1
    assert not app.exception


def test_stale_review_is_not_presented_as_current_advice():
    app = active_app()
    navigate(app, "🔎 Exploración")
    app.button(key="open_EDAAgent").click().run()
    app.button(key="idea_EDAAgent_diagnose").click().run()
    model = app.session_state["eda"]
    model.apply(model.df.iloc[:-1], "explicit test edit", confirm=True)
    app.run()
    assert app.session_state["assistant_flow_EDAAgent"]["phase"] == "ideas"
    assert "response" not in app.session_state["assistant_flow_EDAAgent"]
    assert not app.exception


def test_comparison_finds_a_valid_pair_even_if_latest_model_is_incompatible():
    frame = load_iris(as_frame=True).frame
    engine = Classification(frame)
    features = list(frame.columns[:-1])
    first = engine.train("logistic", "target", features, cv=2)
    second = engine.train("tree", "target", features, cv=2)
    latest = engine.train("tree", "target", features, cv=2, test_size=0.3)
    context = AgentContext(ExploratoryDataAnalysis(frame), experiments=[first, second, latest])
    agent = AgentFactory.get_agent("comparison", context, NoProvider())
    assert agent.advice().goal == "compare"
    result = agent.run(agent.request("guided:compare"))
    assert result.status == "success", result
    assert set(result.executed_actions[0].parameters["experiment_ids"]) == {first.id, second.id}
    assert result.tools[0].result.tables["seleccion"][0]["metrica"] == "Accuracy"
    assert "alternativa favorecida" in result.recommendations[0].text
