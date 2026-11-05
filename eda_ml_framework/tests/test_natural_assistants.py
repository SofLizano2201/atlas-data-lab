"""Contextual ideas, safe execution and real dialog continuity without credentials."""

import pandas as pd
import pytest
from sklearn.datasets import load_iris
from streamlit.testing.v1 import AppTest
from test_integration import APP, active_app, navigate

from atlas.agents.agent import AgentFactory, NoProvider
from atlas.controllers.agent_controller import AgentController
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


@pytest.mark.parametrize(
    "kind", ["eda", "clustering", "classification", "regression", "comparison"]
)
def test_opening_suggests_existing_actions_without_running_them(kind):
    frame = load_iris(as_frame=True).frame
    context = AgentContext(ExploratoryDataAnalysis(frame))
    context.selections = {
        "analysis_selection": {"target": "target", "predictors": list(frame.columns[:-1])}
    }
    agent = AgentFactory.get_agent(kind, context, NoProvider())
    overview = agent.suggestions()
    assert 2 <= len(overview.ideas) <= 3
    assert len({idea.id for idea in overview.ideas}) == len(overview.ideas)
    assert overview.welcome and not context.experiments and not context.pending
    assert not context.conversation and context.eda.version_count == 1
    for idea in overview.ideas:
        assert idea.goal or idea.destination
        if idea.goal:
            assert all(
                call.tool in agent.get_allowed_tools() for call in agent.local_calls(idea.goal)
            )


def test_metadata_reuses_profile_only_until_dataset_version_changes(monkeypatch):
    model = ExploratoryDataAnalysis(pd.DataFrame({"x": [1, 2, 3]}))
    calls = []
    original = model.profile

    def counted():
        calls.append(True)
        return original()

    monkeypatch.setattr(model, "profile", counted)
    context = AgentContext(model)
    metadata = context.metadata()
    metadata["columns"].clear()
    assert context.metadata()["columns"]
    AgentFactory.get_agent("eda", context).suggestions()
    assert len(calls) == 1
    model.apply(model.df.iloc[:-1], "test version", confirm=True)
    assert context.metadata()["rows"] == 2 and len(calls) == 2


def test_technical_details_open_with_and_without_completed_review():
    app = active_app()
    navigate(app, "🔎 Exploración")
    app.button(key="open_EDAAgent").click().run()
    app.toggle(key="assistant_technical_EDAAgent").set_value(True).run()
    assert not app.exception
    assert any(item.label == "Resumen" for item in app.tabs)
    app.toggle(key="assistant_technical_EDAAgent").set_value(False).run()
    idea = next(
        button for button in app.button if button.key and button.key.startswith("idea_EDAAgent_")
    )
    idea.click().run()
    app.toggle(key="assistant_technical_EDAAgent").set_value(True).run()
    assert not app.exception
    assert app.selectbox(key="tool_EDAAgent")


@pytest.mark.parametrize(
    "route,name",
    [
        ("🔎 Exploración", "EDAAgent"),
        ("🧩 Clustering", "ClusteringAgent"),
        ("🎯 Clasificación", "ClassificationAgent"),
        ("📈 Regresión", "RegressionAgent"),
        ("⚖️ Experimentos", "ModelComparisonAgent"),
    ],
)
def test_all_specialist_dialogs_hide_internal_controls_by_default(route, name):
    app = active_app()
    navigate(app, route)
    background_text_areas = len(app.text_area)
    background_json = len(app.get("json"))
    app.button(key=f"open_{name}").click().run()
    assert not app.exception and not app.error
    assert (
        2
        <= len(
            [
                button
                for button in app.button
                if button.key and button.key.startswith(f"idea_{name}_")
            ]
        )
        <= 3
    )
    assert len(app.text_area) == background_text_areas
    assert len(app.get("json")) == background_json
    assert not any(item.key == f"tool_{name}" for item in app.selectbox)
    assert app.session_state[f"assistant_flow_{name}"]["phase"] == "ideas"
    assert not app.session_state["experiments"]
    app.button(key=f"close_{name}").click().run()
    assert not app.session_state[f"agent_open_{name}"]
    assert len(app.session_state["eda"].df) == 150


def test_dialog_retains_completed_review_when_reopened():
    app = active_app()
    navigate(app, "🔎 Exploración")
    app.button(key="open_EDAAgent").click().run()
    app.button(key="idea_EDAAgent_diagnose").click().run()
    response = app.session_state["agent_last_EDAAgent"]
    count = len(app.session_state["agent_context"].conversation["EDAAgent"])
    app.button(key="close_EDAAgent").click().run()
    app.button(key="open_EDAAgent").click().run()
    assert app.session_state["agent_last_EDAAgent"].request_id == response.request_id
    assert app.session_state["assistant_flow_EDAAgent"]["phase"] == "result"
    assert len(app.session_state["agent_context"].conversation["EDAAgent"]) == count
    assert not any(button.key == "idea_EDAAgent_diagnose" for button in app.button)


def test_training_idea_waits_for_confirmation_and_closing_cancels_it():
    app = active_app()
    navigate(app, "🧩 Clustering")
    app.button(key="open_ClusteringAgent").click().run()
    app.button(key="idea_ClusteringAgent_compare_groups").click().run()
    assert not app.session_state["experiments"]
    response = app.session_state["agent_last_ClusteringAgent"]
    assert response.status == "pending_confirmation"
    assert any("Antes de continuar" in item.value for item in app.markdown)
    assert sum("Cantidad de grupos:" in item.value for item in app.caption) == 3
    app.button(key="close_ClusteringAgent").click().run()
    assert response.confirmation_id not in app.session_state["agent_context"].pending
    app.button(key="open_ClusteringAgent").click().run()
    assert app.session_state["assistant_flow_ClusteringAgent"]["phase"] == "explanation"
    app.button(key="resume_ClusteringAgent").click().run()
    app.button(key="confirm_ClusteringAgent").click().run(timeout=60)
    assert len(app.session_state["experiments"]) == 3
    assert any("se completaron 3 de 3" in item.value for item in app.success)
    app.run()
    assert len(app.session_state["experiments"]) == 3 and not app.exception


def test_no_dataset_assistance_leads_to_real_import_screen():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.button(key="open_EDAAgent").click().run()
    assert not app.exception
    app.button(key="assistant_load_EDAAgent").click().run()
    assert app.session_state["navigation"] == "📂 Datos"
    assert "eda" not in app.session_state and not app.exception


def test_preparation_proposes_specific_columns_and_only_previews():
    frame = pd.DataFrame(
        {"price": [1.0, None, 3.0, 4.0], "region": ["a", "a", None, "b"], "empty": [None] * 4}
    )
    context = AgentContext(ExploratoryDataAnalysis(frame))
    agent = AgentFactory.get_agent("eda", context, NoProvider())
    exploration_ids = [idea.id for idea in agent.suggestions().ideas]
    context.selections = {"workspace_page": "preparation"}
    ideas = agent.suggestions().ideas
    assert [idea.id for idea in ideas] != exploration_ids
    assert {idea.id for idea in ideas} == {"prepare_sparse", "prepare_median", "prepare_mode"}
    call = agent.local_calls("prepare_median")[0]
    assert call.parameters == {"operation": "median", "columns": ["price"]}
    response = agent.run(agent.request("guided:prepare_median"))
    assert response.status == "success"
    assert response.tools[0].result.proposal_id
    pd.testing.assert_frame_equal(context.eda.df, frame)
    assert context.eda.version_count == 1


def test_clustering_trials_compare_real_results_without_retraining():
    context = AgentContext(ExploratoryDataAnalysis(load_iris(as_frame=True).data))
    agent = AgentFactory.get_agent("clustering", context, NoProvider())
    response = agent.run(agent.request("guided:compare_groups"))
    assert response.status == "pending_confirmation" and not context.experiments
    response = agent.confirm(response.confirmation_id)
    assert response.status == "success" and len(context.experiments) == 3
    calls = agent.local_calls("compare")
    assert len(calls[0].parameters["experiment_ids"]) == 3
    compared = agent.run(agent.request("guided:compare"))
    assert compared.status == "success"
    assert compared.tools[0].result.tables["seleccion"]
    assert "grupos" in compared.recommendations[0].text
    assert len(context.experiments) == 3


def test_preparation_preview_can_be_applied_only_after_confirmation():
    app = active_app()
    navigate(app, "🧹 Preparación")
    app.button(key="open_EDAAgent").click().run()
    app.button(key="idea_EDAAgent_prepare_duplicates").click().run()
    assert not app.exception
    original_rows = len(app.session_state["eda"].df)
    apply = next(
        button for button in app.button if button.key and button.key.startswith("apply_preview_")
    )
    apply.click().run()
    assert app.session_state["agent_last_EDAAgent"].status == "pending_confirmation"
    assert len(app.session_state["eda"].df) == original_rows
    app.button(key="confirm_EDAAgent").click().run()
    assert not app.exception
    assert len(app.session_state["eda"].df) == original_rows - 1


def test_outlier_results_are_visible_without_technical_details():
    app = active_app()
    navigate(app, "🧹 Preparación")
    app.button(key="open_EDAAgent").click().run()
    app.button(key="idea_EDAAgent_outliers").click().run()
    if app.session_state["agent_last_EDAAgent"].status == "pending_confirmation":
        app.button(key="confirm_EDAAgent").click().run()
    assert not app.exception
    assert any("Registros con valores poco habituales" == metric.label for metric in app.metric)
    response = app.session_state["agent_last_EDAAgent"]
    records = response.tools[0].result.tables["outliers"]
    if records:
        assert any(len(table.value) == len(records) for table in app.dataframe)
    else:
        assert any("No se encontraron registros" in item.value for item in app.info)
    assert not app.toggle(key="assistant_technical_EDAAgent").value


def test_changed_selection_invalidates_pending_idea():
    state = {
        "eda": ExploratoryDataAnalysis(load_iris(as_frame=True).frame),
        "experiments": [],
        "models": {},
        "navigation": "🎯 Clasificación",
        "analysis_target_clasificacion": "target",
    }
    context = AgentController.context_for(state)
    controller = AgentController(context, "ClassificationAgent", state=state)
    controller.open()
    response = controller.choose("configure")
    assert response.status == "pending_confirmation"
    state["analysis_target_clasificacion"] = "sepal length (cm)"
    AgentController.context_for(state)
    assert controller.flow()["phase"] == "ideas"
    assert response.confirmation_id not in context.pending


def test_clustering_agent_reads_current_widgets_and_uses_selected_columns():
    frame = load_iris(as_frame=True).data
    state = {
        "eda": ExploratoryDataAnalysis(frame),
        "experiments": [],
        "models": {},
        "navigation": "🧩 Clustering",
        "analysis_selection": {"predictors": list(frame.columns)},
        "analysis_cluster_columns": [frame.columns[1], frame.columns[2]],
        "analysis_cluster_scaler": "robust",
    }
    context = AgentController.context_for(state)
    agent = AgentFactory.get_agent("clustering", context, NoProvider())
    call = agent.local_calls("train")[0]
    assert call.parameters["columns"] == state["analysis_cluster_columns"]
    assert call.parameters["scaler"] == "robust"


def test_recommendations_exclude_identifiers_dates_and_constants_but_not_unique_prices():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "date": ["2020-01-01"] * 4,
            "price": [10.0, 20.0, None, 40.0],
            "constant": [1] * 4,
        }
    )
    context = AgentContext(ExploratoryDataAnalysis(frame))
    roles = context.eda.feature_roles()
    assert roles["suggested"] == ["price"]
    agent = AgentFactory.get_agent("clustering", context)
    call = agent.local_calls("configure")[0]
    assert call.parameters["predictors"] == ["price"]
    context.selections = {"workspace_page": "preparation"}
    eda = AgentFactory.get_agent("eda", context)
    assert eda.local_calls("prepare_median")[0].parameters["columns"] == ["price"]


def test_large_dataset_does_not_offer_unsupported_clustering_calculations():
    context = AgentContext(
        ExploratoryDataAnalysis(pd.DataFrame({"x": range(6000), "y": range(6000)}))
    )
    agent = AgentFactory.get_agent("clustering", context)
    overview = agent.suggestions()
    assert "muestra" in overview.welcome
    assert not {"compare_groups", "elbow", "project", "train"}.intersection(
        idea.id for idea in overview.ideas
    )


def test_empty_dataset_is_rejected_without_creating_an_assistant_session():
    with pytest.raises(ValueError):
        ExploratoryDataAnalysis(pd.DataFrame())
