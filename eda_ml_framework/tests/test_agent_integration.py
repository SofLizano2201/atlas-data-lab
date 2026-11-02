"""Exercise explicit assistance against actual models and preserve independent manual mode."""

import pandas as pd
from test_integration import active_app, navigate

import atlas.agents.agent as agent_module
from atlas.agents.agent import AgentFactory
from atlas.controllers.agent_controller import AgentController
from atlas.core.schemas import ToolCall
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


def test_manual_pages_do_not_construct_an_llm_provider(monkeypatch):
    def forbidden_provider():
        raise AssertionError("Manual operation must not initialize an LLM provider")

    monkeypatch.setattr(agent_module, "configured_provider", forbidden_provider)
    app = active_app()
    for route in [
        "🏠 Inicio",
        "🔎 Exploración",
        "🧹 Preparación",
        "🧩 Clustering",
        "🎯 Clasificación",
        "📈 Regresión",
        "⚖️ Experimentos",
        "📤 Reportes",
    ]:
        navigate(app, route)
        assert not app.exception


def test_quality_guidance_is_grounded_in_real_missing_values():
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, None, 3.0]})))
    agent = AgentFactory.get_agent("eda", context)
    response = agent.run(agent.request("Calidad", [ToolCall(tool="eda.profile")]))
    assert response.status == "success"
    assert response.evidence[0].metrics["celdas_nulas"] == 1
    assert "faltantes" in response.recommendations[0].text
    assert response.recommendations[0].mode == "deterministic_guidance"
    assert response.executed_actions[0].tool == "eda.profile"


def test_class_distribution_uses_existing_frequency_method():
    context = AgentContext(
        ExploratoryDataAnalysis(pd.DataFrame({"x": range(20), "target": [0] * 18 + [1] * 2}))
    )
    agent = AgentFactory.get_agent("classification", context)
    response = agent.run(
        agent.request(
            "Clases",
            [ToolCall(tool="classification.class_distribution", parameters={"column": "target"})],
        )
    )
    assert response.status == "success"
    assert "clase dominante" in response.recommendations[0].text
    assert response.tools[0].result.tables["frecuencias"]


def test_agent_configuration_updates_real_streamlit_controls_only_after_confirmation():
    app = active_app()
    navigate(app, "🎯 Clasificación")
    app.button(key="open_ClassificationAgent").click().run()
    app.selectbox(key="tool_ClassificationAgent").set_value("analysis.configure").run()
    params = {
        "problem": "classification",
        "target": "target",
        "predictors": ["sepal length (cm)", "sepal width (cm)"],
        "scaler": "robust",
        "algorithm": "forest",
        "cv": 3,
    }
    import json

    app.text_area(key="args_ClassificationAgent_analysis.configure").set_value(
        json.dumps(params)
    ).run()
    app.button(key="quick_ClassificationAgent").click().run()
    assert app.session_state["agent_last_ClassificationAgent"].status == "pending_confirmation"
    assert app.selectbox(key="analysis_scaler_clasificacion").value == "standard"
    app.button(key="confirm_ClassificationAgent").click().run()
    assert not app.exception
    assert app.selectbox(key="analysis_scaler_clasificacion").value == "robust"
    assert app.multiselect(key="atlas_algorithms_clasificacion").value == ["forest"]
    assert len(app.session_state["experiments"]) == 0


def test_cross_module_configuration_does_not_change_selections():
    state = {
        "eda": ExploratoryDataAnalysis(pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})),
        "experiments": [],
        "models": {},
    }
    context = AgentController.context_for(state)
    controller = AgentController(context, "ClassificationAgent", state=state)
    controller.open()
    response = controller.run(
        controller.request(
            "Configura",
            [
                ToolCall(
                    tool="analysis.configure",
                    parameters={"problem": "regression", "target": "target", "predictors": ["x"]},
                )
            ],
        )
    )
    result = controller.confirm(response.confirmation_id)
    assert result.status == "error"
    assert "analysis_selection" not in state
