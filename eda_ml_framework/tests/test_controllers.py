"""Controllers delegate scientific work and preserve explicit session permissions."""

import pandas as pd

from atlas.agents.agent import AgentFactory
from atlas.controllers.agent_controller import AgentController
from atlas.controllers.capabilities import SPECS, CapabilityGateway
from atlas.controllers.eda_controller import EDAController
from atlas.core.schemas import ToolCall
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


def test_controller_and_agent_reuse_the_injected_scientific_method(monkeypatch):
    model = ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}))
    original = model.statistics
    calls = []

    def traced(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "statistics", traced)
    controller = EDAController(model)
    manual = controller.statistics(["x"])
    context = AgentContext(model)
    agent = AgentFactory.get_agent("eda", context, gateway=CapabilityGateway(context))
    response = agent.run(
        agent.request(
            "Estadísticas", [ToolCall(tool="eda.statistics", parameters={"columns": ["x"]})]
        )
    )
    assert response.status == "success" and len(calls) == 2
    assert response.tools[0].result.metrics["x.media"] == manual.loc["x", "media"]
    assert response.evidence[0].metrics == response.tools[0].result.metrics
    assert response.evidence[0].content_hash == model.dataset_version.content_hash


def test_configuration_requires_confirmation_and_closing_invalidates_it():
    model = ExploratoryDataAnalysis(pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]}))
    state = {"eda": model, "experiments": [], "models": {}}
    context = AgentController.context_for(state)
    controller = AgentController(context, "ClassificationAgent", state=state)
    call = ToolCall(
        tool="analysis.configure",
        parameters={
            "problem": "classification",
            "predictors": ["x"],
            "target": "target",
            "scaler": "robust",
        },
    )
    assert controller.run(controller.request("Configura", [call])).status == "blocked"
    controller.open()
    response = controller.run(controller.request("Configura", [call]))
    assert response.status == "pending_confirmation"
    assert "analysis_scaler_clasificacion" not in state
    controller.close()
    assert response.confirmation_id not in context.pending
    controller.open()
    response = controller.run(controller.request("Configura", [call]))
    confirmed = controller.confirm(response.confirmation_id)
    assert confirmed.status == "success"
    assert state["analysis_scaler_clasificacion"] == "robust"
    assert not state["experiments"]


def test_every_registered_tool_has_actual_backend_reference_and_schemas():
    for spec in SPECS.values():
        references = (
            spec.backend_method
            if isinstance(spec.backend_method, tuple)
            else (spec.backend_method,)
        )
        assert all(callable(reference) for reference in references)
        assert spec.arguments.model_json_schema() and spec.output_schema.model_json_schema()
        assert spec.module
        if not spec.read_only:
            assert spec.confirmation
