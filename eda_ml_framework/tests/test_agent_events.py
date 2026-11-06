"""Application events update feedback, without grants to train or modify data."""

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from atlas.agents.agent import AgentFactory, NoProvider
from atlas.controllers.agent_controller import AgentController
from atlas.controllers.automatic_eda_controller import EDALimits
from atlas.controllers.workflow_controller import WorkflowController
from atlas.core.events import EventType, emit
from atlas.core.schemas import ToolCall
from atlas.services.dataset_service import DatasetService


def state_for(frame=None):
    state = {"eda_limits": EDALimits(max_charts=0), "navigation": "📂 Datos"}
    DatasetService.activate(
        state, frame if frame is not None else load_iris(as_frame=True).frame, "example.csv"
    )
    return state


def test_analysis_provenance_survives_contract_reload():
    from types import SimpleNamespace

    from atlas.core.schemas import ToolResult

    state = state_for()
    context = AgentController.context_for(state)
    record = SimpleNamespace(
        dataset=context.dataset,
        source_dataset_id=None,
        source_experiment_id=None,
        metrics={},
        tables={},
    )
    result = ToolResult.model_construct(
        call=ToolCall(tool="eda.profile", parameters={}, reason="Review existing evidence"),
        status="completed",
        result=record,
    )
    agent = AgentFactory.get_agent("eda", context, NoProvider())
    evidence = agent.interpret_result(result)["evidence"][0]
    assert evidence["dataset_id"] == context.dataset.dataset_id
    assert evidence["content_hash"] == context.dataset.content_hash


def test_event_ids_and_feedback_are_stable_on_rerun():
    state = state_for()
    first = emit(state, EventType.DATASET_CONFIRMED, {"name": "example.csv"})
    assert emit(state, EventType.DATASET_CONFIRMED, {"name": "example.csv"}) is first
    WorkflowController.synchronize(state)
    count = len(state["workflow_events"])
    feedback = state["intelligence_feedback"]["EDAAgent"]
    WorkflowController.synchronize(state)
    assert len(state["workflow_events"]) == count
    assert state["intelligence_feedback"]["EDAAgent"] == feedback
    assert not state["experiments"] and not state["agent_context"].pending


def test_five_specialists_receive_events_without_training():
    state = state_for()
    context = AgentController.context_for(state)
    context.selections = {
        "analysis_selection": {"target": "target", "predictors": list(state["eda"].df.columns[:-1])}
    }
    event = emit(state, EventType.TARGET_SELECTED, {"target": "target"})
    for kind in ["eda", "clustering", "classification", "regression", "comparison"]:
        feedback = AgentFactory.get_agent(kind, context, NoProvider()).on_event(event)
        assert feedback.title and feedback.message and feedback.evidence
    assert not context.experiments and not context.pending
    assert context.eda.version_count == 1


def test_target_suggestions_do_not_set_a_definitive_target():
    state = state_for(pd.DataFrame({"id": range(30), "price": range(30), "group": ["a", "b"] * 15}))
    WorkflowController.synchronize(state)
    feedback = state["intelligence_feedback"]["EDAAgent"]
    assert any(candidate["column"] == "price" for candidate in feedback.candidates)
    assert all(candidate["column"] != "id" for candidate in feedback.candidates)
    assert not state.get("analysis_selection", {}).get("target")


def test_navigation_and_returning_to_previous_selection_update_context():
    state = state_for()
    WorkflowController.synchronize(state)
    state["navigation"] = "🎯 Clasificación"
    state["analysis_target_clasificacion"] = "target"
    state["analysis_predictors_clasificacion"] = ["sepal length (cm)"]
    WorkflowController.synchronize(state)
    assert "ClassificationAgent" in state["intelligence_feedback"]
    count = len(state["workflow_events"])
    state["analysis_predictors_clasificacion"] = ["sepal width (cm)"]
    WorkflowController.synchronize(state)
    state["analysis_predictors_clasificacion"] = ["sepal length (cm)"]
    WorkflowController.synchronize(state)
    assert len(state["workflow_events"]) > count


def test_events_cache_evidence_and_cannot_train(monkeypatch):
    state = state_for()
    context = AgentController.context_for(state)
    context.selections = {"analysis_selection": {"target": "target"}}
    agent = AgentFactory.get_agent("classification", context, NoProvider())
    calls = []
    execute = agent.gateway.execute

    def counted(*args, **kwargs):
        calls.append(True)
        return execute(*args, **kwargs)

    monkeypatch.setattr(agent.gateway, "execute", counted)
    first = emit(state, EventType.TARGET_SELECTED, {"target": "target"})
    second = emit(state, EventType.FEATURES_CHANGED, {"features": ["sepal length (cm)"]})
    agent.on_event(first)
    agent.on_event(second)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="only read"):
        agent.read_event_evidence(ToolCall(tool="classification.train", parameters={}))
    assert not state["experiments"]
