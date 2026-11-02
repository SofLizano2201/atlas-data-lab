"""Finite local decision policies run through the same validated permission boundary."""

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from atlas.agents.agent import AgentFactory, NoProvider, configured_provider
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


@pytest.mark.parametrize("specialist", ["eda", "clustering", "classification", "regression"])
def test_offline_diagnosis_without_environment_or_provider(specialist):
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, None, 3.0]})))
    agent = AgentFactory.get_agent(specialist, context, NoProvider())
    response = agent.run(agent.request("guided:diagnose"))
    assert response.status == "success"
    assert response.evidence[0].metrics["celdas_nulas"] == 1
    assert response.provider is None
    assert response.recommendations[0].mode == "deterministic_guidance"


@pytest.mark.parametrize(
    "specialist,goal",
    [("eda", "explore"), ("clustering", "project"), ("classification", "classes")],
)
def test_specialist_decisions_execute_existing_tools(specialist, goal):
    frame = load_iris(as_frame=True).frame
    context = AgentContext(ExploratoryDataAnalysis(frame))
    context.selections = {"analysis_selection": {"target": "target"}}
    agent = AgentFactory.get_agent(specialist, context, NoProvider())
    response = agent.run(agent.request(f"guided:{goal}"))
    if response.status == "pending_confirmation":
        response = agent.confirm(response.confirmation_id)
    assert response.status == "success", response
    assert response.tools[0].call.tool in agent.get_allowed_tools()


@pytest.mark.parametrize("specialist", ["clustering", "classification", "regression"])
def test_local_training_waits_for_authorization(specialist):
    frame = (load_diabetes if specialist == "regression" else load_iris)(as_frame=True).frame
    context = AgentContext(ExploratoryDataAnalysis(frame))
    context.selections = {
        "analysis_selection": {"target": "target", "predictors": list(frame.columns[:-1]), "cv": 2}
    }
    agent = AgentFactory.get_agent(specialist, context, NoProvider())
    response = agent.run(agent.request("guided:train"))
    assert response.status == "pending_confirmation", response
    assert not context.experiments
    executed = agent.confirm(response.confirmation_id)
    assert executed.status == "success", executed
    assert len(context.experiments) == 1


def test_missing_prerequisites_and_untrusted_command_are_blocked():
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1, 2, 3]})))
    agent = AgentFactory.get_agent("regression", context, NoProvider())
    assert agent.run(agent.request("guided:train")).status == "blocked"
    assert agent.run(agent.request("guided:__import__('os')")).status == "blocked"
    assert agent.run(agent.request("guided:evaluate")).status == "blocked"
    assert not context.experiments


def test_local_provider_needs_no_key_and_remote_address_is_rejected():
    provider = configured_provider(
        {"ATLAS_LLM_PROVIDER": "ollama", "ATLAS_LLM_MODEL": "local-test"}
    )
    assert provider.base_url == "http://127.0.0.1:11434/v1"
    assert provider.api_key.get_secret_value() == ""
    with pytest.raises(ValueError):
        configured_provider(
            {
                "ATLAS_LLM_PROVIDER": "ollama",
                "ATLAS_LLM_MODEL": "local-test",
                "ATLAS_LLM_BASE_URL": "https://example.org/v1",
            }
        )


def test_comparison_offline_requires_real_compatible_experiments():
    frame = load_iris(as_frame=True).frame
    context = AgentContext(ExploratoryDataAnalysis(frame))
    context.selections = {"analysis_selection": {"target": "target"}}
    comparison = AgentFactory.get_agent("comparison", context, NoProvider())
    assert comparison.run(comparison.request("guided:compare")).status == "blocked"
    classifier = AgentFactory.get_agent("classification", context, NoProvider())
    for _ in range(2):
        pending = classifier.run(classifier.request("guided:train"))
        assert classifier.confirm(pending.confirmation_id).status == "success"
    result = comparison.run(comparison.request("guided:compare"))
    assert result.status == "success", result
    assert result.evidence
