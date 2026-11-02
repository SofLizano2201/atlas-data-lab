"""Verify dynamic dispatch and a single execution interface for all five specialists."""

from inspect import isabstract

import pandas as pd
import pytest

from atlas.agents.agent import AgentFactory, BaseAgent
from atlas.controllers.capabilities import CapabilityGateway
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("eda", "EDAAgent"),
        ("clustering", "ClusteringAgent"),
        ("classification", "ClassificationAgent"),
        ("regression", "RegressionAgent"),
        ("comparison", "ModelComparisonAgent"),
    ],
)
def test_factory_polymorphism(kind, expected):
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1, 2, 3]})))
    agent = AgentFactory.get_agent(kind, context, gateway=CapabilityGateway(context))
    assert isinstance(agent, BaseAgent) and agent.name == expected
    assert all(
        method in type(agent).__dict__
        for method in ["get_instructions", "get_allowed_tools", "interpret_result"]
    )
    assert agent.get_instructions() and agent.get_allowed_tools()
    assert agent.run(agent.request("Analiza el contexto")).status == "unavailable"


def test_base_is_abstract_and_unknown_factory_type_is_rejected():
    assert isabstract(BaseAgent)
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1, 2, 3]})))
    with pytest.raises(TypeError):
        BaseAgent(context)
    with pytest.raises(ValueError, match="Unknown"):
        AgentFactory.get_agent("execute_python", context)
