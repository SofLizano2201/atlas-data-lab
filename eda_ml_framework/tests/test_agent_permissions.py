"""Reject unauthorized capabilities and stale confirmations without changing data."""

from time import monotonic

import pandas as pd
import pytest

from atlas.agents.agent import AgentFactory
from atlas.controllers.capabilities import CapabilityGateway
from atlas.core.config import AgentConfig
from atlas.core.schemas import ToolCall
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


@pytest.mark.parametrize("tool", ["os.system", "__getattribute__", "regression.train"])
def test_eda_cannot_request_unapproved_methods(tool):
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, 2.0, 3.0]})))
    agent = AgentFactory.get_agent("eda", context)
    response = agent.run(agent.request("Ejecuta", [ToolCall(tool=tool)]))
    assert response.status == "blocked"
    assert not context.experiments
    assert context.eda.dimensions() == (3, 1)


def test_injected_confirmation_policy_is_used():
    context = AgentContext(ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, None, 3.0]})))
    agent = AgentFactory.get_agent("eda", context, config=AgentConfig(confirmation_ttl=0))
    proposal = agent.run(
        agent.request(
            "Propón",
            [ToolCall(tool="eda.preview", parameters={"operation": "median", "columns": ["x"]})],
        )
    )
    identifier = proposal.tools[0].result.proposal_id
    pending = agent.run(
        agent.request(
            "Aplica", [ToolCall(tool="eda.apply", parameters={"proposal_id": identifier})]
        )
    )
    context.pending[pending.confirmation_id].created = monotonic() - 1
    assert agent.confirm(pending.confirmation_id).status == "blocked"
    assert context.eda.profile()["missing_cells"] == 1


def test_gateway_also_guards_expensive_calls_without_a_confirmation():
    context = AgentContext(
        ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]}))
    )
    with pytest.raises(ValueError, match="confirmación"):
        CapabilityGateway(context).execute(
            ToolCall(
                tool="cluster.train", parameters={"algorithm": "kmeans", "columns": ["x", "y"]}
            ),
            {"clustering"},
        )
    assert not context.experiments
