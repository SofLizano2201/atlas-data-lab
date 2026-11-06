"""Factory for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.config import AgentConfig
from atlas.core.session import AgentContext

from .base import BaseAgent
from .classification_agent import ClassificationAgent
from .clustering_agent import ClusteringAgent
from .comparison_agent import ModelComparisonAgent
from .eda_agent import EDAAgent
from .providers import LLMProvider
from .regression_agent import RegressionAgent

AGENTS = {
    cls.name: cls
    for cls in [
        EDAAgent,
        ClusteringAgent,
        ClassificationAgent,
        RegressionAgent,
        ModelComparisonAgent,
    ]
}


class AgentFactory:
    """Construct any specialist through the same BaseAgent contract."""

    registry = AGENTS

    @classmethod
    def get_agent(
        cls,
        agent_type: str,
        context: AgentContext,
        provider: LLMProvider | None = None,
        *,
        gateway: Any = None,
        logger: Any = None,
        config: AgentConfig | None = None,
    ) -> BaseAgent:
        aliases = {
            "eda": "EDAAgent",
            "clustering": "ClusteringAgent",
            "classification": "ClassificationAgent",
            "regression": "RegressionAgent",
            "comparison": "ModelComparisonAgent",
        }
        name = aliases.get(agent_type, agent_type)
        if name not in cls.registry:
            raise ValueError("Unknown agent type.")
        return cls.registry[name](context, provider, gateway=gateway, logger=logger, config=config)
