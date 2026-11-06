"""Public agent contract and specialized factory."""

from atlas.core.session import AgentContext

from .base import BaseAgent
from .classification_agent import ClassificationAgent
from .clustering_agent import ClusteringAgent
from .comparison_agent import ModelComparisonAgent
from .eda_agent import EDAAgent
from .factory import AgentFactory
from .regression_agent import RegressionAgent

__all__ = [
    "AgentContext",
    "AgentFactory",
    "BaseAgent",
    "ClassificationAgent",
    "ClusteringAgent",
    "EDAAgent",
    "ModelComparisonAgent",
    "RegressionAgent",
]
