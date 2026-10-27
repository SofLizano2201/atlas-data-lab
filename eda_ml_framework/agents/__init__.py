"""Orquestación opcional; las clases académicas no dependen de los agentes."""

from .context import AgentContext
from .specialists import (
    ClassificationAgent,
    ClusteringAgent,
    EDAAgent,
    ModelComparisonAgent,
    RegressionAgent,
)

__all__ = [
    "AgentContext",
    "ClassificationAgent",
    "ClusteringAgent",
    "EDAAgent",
    "ModelComparisonAgent",
    "RegressionAgent",
]
