"""Legacy import facade. Implementations live in focused modules in this package."""

from atlas.core.session import AgentContext

from .base import BaseAgent
from .classification_agent import ClassificationAgent
from .clustering_agent import ClusteringAgent
from .comparison_agent import ModelComparisonAgent
from .contracts import Explanation, PendingPlan, Plan, PlannedCall
from .defaults import default_arguments
from .eda_agent import EDAAgent
from .factory import AGENTS, AgentFactory
from .policies import (
    COMMON_POLICIES,
    DOMAIN_IDEA_COPY,
    IDEA_COPY,
    SAFETY,
    SPECIALIST_POLICIES,
    DecisionPolicy,
)
from .providers import (
    JSONLLMProvider,
    LLMProvider,
    NoProvider,
    NoRedirect,
    ProviderError,
    configured_provider,
)
from .regression_agent import RegressionAgent

__all__ = [
    "AGENTS",
    "COMMON_POLICIES",
    "DOMAIN_IDEA_COPY",
    "IDEA_COPY",
    "SAFETY",
    "SPECIALIST_POLICIES",
    "AgentContext",
    "AgentFactory",
    "BaseAgent",
    "ClassificationAgent",
    "ClusteringAgent",
    "DecisionPolicy",
    "EDAAgent",
    "Explanation",
    "JSONLLMProvider",
    "LLMProvider",
    "ModelComparisonAgent",
    "NoProvider",
    "NoRedirect",
    "PendingPlan",
    "Plan",
    "PlannedCall",
    "ProviderError",
    "RegressionAgent",
    "configured_provider",
    "default_arguments",
]
