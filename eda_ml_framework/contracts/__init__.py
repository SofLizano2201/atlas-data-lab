"""Contratos compartidos entre framework, agentes e interfaz."""

from .models import (
    AgentDecision,
    AgentRequest,
    AgentResponse,
    AnalysisResult,
    DatasetVersion,
    ExperimentResult,
    ModelResult,
    ToolCall,
    ToolResult,
)

__all__ = [
    "AgentDecision",
    "AgentRequest",
    "AgentResponse",
    "AnalysisResult",
    "DatasetVersion",
    "ExperimentResult",
    "ModelResult",
    "ToolCall",
    "ToolResult",
]
