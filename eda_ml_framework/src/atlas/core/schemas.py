"""Shared typed infrastructure and dependency contracts."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Status = Literal["success", "error", "pending_confirmation", "unavailable", "blocked"]


def identifier() -> str:
    """Provide identifier using validated inputs."""
    return uuid4().hex


class Contract(BaseModel):
    """Define the Contract contract."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DatasetVersion(Contract):
    """Define the DatasetVersion contract."""

    dataset_id: str
    version: int = Field(ge=0)
    content_hash: str
    rows: int = Field(ge=1)
    columns: list[str]
    operation: str


class ToolCall(Contract):
    """Define the ToolCall contract."""

    tool: str = Field(min_length=1, max_length=80)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str = Field(default="Solicitud explícita del usuario.", max_length=2000)


class AgentRequest(Contract):
    """Define the AgentRequest contract."""

    agent: str
    operation: str = Field(min_length=1, max_length=8000)
    dataset_id: str
    session_id: str
    calls: list[ToolCall] = Field(default_factory=list, max_length=3)
    consent_metadata: bool = False
    request_id: str = Field(default_factory=identifier)


class ModelResult(Contract):
    """Define the ModelResult contract."""

    model_id: str
    algorithm: str
    problem: str
    version: int = Field(ge=1)
    artifact_id: str


class AnalysisResult(Contract):
    """Define the AnalysisResult contract."""

    summary: str
    dataset: DatasetVersion
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, JsonValue]]] = Field(default_factory=dict)
    visualizations: list[str] = Field(default_factory=list)
    proposal_id: str | None = None
    source_dataset_id: str | None = None
    source_experiment_id: str | None = None


class ExperimentResult(Contract):
    """Define the ExperimentResult contract."""

    experiment_id: str
    dataset_id: str
    model: ModelResult
    metrics: dict[str, float | int | None]
    configuration: dict[str, Any]
    seconds: float
    cohort: str
    visualizations: list[str] = Field(default_factory=list)


class ToolResult(Contract):
    """Define the ToolResult contract."""

    call: ToolCall
    status: Status
    result: AnalysisResult | ExperimentResult | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


class IntelligenceFeedback(Contract):
    """Evidence-based panel content, separate from tool execution responses."""

    agent: str
    event_id: str
    title: str
    message: str
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    candidates: list[dict[str, str]] = Field(default_factory=list)
    priority: str = "information"


class AgentDecision(Contract):
    """Define the AgentDecision contract."""

    agent: str
    action: str
    justification: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class EvidenceRecord(Contract):
    """Reference calculated results rather than numbers inferred from prose."""

    tool: str
    source_id: str | None = None
    dataset_id: str
    content_hash: str
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    tables: list[str] = Field(default_factory=list)


class Recommendation(Contract):
    """An advisory next step which never applies itself."""

    focus: str
    text: str
    basis: str
    mode: Literal["deterministic_guidance", "llm_guidance"]


class AssistantIdea(Contract):
    """A validated contextual invitation backed by a capability or internal navigation."""

    id: str
    title: str
    description: str
    why: str
    expected: str
    effect: str
    goal: str | None = None
    destination: str | None = None


class AssistantOverview(Contract):
    """A small set of evidence-based ideas for one dataset version."""

    welcome: str
    ideas: list[AssistantIdea] = Field(min_length=2, max_length=3)


class AssistantAdvice(Contract):
    """A concise, actionable explanation grounded in current session metadata."""

    title: str
    reason: str
    next_step: str
    goal: str
    action_label: str
    runnable: bool = True
    destination: str | None = None
    destination_label: str | None = None


class AgentResponse(Contract):
    """Define the AgentResponse contract."""

    agent: str
    request_id: str
    operation: str
    dataset_id: str
    status: Status
    message: str
    tools: list[ToolResult] = Field(default_factory=list)
    decisions: list[AgentDecision] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confirmation_id: str | None = None
    provider: str | None = None
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    proposed_actions: list[ToolCall] = Field(default_factory=list)
    executed_actions: list[ToolCall] = Field(default_factory=list)
