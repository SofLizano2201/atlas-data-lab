"""Resultados tipados; el frontend nunca extrae métricas de texto del LLM."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Status = Literal["success", "error", "pending_confirmation", "unavailable", "blocked"]


def identifier() -> str:
    """Identificador opaco para solicitudes, planes y artefactos."""
    return uuid4().hex


class Contract(BaseModel):
    """Rechazar campos inesperados y coerciones ambiguas en fronteras de confianza."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DatasetVersion(Contract):
    """Identidad de contenido y metadatos de una instantánea."""

    dataset_id: str
    version: int = Field(ge=0)
    content_hash: str
    rows: int = Field(ge=1)
    columns: list[str]
    operation: str


class ToolCall(Contract):
    """Nombre del registro y argumentos validados antes de ejecutar."""

    tool: str = Field(min_length=1, max_length=80)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str = Field(default="Solicitud explícita del usuario.", max_length=2000)


class AgentRequest(Contract):
    """La autorización se valida con un ticket local; el modelo no puede concederla."""

    agent: str
    operation: str = Field(min_length=1, max_length=8000)
    dataset_id: str
    session_id: str
    calls: list[ToolCall] = Field(default_factory=list, max_length=3)
    consent_metadata: bool = False
    request_id: str = Field(default_factory=identifier)


class ModelResult(Contract):
    """Descriptor de un modelo; el objeto entrenado permanece en memoria local."""

    model_id: str
    algorithm: str
    problem: str
    version: int = Field(ge=1)
    artifact_id: str


class AnalysisResult(Contract):
    """Cálculos y tablas serializables; figuras mediante referencias locales."""

    summary: str
    dataset: DatasetVersion
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, JsonValue]]] = Field(default_factory=dict)
    visualizations: list[str] = Field(default_factory=list)
    proposal_id: str | None = None
    source_dataset_id: str | None = None
    source_experiment_id: str | None = None


class ExperimentResult(Contract):
    """Resultado supervisado o de clustering con procedencia completa."""

    experiment_id: str
    dataset_id: str
    model: ModelResult
    metrics: dict[str, float | int | None]
    configuration: dict[str, Any]
    seconds: float
    cohort: str
    visualizations: list[str] = Field(default_factory=list)


class ToolResult(Contract):
    """Traza exacta de lo ejecutado y su estado."""

    call: ToolCall
    status: Status
    result: AnalysisResult | ExperimentResult | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


class AgentDecision(Contract):
    """Decisión explicable y auditable sin almacenar razonamiento interno del modelo."""

    agent: str
    action: str
    justification: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class AgentResponse(Contract):
    """Respuesta estructurada, ligada a solicitud, sesión y versión de datos."""

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
