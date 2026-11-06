"""Base for the modular ATLAS assistant system."""

import json
import logging
from abc import ABC, abstractmethod
from time import monotonic
from typing import Any
from uuid import uuid4

from atlas.core.config import AgentConfig
from atlas.core.dependencies import build_gateway
from atlas.core.interfaces import CapabilityAccess
from atlas.core.schemas import (
    AgentDecision,
    AgentRequest,
    AgentResponse,
    ToolCall,
    ToolResult,
)
from atlas.core.session import AgentContext

from .contracts import PendingPlan, Plan
from .events import AgentEventMixin
from .guidance import AssistantGuidanceMixin
from .planning import AgentPlanningMixin
from .policies import SAFETY
from .providers import LLMProvider, NoProvider, ProviderError


class BaseAgent(AgentEventMixin, AssistantGuidanceMixin, AgentPlanningMixin, ABC):
    """Common specialist contract for explicit requests and read-only application events."""

    def read_event_evidence(self, call: ToolCall) -> ToolResult:
        """Reuse read-only evidence; event handling cannot grant training or mutation."""
        if (
            call.tool
            not in {
                "eda.profile",
                "classification.class_distribution",
                "model.metrics",
                "comparison.compare",
            }
            or call.tool not in self.get_allowed_tools()
        ):
            raise ValueError("Event handlers may only read authorized evidence")
        key = json.dumps([self.context.key, call.tool, call.parameters], sort_keys=True)
        if not hasattr(self.context, "event_evidence"):
            self.context.event_evidence = {}
        if key not in self.context.event_evidence:
            self.context.event_evidence[key] = self.gateway.execute(call, self.problems)
        while len(self.context.event_evidence) > 30:
            self.context.event_evidence.pop(next(iter(self.context.event_evidence)))
        return self.context.event_evidence[key]

    name = "BaseAgent"
    objective = ""
    allowed_tools: frozenset[str] = frozenset()
    problems: frozenset[str] = frozenset()

    def __init__(
        self,
        context: AgentContext,
        provider: LLMProvider | None = None,
        *,
        gateway: CapabilityAccess | None = None,
        logger: Any = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.context = context
        self.provider = provider or NoProvider()
        self.gateway = gateway or build_gateway(context)
        self.logger = logger or logging.getLogger("atlas.agents")
        self.config = config or AgentConfig()

    def request(
        self, operation: str, calls: list[ToolCall] | None = None, consent: bool = False
    ) -> AgentRequest:
        """Provide request using validated inputs."""
        return AgentRequest(
            agent=self.name,
            operation=operation,
            calls=calls or [],
            dataset_id=self.context.key,
            session_id=self.context.session_id,
            consent_metadata=consent,
        )

    def _response(
        self, request: AgentRequest, status: str, message: str, **kwargs: Any
    ) -> AgentResponse:
        decision = AgentDecision(agent=self.name, action=status, justification=message)
        tools = kwargs.get("tools", [])
        kwargs.setdefault(
            "executed_actions", [tool.call for tool in tools if tool.status == "success"]
        )
        kwargs.setdefault(
            "proposed_actions",
            [tool.call for tool in tools if tool.status == "pending_confirmation"],
        )
        self.logger.info(
            "Agent action completed",
            extra={
                "agent_name": self.name,
                "execution_status": status,
                "request_id": request.request_id,
            },
        )
        self.context.decisions.append(decision)
        response = AgentResponse(
            agent=self.name,
            request_id=request.request_id,
            operation=request.operation,
            dataset_id=request.dataset_id,
            status=status,
            message=message,
            decisions=[decision],
            **kwargs,
        )
        self.context.conversation.setdefault(self.name, []).append(response)
        self.context.conversation[self.name] = self.context.conversation[self.name][
            -self.config.conversation_limit :
        ]
        self.context.decisions[:] = self.context.decisions[-self.config.decision_limit :]
        return response

    def _validate(self, request: AgentRequest) -> None:
        if request.agent != self.name or request.session_id != self.context.session_id:
            raise ValueError("La solicitud pertenece a otro agente o sesión.")
        if request.dataset_id != self.context.key:
            raise ValueError("El dataset cambió; genera una nueva solicitud.")
        if (
            len(self.context.experiments) + sum(c.tool.endswith(".train") for c in request.calls)
            > 30
        ):
            raise ValueError(
                "Límite de 30 experimentos por sesión; exporte antes de iniciar otra sesión."
            )
        if any(c.tool == "eda.apply" for c in request.calls) and len(request.calls) != 1:
            raise ValueError("Aplicar una transformación requiere un plan independiente.")
        for call in request.calls:
            if call.tool not in self.get_allowed_tools() or call.tool not in self.gateway.tools:
                raise ValueError(f"Herramienta no permitida para {self.name}: {call.tool}.")
            self.gateway.tools[call.tool].arguments.model_validate(call.parameters)

    def run(self, request: AgentRequest) -> AgentResponse:
        """Provide run using validated inputs."""
        try:
            self._validate(request)
            known_goals = self.guided_actions()
            literal_goal = next(
                (goal for goal, label in known_goals.items() if request.operation in {goal, label}),
                None,
            )
            if not request.calls and literal_goal:
                request = request.model_copy(
                    update={"operation": f"guided:{literal_goal}"}, deep=True
                )
            if not request.calls and request.operation.startswith("guided:"):
                request = request.model_copy(
                    update={"calls": self.local_calls(request.operation.removeprefix("guided:"))},
                    deep=True,
                )
                self._validate(request)
            if not request.calls:
                if not self.provider.available:
                    return self._response(
                        request,
                        "unavailable",
                        "Asistente conversacional no disponible. Las acciones rápidas deterministas y el framework siguen funcionando.",
                    )
                if not request.consent_metadata:
                    return self._response(
                        request,
                        "blocked",
                        "Autoriza el envío de pregunta, esquema y conteos al proveedor. No se envían filas del dataset.",
                    )
                metadata = self.context.metadata()
                metadata["experiments"] = [
                    e for e in metadata["experiments"] if e["problem"] in self.problems
                ]
                metadata["recent_actions"] = [
                    {
                        "request": r.operation[:400],
                        "status": r.status,
                        "tools": [t.call.tool for t in r.tools],
                    }
                    for r in self.context.conversation.get(self.name, [])[-6:]
                    if r.provider
                ]
                plan = Plan.model_validate(
                    self.provider.plan(
                        self.get_instructions() + " " + SAFETY,
                        request.operation,
                        metadata,
                        [
                            self.gateway.tools[name].schema()
                            for name in sorted(self.get_allowed_tools())
                        ],
                    )
                )
                calls = [
                    ToolCall(tool=c.tool, parameters=json.loads(c.parameters_json), reason=c.reason)
                    for c in plan.calls
                ]
                request = request.model_copy(update={"calls": calls}, deep=True)
                self._validate(request)
                if not calls:
                    return self._response(
                        request,
                        "success",
                        "Respuesta conversacional sin ejecución de herramientas: "
                        + plan.explanation,
                        provider=self.provider.name,
                    )
            if any(self.requires_confirmation(call) for call in request.calls):
                token = uuid4().hex
                if len(self.context.pending) >= self.config.max_pending_plans:
                    self.context.pending.pop(next(iter(self.context.pending)))
                self.context.pending[token] = PendingPlan(
                    request.model_copy(deep=True), self.context.key, monotonic()
                )
                return self._response(
                    request,
                    "pending_confirmation",
                    "Revisa herramientas y parámetros antes de confirmar. Aún no se ha ejecutado ninguna operación de este plan.",
                    confirmation_id=token,
                    tools=[
                        ToolResult(call=c, status="pending_confirmation") for c in request.calls
                    ],
                )
            return self._execute(request)
        except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
            return self._response(request, "blocked", str(exc))

    def requires_confirmation(self, call: ToolCall) -> bool:
        """Provide requires confirmation using validated inputs."""
        if self.gateway.tools[call.tool].confirmation:
            return True
        if self.context.dataset.rows * len(self.context.dataset.columns) > 1000000:
            return True
        return call.tool == "eda.preview" and call.parameters.get("operation") in {
            "knn",
            "onehot",
            "boxcox",
            "yeojohnson",
        }

    def confirm(self, token: str) -> AgentResponse:
        """Provide confirm using validated inputs."""
        pending = self.context.pending.get(token)
        if not pending or pending.request.agent != self.name:
            request = self.request("Confirmación")
            return self._response(
                request, "blocked", "Confirmación inexistente, consumida o de otro agente."
            )
        self.context.pending.pop(token)
        try:
            if (
                pending.dataset_key != self.context.key
                or monotonic() - pending.created > self.config.confirmation_ttl
            ):
                raise ValueError("La confirmación caducó o cambió el dataset; crea un plan nuevo.")
            self._validate(pending.request)
            return self._execute(pending.request, confirmed=True)
        except (ValueError, TypeError, KeyError) as exc:
            return self._response(pending.request, "blocked", str(exc))

    def cancel(self, token: str) -> None:
        """Provide cancel using validated inputs."""
        pending = self.context.pending.get(token)
        if pending and pending.request.agent == self.name:
            self.context.pending.pop(token)

    def _execute(self, request: AgentRequest, *, confirmed: bool = False) -> AgentResponse:
        from atlas.core.events import EventType, emit

        outputs = []
        state = self.context.application_state
        for call in request.calls:
            try:
                if state is not None and call.tool.endswith(".train"):
                    emit(
                        state,
                        EventType.TRAINING_STARTED,
                        {
                            "request_id": request.request_id,
                            "tool": call.tool,
                            "parameters": call.parameters,
                        },
                    )
                outputs.append(self.gateway.execute(call, self.problems, confirmed=confirmed))
                if state is not None and call.tool == "eda.apply":
                    emit(state, EventType.DATA_CLEANED)
                if state is not None and call.tool == "comparison.compare":
                    emit(state, EventType.COMPARISON_UPDATED, call.parameters)
            except (ValueError, TypeError, KeyError, ImportError, RuntimeError, OSError) as exc:
                outputs.append(ToolResult(call=call, status="error", error=str(exc)))
                if state is not None:
                    emit(
                        state,
                        EventType.ANALYSIS_FAILED,
                        {"request_id": request.request_id, "tool": call.tool, "message": str(exc)},
                    )
                break
        status = "error" if any(t.status == "error" for t in outputs) else "success"
        summaries = [
            o.result.summary
            if hasattr(o.result, "summary")
            else f"Experimento registrado: {o.result.experiment_id}."
            for o in outputs
            if o.result
        ]
        message = " ".join(summaries) or "La operación no pudo completarse."
        warnings = [warning for o in outputs for warning in o.warnings]
        provider_name = None
        if self.provider.available and request.consent_metadata and (status == "success"):
            evidence = {
                "results": [
                    {"tool": o.call.tool, "metrics": o.result.metrics, "status": o.status}
                    for o in outputs
                    if o.result
                ],
                "warnings": warnings,
            }
            try:
                message += "\n\nInterpretación del LLM: " + self.provider.explain(
                    self.get_instructions() + " " + SAFETY, evidence
                )
                provider_name = self.provider.name
            except (ProviderError, ValueError, TypeError, RuntimeError, OSError):
                warnings.append(
                    "La interpretación del LLM falló; se conservan los cálculos reales."
                )
        interpretations = [
            self.interpret_result(output) for output in outputs if output.result is not None
        ]
        warnings.extend(warning for item in interpretations for warning in item.get("warnings", []))
        return self._response(
            request,
            status,
            message,
            tools=outputs,
            warnings=warnings,
            provider=provider_name,
            evidence=[e for item in interpretations for e in item["evidence"]],
            recommendations=[r for item in interpretations for r in item["recommendations"]],
        )

    @abstractmethod
    def get_instructions(self) -> str:
        """Return the specialist's English planning instructions."""
        raise NotImplementedError

    @abstractmethod
    def get_allowed_tools(self) -> frozenset[str]:
        """Return the specialist's explicit capability allowlist."""
        raise NotImplementedError

    @abstractmethod
    def interpret_result(self, result: ToolResult) -> dict[str, Any]:
        """Return evidence references and a specialist recommendation without calculating metrics."""
        raise NotImplementedError

    def _interpret(self, result: ToolResult, focus: str, next_action: str) -> dict[str, Any]:
        if result.result is None:
            return {
                "evidence": [],
                "recommendations": [],
                "warnings": ["No hay resultados calculados que interpretar."],
            }
        record = result.result
        # Session records can predate a development reload of the contract module.
        # Both supported result contracts expose different provenance fields.
        is_analysis = hasattr(record, "dataset")
        next_action = {
            "analysis.configure": "La configuración está preparada en el formulario. Revísala y pulsa Entrenar y evaluar cuando quieras crear el modelo.",
            "eda.preview": "Revisa los resultados propuestos antes de aplicar el cambio. Tus datos todavía no se han modificado.",
            "eda.apply": "El cambio se aplicó al dataset de trabajo. Revisa el resultado antes de seguir; puedes restaurar una versión anterior en Preparación.",
        }.get(result.call.tool, next_action)
        if result.call.tool == "eda.profile":
            next_action = self.advice().next_step
        source_id = (
            record.source_experiment_id
            if hasattr(record, "source_experiment_id")
            else record.experiment_id
        )
        return {
            "evidence": [
                {
                    "tool": result.call.tool,
                    "source_id": source_id,
                    "dataset_id": record.dataset.dataset_id
                    if hasattr(record, "dataset")
                    else record.dataset_id,
                    "metrics": dict(record.metrics),
                    "content_hash": (record.source_dataset_id or record.dataset.content_hash)
                    if is_analysis
                    else record.dataset_id,
                    "tables": list(record.tables) if is_analysis else [],
                }
            ],
            "recommendations": [
                {
                    "focus": focus,
                    "text": next_action,
                    "basis": result.call.tool,
                    "mode": "deterministic_guidance",
                }
            ],
            "warnings": [
                "Orientación determinista sobre resultados registrados; no demuestra causalidad ni estabilidad."
            ],
        }
