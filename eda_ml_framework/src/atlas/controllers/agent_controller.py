"""Coordinate explicit assistance sessions through the polymorphic agent contract."""

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from atlas.core.dependencies import build_agent
from atlas.core.schemas import AgentRequest, AgentResponse, ToolCall
from atlas.core.session import AgentContext

if TYPE_CHECKING:
    from atlas.agents.agent import BaseAgent


class AgentController:
    """Connect a view to one injected specialist without prompts or provider code."""

    @staticmethod
    def default_arguments(context, tool, problems):
        from atlas.core.dependencies import get_default_arguments

        return get_default_arguments(context, tool, problems)

    def __init__(
        self,
        context: AgentContext,
        name: str,
        *,
        state: MutableMapping[str, Any] | None = None,
        agent: "BaseAgent | None" = None,
    ) -> None:
        self.context = context
        self.name = name
        self.state = state if state is not None else {}
        if agent is None:
            self._agent, self.configuration_error = build_agent(name, context)
        else:
            self._agent, self.configuration_error = (agent, None)

    @staticmethod
    def context_for(state: MutableMapping[str, Any]) -> AgentContext:
        if "eda" not in state:
            raise ValueError("Confirma un dataset antes de abrir el asistente.")
        if "agent_context" not in state or state["agent_context"].eda is not state["eda"]:
            state["agent_context"] = AgentContext(
                state["eda"], experiments=state["experiments"], engines=state["models"]
            )
        context = state["agent_context"]
        context.application_state = state
        context.selections = {
            key: state.get(key)
            for key in ["selected_variables", "preprocessing", "problem", "analysis_selection"]
        }
        route = state.get("navigation", "")
        if route in {"🏠 Inicio", "📂 Datos", "🔎 Exploración", "🧹 Preparación"}:
            context.selections = {
                "analysis_selection": state.get("eda_selection", {}),
                "active_module": "eda",
            }
        elif route in {"⚖️ Experimentos", "experiment_history"}:
            context.selections = {
                "analysis_selection": state.get("comparison_selection", {}),
                "active_module": "comparison",
            }
        return context

    @property
    def objective(self) -> str:
        return self._agent.objective

    def guided_actions(self) -> dict[str, str]:
        """Forward the specialist's finite local command catalog."""
        return self._agent.guided_actions()

    def provider_status(self) -> dict[str, Any]:
        return self._agent.provider_status()

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self._agent.get_allowed_tools()

    @property
    def problems(self) -> frozenset[str]:
        return self._agent.problems

    @property
    def provider(self) -> Any:
        return self._agent.provider

    def open(self) -> None:
        self.state[f"agent_open_{self.name}"] = True

    def close(self) -> None:
        self.state[f"agent_open_{self.name}"] = False
        for token, pending in list(self.context.pending.items()):
            if pending.request.agent == self.name:
                self._agent.cancel(token)

    def request(
        self, operation: str, calls: list[ToolCall] | None = None, consent: bool = False
    ) -> AgentRequest:
        return self._agent.request(operation, calls, consent)

    def run(self, request: AgentRequest) -> AgentResponse:
        if not self.state.get(f"agent_open_{self.name}", False):
            return AgentResponse(
                agent=self.name,
                request_id=request.request_id,
                operation=request.operation,
                dataset_id=request.dataset_id,
                status="blocked",
                message="Abre una sesión de asistencia antes de ejecutar acciones.",
            )
        return self._agent.run(request)

    def confirm(self, token: str) -> AgentResponse:
        if not self.state.get(f"agent_open_{self.name}", False):
            return self.run(self.request("Confirmar plan"))
        return self._agent.confirm(token)

    def cancel(self, token: str) -> None:
        self._agent.cancel(token)
