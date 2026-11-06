"""Coordinate explicit assistance sessions through the polymorphic agent contract."""

import json
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from atlas.core.dependencies import build_agent, build_offline_agent
from atlas.core.schemas import AgentRequest, AgentResponse, ToolCall
from atlas.core.session import AgentContext

if TYPE_CHECKING:
    from atlas.agents.base import BaseAgent


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
        offline: bool = False,
    ) -> None:
        self.context = context
        self.name = name
        self.state = state if state is not None else {}
        if agent is None:
            self._agent, self.configuration_error = (
                build_offline_agent if offline else build_agent
            )(name, context)
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
        if route == "🧩 Clustering":
            selection = dict(state.get("analysis_selection") or {})
            for field, key in {
                "predictors": "analysis_cluster_columns",
                "scaler": "analysis_cluster_scaler",
                "random_state": "analysis_cluster_seed",
            }.items():
                if key in state:
                    selection[field] = state[key]
            context.selections["analysis_selection"] = selection
        problem = {"🎯 Clasificación": "clasificacion", "📈 Regresión": "regresion"}.get(route)
        if problem:
            selection = dict(state.get("analysis_selection") or {})
            for field, suffix in {
                "target": "target",
                "predictors": "predictors",
                "test_size": "size",
                "cv": "cv",
                "random_state": "seed",
                "scaler": "scaler",
            }.items():
                key = f"analysis_{suffix}_{problem}"
                if key in state:
                    selection[field] = state[key]
            context.selections["analysis_selection"] = selection
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
        context.selections["workspace_page"] = {
            "🏠 Inicio": "home",
            "📂 Datos": "data",
            "🔎 Exploración": "exploration",
            "🧹 Preparación": "preparation",
        }.get(route)
        return context

    @property
    def objective(self) -> str:
        return self._agent.objective

    def guided_actions(self) -> dict[str, str]:
        """Forward the specialist's finite local command catalog."""
        return self._agent.guided_actions()

    def advice(self):
        return self._agent.advice()

    def suggestions(self):
        return self._agent.suggestions()

    def _selection_signature(self) -> str:
        return json.dumps(self.context.selections, sort_keys=True, default=str)

    def flow(self) -> dict[str, Any]:
        """Keep presentation continuity scoped to a dataset version and current selections."""
        key = f"assistant_flow_{self.name}"
        flow = self.state.get(key)
        signature = self._selection_signature()
        if flow is None or flow["dataset"] != self.context.key or flow["selection"] != signature:
            for token, pending in list(self.context.pending.items()):
                if pending.request.agent == self.name:
                    self._agent.cancel(token)
            flow = {"dataset": self.context.key, "selection": signature, "phase": "ideas"}
            self.state[key] = flow
            self.state.pop(f"agent_last_{self.name}", None)
        return flow

    def overview(self):
        flow = self.flow()
        stamp = [(item.id, item.cohort) for item in self.context.experiments]
        if "overview" not in flow or flow.get("experiments") != stamp:
            flow["overview"] = self.suggestions()
            flow["experiments"] = stamp
        return flow["overview"]

    def choose(self, identifier: str) -> AgentResponse | None:
        flow = self.flow()
        idea = next((idea for idea in self.overview().ideas if idea.id == identifier), None)
        if idea is None:
            raise ValueError("Esta idea ya no está disponible. Vuelve a revisar las opciones.")
        if flow.get("idea") == idea and flow.get("response") is not None:
            return flow["response"]
        flow.update(idea=idea, phase="explanation")
        if idea.goal:
            response = self.run(self.request(f"guided:{idea.goal}"))
            self.finish(response)
            return response
        return None

    def finish(self, response: AgentResponse) -> None:
        flow = self.state.setdefault(f"assistant_flow_{self.name}", {})
        if "eda" in self.state:
            self.context_for(self.state)
        flow.update(
            dataset=self.context.key,
            selection=self._selection_signature(),
            response=response,
            phase="result",
        )
        self.state[f"agent_last_{self.name}"] = response

    def back(self) -> None:
        flow = self.flow()
        response = flow.get("response")
        if response and response.confirmation_id:
            self.cancel(response.confirmation_id)
        flow.pop("overview", None)
        flow.pop("idea", None)
        flow.pop("response", None)
        flow["phase"] = "ideas"

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
        self.state["active_assistant"] = self.name

    def close(self) -> None:
        self.state[f"agent_open_{self.name}"] = False
        for token, pending in list(self.context.pending.items()):
            if pending.request.agent == self.name:
                self._agent.cancel(token)
        flow = self.state.get(f"assistant_flow_{self.name}", {})
        response = flow.get("response")
        if response and response.confirmation_id:
            flow.pop("response", None)
            flow["phase"] = "explanation"

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
