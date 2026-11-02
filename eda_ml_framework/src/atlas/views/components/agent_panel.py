"""Spanish Streamlit views: rendering and input collection only."""

import json

import pandas as pd
import streamlit as st

from atlas.controllers.agent_controller import AgentController
from atlas.controllers.capabilities import SPECS
from atlas.core.schemas import AgentResponse, AnalysisResult, ToolCall
from atlas.core.session import AgentContext
from atlas.views.components.ui import figure

LABELS = {
    "EDAAgent": "EDA",
    "ClusteringAgent": "Clustering",
    "ClassificationAgent": "Clasificación",
    "RegressionAgent": "Regresión",
    "ModelComparisonAgent": "Comparación",
}


def prepare_application(proposal_id: str) -> None:
    """Provide prepare application using validated inputs."""
    st.session_state["tool_EDAAgent"] = "eda.apply"
    st.session_state["args_EDAAgent_eda.apply"] = json.dumps({"proposal_id": proposal_id})


def session_context() -> AgentContext:
    """Provide session context using validated inputs."""
    return AgentController.context_for(st.session_state)


def render_response(response: AgentResponse, context: AgentContext, key: str) -> None:
    """Provide render response using validated inputs."""
    st.caption(f"{response.agent} · {response.status} · solicitud {response.request_id[:8]}")
    st.write(response.message)
    if response.evidence:
        with st.expander("Evidencia y referencias de resultados"):
            st.json([record.model_dump(mode="json") for record in response.evidence])
    for recommendation in response.recommendations:
        st.caption("Orientación determinista basada en la herramienta " + recommendation.basis)
        st.info(recommendation.text)
    for warning in response.warnings:
        st.warning(warning)
    for index, output in enumerate(response.tools):
        with st.expander(f"{output.call.tool} · {output.status}", expanded=False):
            st.caption(output.call.reason)
            st.json(output.call.parameters)
            if output.error:
                st.error(output.error)
            if output.result:
                result = output.result
                if isinstance(result, AnalysisResult):
                    st.caption(
                        f"Contexto al ejecutar: versión {result.dataset.version} · dataset {result.dataset.content_hash[:12]}"
                    )
                    if result.source_experiment_id:
                        st.caption(
                            f"Experimento de origen: {result.source_experiment_id} · dataset entrenado {result.source_dataset_id[:12]}"
                        )
                st.dataframe(pd.Series(result.metrics, name="valor").to_frame())
                if isinstance(result, AnalysisResult):
                    for name, records in result.tables.items():
                        st.caption(f"{name} · vista limitada a 200 filas")
                        st.dataframe(pd.DataFrame(records), use_container_width=True)
                    for image_id in result.visualizations:
                        if image_id in context.artifacts:
                            figure(context.artifacts[image_id], f"agent_{key}_{index}_{image_id}")
                    if result.proposal_id:
                        st.code(result.proposal_id, language=None)
                        st.caption(
                            "ID de propuesta para la herramienta eda.apply. Aún no aplicada."
                        )
                        st.button(
                            "Preparar aplicación de esta propuesta",
                            key=f"apply_proposal_{key}_{index}",
                            on_click=prepare_application,
                            args=(result.proposal_id,),
                        )
                else:
                    st.caption(
                        f"Experimento {result.experiment_id} · versión {result.model.version}"
                    )
    st.download_button(
        "Respuesta estructurada JSON",
        response.model_dump_json(indent=2),
        "respuesta_agente.json",
        "application/json",
        key=f"response_{key}",
    )


def render_agent_panel(
    name: str, *, show_button: bool = True, button_only: bool = False, label: str | None = None
) -> None:
    """Provide render agent panel using validated inputs."""
    if show_button and st.button(
        label
        or (
            f"Usar agente {LABELS[name]}"
            if name != "ModelComparisonAgent"
            else "Usar agente de Comparación"
        ),
        key=f"open_{name}",
    ):
        st.session_state[f"agent_open_{name}"] = not st.session_state.get(
            f"agent_open_{name}", False
        )
        if not st.session_state[f"agent_open_{name}"] and "agent_context" in st.session_state:
            AgentController(st.session_state.agent_context, name, state=st.session_state).close()
    if button_only or not st.session_state.get(f"agent_open_{name}", False):
        return
    if "eda" not in st.session_state:
        st.info("Carga y confirma un dataset para activar las herramientas de este agente.")
        return
    context = session_context()
    agent = AgentController(context, name, state=st.session_state)
    provider = agent.provider
    if agent.configuration_error:
        st.warning(agent.configuration_error)
    with st.container(border=True, key=f"panel_{name}"):
        st.subheader(f"Asistente · {LABELS[name]}")
        st.caption(
            f"{st.session_state.get('dataset_name', 'Dataset activo')} · versión {context.dataset.version} · {context.dataset.rows:,} filas"
        )
        with st.expander("Objetivo y contexto del agente"):
            st.caption(agent.objective)
            st.caption(
                f"Dataset {context.dataset.content_hash[:12]}. Las filas permanecen en este equipo."
            )
        if not provider.available:
            st.caption("Motor local activo · sin API · objetivos guiados y evidencia real.")
        if provider.available:
            st.caption(f"Proveedor configurado: {provider.name}")
            if st.button("Comprobar modelo local", key=f"probe_{name}"):
                status = agent.provider_status()
                st.info(status["message"])
        consent = False
        if provider.available:
            consent = st.checkbox(
                "Autorizar el envío al LLM de mi pregunta e interacciones previas de este agente, nombres/tipos de columnas, conteos y métricas. No se enviarán filas ni tablas.",
                key=f"consent_{name}",
                disabled=not provider.available,
            )
        tabs = st.tabs(["Acciones rápidas", "Conversación", "Historial"])
        response = None
        with tabs[0]:
            goals = agent.guided_actions()
            goal = st.selectbox(
                "¿Qué necesitas revisar?", list(goals), format_func=goals.get, key=f"goal_{name}"
            )
            primary, secondary = st.columns([2, 1])
            if primary.button("Analizar con motor local", key=f"guided_{name}", type="primary"):
                with st.spinner("Evaluando requisitos y evidencia local…"):
                    response = agent.run(agent.request(f"guided:{goal}"))
            if secondary.button("Descartar recomendación", key=f"dismiss_{name}"):
                st.session_state[f"agent_last_{name}"] = None
            st.caption(
                "Se ejecuta una herramienta autorizada según el objetivo elegido. Entrenar o modificar configuraciones requiere confirmación."
            )
        with tabs[0], st.expander("Herramientas y parámetros avanzados"):
            actions = sorted(agent.allowed_tools, key=lambda tool: (tool != "eda.profile", tool))
            tool = st.selectbox(
                "Herramienta permitida",
                actions,
                key=f"tool_{name}",
                format_func=lambda t: f"{t} — {SPECS[t].description}",
            )
            defaults = defaults_for(tool, context, agent.problems)
            params = st.text_area(
                "Parámetros JSON (datos, nunca código)",
                json.dumps(defaults, ensure_ascii=False, indent=2),
                key=f"args_{name}_{tool}",
            )
            with st.expander("Esquema de parámetros admitidos"):
                st.json(SPECS[tool].arguments.model_json_schema())
            if st.button("Ejecutar acción determinista", key=f"quick_{name}"):
                try:
                    call = ToolCall(tool=tool, parameters=json.loads(params))
                    with st.spinner("Validando y ejecutando herramienta…"):
                        response = agent.run(
                            agent.request(SPECS[tool].description, [call], consent)
                        )
                except (ValueError, TypeError) as exc:
                    st.error(f"Parámetros inválidos: {exc}")
        with tabs[1]:
            question = st.text_area(
                "Pregunta para el agente",
                key=f"question_{name}",
                disabled=not provider.available,
                placeholder="¿Qué variables debo revisar y por qué?",
            )
            if st.button("Enviar al agente", key=f"ask_{name}", disabled=not provider.available):
                if question.strip():
                    with st.spinner("Consultando al proveedor configurado…"):
                        response = agent.run(agent.request(question, consent=consent))
                else:
                    st.warning("Escribe una pregunta antes de enviar.")
        with tabs[2]:
            history = context.conversation.get(name, [])
            if history:
                selected = st.selectbox(
                    "Conversaciones de esta sesión",
                    history,
                    format_func=lambda r: f"{r.operation[:60]} · {r.status}",
                    key=f"history_{name}",
                )
                render_response(selected, context, f"history_{name}_{selected.request_id}")
            else:
                st.caption("Aún no hay acciones de este agente.")
        if response is not None:
            st.session_state[f"agent_last_{name}"] = response
            st.session_state.work = context.eda.df
            for experiment in context.experiments:
                st.session_state.metrics[experiment.id] = experiment.metrics
        latest = st.session_state.get(f"agent_last_{name}")
        if latest:
            render_response(latest, context, f"latest_{name}_{latest.request_id}")
            token = latest.confirmation_id
            if token and token in context.pending:
                st.warning(
                    "Este plan ajusta modelos, calcula una operación costosa o modifica datos. Confirma únicamente después de revisar todos los parámetros."
                )
                a, b = st.columns(2)
                if a.button("Confirmar y ejecutar plan", key=f"confirm_{name}", type="primary"):
                    with st.spinner("Ejecutando el plan confirmado…"):
                        st.session_state[f"agent_last_{name}"] = agent.confirm(token)
                    st.session_state.work = context.eda.df
                    st.session_state.graphs = {}
                    st.rerun()
                if b.button("Cancelar plan", key=f"cancel_{name}"):
                    agent.cancel(token)
                    st.session_state[f"agent_last_{name}"] = None
                    st.rerun()


def defaults_for(tool: str, context: AgentContext, problems: set[str]) -> dict:
    return AgentController.default_arguments(context, tool, problems)
