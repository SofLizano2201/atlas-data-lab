"""Asistencia contextual: acciones reales, conversación opcional y confirmaciones."""

import json

import pandas as pd
import streamlit as st

from agents import AgentContext
from agents.providers import ProviderError, configured_provider
from agents.specialists import AGENTS
from agents.tools import SPECS
from contracts import AgentResponse, AnalysisResult, ToolCall
from frontend.components.ui import figure

LABELS = {
    "EDAAgent": "EDA",
    "ClusteringAgent": "Clustering",
    "ClassificationAgent": "Clasificación",
    "RegressionAgent": "Regresión",
    "ModelComparisonAgent": "Comparación",
}


def prepare_application(proposal_id: str) -> None:
    """Rellenar la acción permitida; todavía necesita revisión y confirmación."""
    st.session_state["tool_EDAAgent"] = "eda.apply"
    st.session_state["args_EDAAgent_eda.apply"] = json.dumps({"proposal_id": proposal_id})


def session_context() -> AgentContext:
    """Enlazar agentes con el mismo dataset e historial que la interfaz manual."""
    if (
        "agent_context" not in st.session_state
        or st.session_state.agent_context.eda is not st.session_state.eda
    ):
        st.session_state.agent_context = AgentContext(
            st.session_state.eda,
            experiments=st.session_state.experiments,
            engines=st.session_state.models,
        )
    return st.session_state.agent_context


def render_response(response: AgentResponse, context: AgentContext, key: str) -> None:
    """Renderizar contratos; números y figuras provienen de herramientas, no de prosa."""
    st.caption(f"{response.agent} · {response.status} · solicitud {response.request_id[:8]}")
    st.write(response.message)
    for warning in response.warnings:
        st.warning(warning)
    for index, output in enumerate(response.tools):
        with st.expander(f"{output.call.tool} · {output.status}", expanded=True):
            st.caption(output.call.reason)
            st.json(output.call.parameters)
            if output.error:
                st.error(output.error)
            if output.result:
                result = output.result
                if isinstance(result, AnalysisResult):
                    st.caption(
                        f"Contexto al ejecutar: versión {result.dataset.version} · "
                        f"dataset {result.dataset.content_hash[:12]}"
                    )
                    if result.source_experiment_id:
                        st.caption(
                            f"Experimento de origen: {result.source_experiment_id} · "
                            f"dataset entrenado {result.source_dataset_id[:12]}"
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
    """Panel contextual persistente; sin proveedor, conservar herramientas deterministas."""
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
    if button_only or not st.session_state.get(f"agent_open_{name}", False):
        return
    if "eda" not in st.session_state:
        st.info("Carga y confirma un dataset para activar las herramientas de este agente.")
        return
    context = session_context()
    try:
        provider = configured_provider()
    except ProviderError as exc:
        st.warning(str(exc))
        from agents.providers import NoProvider

        provider = NoProvider()
    agent = AGENTS[name](context, provider)
    with st.container(border=True, key=f"panel_{name}"):
        st.subheader(f"Asistente · {LABELS[name]}")
        st.caption(agent.objective)
        st.caption(
            f"Dataset {context.dataset.content_hash[:12]} · versión {context.dataset.version} · "
            f"{context.dataset.rows} filas. Las filas permanecen en este equipo."
        )
        if not provider.available:
            st.info(
                "Asistente conversacional no disponible: no hay proveedor/modelo configurado. "
                "Puedes usar las acciones deterministas siguientes; no son respuestas de un LLM."
            )
        if provider.available:
            st.caption(f"Proveedor configurado: {provider.name}")
        consent = st.checkbox(
            "Autorizar el envío al LLM de mi pregunta e interacciones previas de este agente, nombres/tipos de columnas, "
            "conteos y métricas. No se enviarán filas ni tablas.",
            key=f"consent_{name}",
            disabled=not provider.available,
        )
        tabs = st.tabs(["Acciones rápidas", "Conversación", "Historial"])
        response = None
        with tabs[0]:
            actions = sorted(agent.allowed_tools, key=lambda tool: (tool != "eda.profile", tool))
            tool = st.selectbox(
                "Herramienta permitida",
                actions,
                key=f"tool_{name}",
                format_func=lambda t: f"{t} — {SPECS[t].description}",
            )
            defaults = defaults_for(tool, context, agent.problems)
            # Una clave por herramienta evita arrastrar argumentos de otro tipo de operación.
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
                    "Este plan ajusta modelos, calcula una operación costosa o modifica datos. "
                    "Confirma únicamente después de revisar todos los parámetros."
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
    """Ejemplos construidos desde el contexto; no se ejecutan hasta pulsar el botón."""
    df = context.eda.df
    numeric = list(df.select_dtypes(include="number"))
    columns = numeric[:4] or list(df)[:1]
    ids = [e.id for e in context.experiments if e.problem in problems]
    if tool in {"eda.profile", "eda.correlation"}:
        return {}
    if tool == "eda.statistics":
        return {"columns": columns}
    if tool == "eda.plot":
        return {"kind": "histogram" if numeric else "count", "x": columns[0]}
    if tool == "eda.test":
        return {"test": "shapiro", "columns": columns[:1]}
    if tool == "eda.preview":
        return {"operation": "median", "columns": columns}
    if tool == "eda.apply":
        return {"proposal_id": next(reversed(context.proposals), "")}
    if tool == "cluster.reduce":
        return {"columns": columns, "method": "pca", "dimensions": 2}
    if tool == "cluster.distances":
        return {"columns": columns, "metric": "euclidean"}
    if tool == "cluster.train":
        return {"columns": columns, "algorithm": "kmeans", "parameters": {"n_clusters": 3}}
    if tool.endswith(".train"):
        source = context.eda.original
        target = str(source.columns[-1])
        return {
            "algorithm": "logistic" if tool.startswith("classification") else "ridge",
            "target": target,
            "predictors": [str(c) for c in source if c != target][:12],
        }
    if tool == "comparison.compare":
        eligible = [e for e in context.experiments if e.problem in problems]
        if not eligible:
            return {"experiment_ids": [], "metric": "RMSE", "direction": "min"}
        anchor = eligible[-1]
        compatible = [
            e.id for e in eligible if (e.problem, e.cohort) == (anchor.problem, anchor.cohort)
        ]
        metric = (
            "Accuracy"
            if anchor.problem == "clasificacion"
            else "RMSE"
            if (anchor.problem == "regresion")
            else "Silhouette"
            if "Silhouette" in anchor.metrics
            else "ruido"
        )
        return {
            "experiment_ids": compatible[-2:],
            "metric": metric,
            "direction": "max" if metric in {"Accuracy", "Silhouette"} else "min",
        }
    if tool == "model.plot":
        return {
            "experiment_id": ids[-1] if ids else "",
            "kind": "confusion" if ("clasificacion" in problems) else "actual_predicted",
        }
    return {"experiment_id": ids[-1] if ids else ""}
