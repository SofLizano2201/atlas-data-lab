"""Persistent contextual rendering; no agent policies or scientific calculations."""

from html import escape

import pandas as pd
import streamlit as st

from atlas.controllers.agent_controller import AgentController
from atlas.controllers.automatic_eda_controller import AutomaticEDAController
from atlas.controllers.workflow_controller import WorkflowController
from atlas.views.components.agent_panel import render_agent_panel
from atlas.views.components.sidebar import NAV_ITEMS, icon_svg, navigate
from atlas.views.components.tables import data_table
from atlas.views.components.ui import figure

LABELS = {
    "profile": "Perfil del dataset",
    "schema": "Tipos y valores únicos",
    "feature_roles": "Roles de columnas",
    "memory_usage": "Consumo de memoria",
    "missing_values": "Información faltante",
    "type_inconsistencies": "Posibles inconsistencias",
    "descriptive_statistics": "Estadísticas numéricas",
    "percentiles": "Percentiles",
    "outliers_iqr": "Valores poco habituales (IQR)",
    "outliers_zscore": "Valores poco habituales (Z-score)",
    "outliers_isolation": "Valores poco habituales (Isolation Forest)",
    "correlation_pearson": "Relaciones lineales",
    "correlation_spearman": "Relaciones por orden",
    "correlation_kendall": "Concordancia de orden",
    "multicollinearity_vif": "Redundancia numérica",
    "hypothesis_tests": "Pruebas de hipótesis",
    "domain_validation": "Reglas del dominio",
    "chart": "Distribución",
    "frequencies": "Frecuencias",
    "rare_categories": "Categorías poco frecuentes",
    "grouped_statistics": "Estadísticas agrupadas",
    "contingency": "Relación entre categorías",
}


def operation_label(name):
    prefix, _, columns = name.partition(":")
    return LABELS.get(prefix, "Diagnóstico adicional") + (f" · {columns}" if columns else "")


def open_workspace(slug):
    navigate(next(item.route for item in NAV_ITEMS if item.slug == slug))


def render_intelligence_panel():
    with st.container(key="atlas_intelligence_panel"):
        heading, collapse = st.columns([4, 1])
        heading.markdown("### Inteligencia ATLAS")
        if collapse.button(
            "",
            icon=":material/chevron_right:",
            help="Ocultar Inteligencia ATLAS",
            key="intelligence_collapse",
        ):
            st.session_state.intelligence_visible = False
            st.rerun()
        if "eda" not in st.session_state:
            st.caption("Asistencia contextual")
            st.write("Confirma tus datos y prepararé su exploración inicial automáticamente.")
            st.button(
                "Cargar mis datos",
                key="intelligence_load",
                on_click=open_workspace,
                args=("data",),
                type="primary",
            )
            return
        name, feedback = WorkflowController.refresh_feedback(st.session_state)
        context = AgentController.context_for(st.session_state)
        module = {
            "EDAAgent": "Exploración y preparación",
            "ClusteringAgent": "Agrupamiento",
            "ClassificationAgent": "Categorías",
            "RegressionAgent": "Cantidades",
            "ModelComparisonAgent": "Evaluación",
        }[name]
        st.caption(f"Asistencia contextual · {module}")
        st.html(
            f'<div class="atlas-intelligence-context">{escape(st.session_state.get("dataset_name", "Dataset"))} · versión {context.dataset.version}</div>'
        )
        signature = feedback.model_dump_json()
        dismissed = st.session_state.setdefault("dismissed_intelligence", [])
        if signature not in dismissed:
            st.html(
                f'<section class="atlas-intelligence-finding {escape(feedback.priority)}"><h4>{escape(feedback.title)}</h4><p>{escape(feedback.message)}</p></section>'
            )
            if st.button("Ocultar este aviso", key="intelligence_dismiss"):
                st.session_state.dismissed_intelligence = [*dismissed, signature][-30:]
                st.rerun()
        else:
            st.caption("Aviso oculto. Los resultados se conservan.")
        st.markdown("**Qué podemos revisar**")
        agent = AgentController(context, name, state=st.session_state, offline=True)
        for index, idea in enumerate(agent.suggestions().ideas[:3]):
            with st.container(border=True, key=f"intelligence_idea_{index}"):
                st.markdown(f"**{idea.title}**")
                st.caption(idea.description)
                if st.button(
                    "Revisar esta opción",
                    key=f"intelligence_action_{index}",
                    use_container_width=True,
                ):
                    agent.open()
                    response = agent.choose(idea.id)
                    if response:
                        agent.finish(response)
                    st.rerun()
        if name == "EDAAgent":
            with st.container(border=True, key="intelligence_next_step"):
                st.markdown(
                    '<div class="atlas-panel-section">'
                    + icon_svg("search-check")
                    + "<div><span>CONTINÚA POR AQUÍ</span><strong class=atlas-panel-title>Revisa tus resultados</strong></div></div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Abre la exploración para ver qué contienen tus datos y elegir qué quieres investigar."
                )
                st.button(
                    "Explorar mis resultados",
                    key="intelligence_eda",
                    type="primary",
                    on_click=open_workspace,
                    args=("exploration",),
                    use_container_width=True,
                )
        if feedback.candidates:
            with st.container(border=True, key="intelligence_targets"):
                st.markdown(
                    '<div class="atlas-panel-section">'
                    + icon_svg("sliders-horizontal")
                    + "<div><strong class=atlas-panel-title>¿Qué podrías predecir?</strong></div></div>",
                    unsafe_allow_html=True,
                )
                run = st.session_state.get("autonomous_runs", {}).get(
                    st.session_state.get("autonomous_active")
                )
                if run and run.target:
                    st.caption(
                        f"El análisis automático está explorando «{run.target}». "
                        + (
                            "Es una propuesta que puedes cambiar."
                            if run.provisional
                            else "Tú elegiste esta columna."
                        )
                    )
                else:
                    st.caption(
                        "Estas columnas podrían servir como punto de partida. Primero decide qué necesitas averiguar."
                    )
                with st.expander("Ver ideas de predicción"):
                    for candidate in feedback.candidates:
                        st.markdown(f"**{candidate['column']}**")
                        st.write(f"Podría servir para {candidate['objective']}.")
                    st.caption(
                        "Son sugerencias por nombre y tipo de dato; no cambian el objetivo. El agrupamiento busca similitudes y no necesita una columna que predecir."
                    )
        with st.expander("Qué revisé y qué debes tener en cuenta"):
            for evidence in feedback.evidence:
                st.write(evidence)
            for warning in feedback.warnings:
                st.warning(warning)
            st.caption(
                "Orientación local basada en los resultados disponibles. Tus datos no se envían a un servicio de IA."
            )
        recent = activity_groups(st.session_state.get("workflow_events", {}).values())
        with st.container(border=True, key="intelligence_activity"):
            st.markdown(
                '<div class="atlas-panel-section">'
                + icon_svg("file-chart-column")
                + "<div><strong class=atlas-panel-title>Lo que ya se hizo</strong></div></div>",
                unsafe_allow_html=True,
            )
            if recent:
                rows = []
                for item in reversed(recent[-4:]):
                    repetition = (
                        f"Se realizó {item['count']} veces en esta sesión"
                        if item["count"] > 1
                        else "Disponible en esta sesión"
                    )
                    rows.append(
                        f"<li><strong>{escape(item['label'])}</strong><span>{escape(repetition)}</span></li>"
                    )
                st.markdown(
                    '<ol class="atlas-activity-list">' + "".join(rows) + "</ol>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    "Aquí aparecerán las acciones completadas cuando empieces a analizar tus datos."
                )
        if st.button(
            "Ayúdame a decidir el siguiente paso",
            key="intelligence_more_help",
            use_container_width=True,
        ):
            agent.open()
            st.rerun()
        if st.session_state.get(f"agent_open_{name}") and not st.session_state.get(
            "dialog_rendered_this_run"
        ):
            render_agent_panel(name, show_button=False)


def render_automatic_eda_results():
    execution = st.session_state.get("automatic_eda", {}).get(
        AutomaticEDAController.key(st.session_state.eda)
    )
    if execution is None:
        st.info("La exploración inicial todavía no está disponible para esta versión.")
        return
    st.markdown("### Exploración calculada al confirmar tus datos")
    completed = len(
        [item for item in execution.operations.values() if item["status"] == "completed"]
    )
    st.caption(
        f"{completed} operaciones completadas · {execution.seconds:.1f} s. Consulta también las operaciones omitidas o fallidas. No se modificaron datos ni se entrenaron modelos."
    )
    if st.button(
        "Actualizar la exploración inicial",
        key="automatic_eda_refresh",
        help="Repite las operaciones compatibles con los límites configurados; no cambia tus datos.",
    ):
        with st.spinner("Actualizando estadísticas y gráficos compatibles…"):
            AutomaticEDAController.run(st.session_state, refresh=True)
        st.rerun()
    tables, charts, coverage = st.tabs(
        ["Resultados calculados", "Gráficos iniciales", "Qué se analizó y qué no"]
    )
    with tables:
        available = {
            name: item
            for name, item in execution.operations.items()
            if item["status"] == "completed"
            and (
                isinstance(item["value"], (pd.DataFrame, pd.Series)) or name.startswith("outliers")
            )
        }
        if available:
            name = st.selectbox(
                "Resultado de la exploración inicial",
                list(available),
                format_func=operation_label,
                key="automatic_eda_result",
            )
            value = available[name]["value"]
            if isinstance(value, dict) and "rows" in value:
                st.metric("Registros señalados para revisar", value["count"])
                data_table(value["rows"], use_container_width=True, target=st)
            else:
                data_table(value, use_container_width=True, target=st)
    with charts:
        for name, item in execution.operations.items():
            if name.startswith("chart:") and item["status"] == "completed":
                st.caption(
                    "Muestra reproducible solo para el gráfico; las estadísticas usan todos los registros."
                    if item["sampled"]
                    else "Gráfico con todos los registros."
                )
                figure(item["value"], f"auto_{execution.key}_{name}")
    with coverage:
        statuses = {"completed": "Completado", "failed": "Falló", "skipped": "No ejecutado"}
        data_table(
            pd.DataFrame(
                [
                    {
                        "Análisis": operation_label(name),
                        "Estado": statuses[item["status"]],
                        "Alcance": "Muestra para gráfico"
                        if item.get("sampled")
                        else "Dataset completo"
                        if item["status"] == "completed"
                        else "Sin resultado calculado",
                    }
                    for name, item in execution.operations.items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
            target=st,
        )
        st.caption(
            "Las pruebas de hipótesis requieren una pregunta y supuestos definidos. No se ejecutan automáticamente. Los análisis costosos se limitan por tamaño, cantidad de variables y tiempo disponible."
        )
        with st.expander("Motivos exactos y registro de ejecución"):
            data_table(
                pd.DataFrame(
                    [
                        {
                            "Operación": name,
                            "Estado": item["status"],
                            "Motivo": item.get("reason", ""),
                        }
                        for name, item in execution.operations.items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
                target=st,
            )
    if execution.report:
        st.download_button(
            "Descargar exploración inicial (HTML)",
            AutomaticEDAController.report(st.session_state),
            "atlas_eda_inicial.html",
            "text/html",
            key="automatic_eda_report",
        )


def activity_groups(events):
    """Group consecutive repeated display events; retain the original activity history."""
    labels = {
        "EDA_COMPLETED": "Exploración de datos lista",
        "DATA_CLEANED": "Cambios de preparación guardados",
        "TRAINING_COMPLETED": "Modelo de predicción creado",
        "CLUSTERING_COMPLETED": "Grupos encontrados",
        "EXPERIMENT_SAVED": "Análisis guardado",
        "COMPARISON_UPDATED": "Alternativas comparadas",
        "REPORT_GENERATED": "Informe preparado",
    }
    groups = []
    for event in events:
        if event.type not in labels or event.status == "outdated" or event.payload.get("reused"):
            continue
        signature = (
            event.type,
            event.dataset_id,
            event.dataset_version,
            event.payload.get("experiment_id"),
        )
        if groups and groups[-1]["signature"] == signature:
            groups[-1]["count"] += 1
        else:
            groups.append({"signature": signature, "label": labels[event.type], "count": 1})
    return groups


