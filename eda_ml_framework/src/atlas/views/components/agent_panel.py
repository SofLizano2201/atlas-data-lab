"""One contextual assistant dialog shared by all five specialists."""

import json
from html import escape

import pandas as pd
import streamlit as st

from atlas.controllers.agent_controller import AgentController
from atlas.controllers.capabilities import SPECS
from atlas.core.schemas import AgentResponse, AnalysisResult, ExperimentResult
from atlas.views.components.sidebar import NAV_ITEMS, icon_mask, navigate
from atlas.views.components.tables import data_table
from atlas.views.components.ui import figure, format_number

ASSISTANTS = {
    "EDAAgent": (
        "Asistente de datos",
        "Te ayudo a comprender y preparar tus datos.",
        "search-check",
    ),
    "ClusteringAgent": (
        "Asistente de agrupamiento",
        "Te ayudo a encontrar patrones y entender tus grupos.",
        "network",
    ),
    "ClassificationAgent": (
        "Asistente de clasificación",
        "Te ayudo a predecir categorías y entender los errores.",
        "tags",
    ),
    "RegressionAgent": (
        "Asistente de predicciones",
        "Te ayudo a revisar predicciones de cantidades.",
        "chart-no-axes-combined",
    ),
    "ModelComparisonAgent": (
        "Asistente de comparación",
        "Te ayudo a contrastar alternativas de forma justa.",
        "flask-conical",
    ),
}
METRICS = {
    "outlier_rows": (
        "Registros con valores poco habituales",
        "Casos señalados para revisar; no se han eliminado.",
        ",.0f",
    ),
    "filas": ("Registros revisados", "Casos disponibles en estos datos.", ",.0f"),
    "celdas_nulas": ("Datos incompletos", "Celdas que no contienen información.", ",.0f"),
    "duplicadas_adicionales": (
        "Filas repetidas adicionales",
        "Revísalas antes de decidir si eliminarlas.",
        ",.0f",
    ),
    "Accuracy": (
        "Aciertos globales",
        "Proporción de respuestas correctas en los datos de prueba.",
        ".1%",
    ),
    "F1_macro": (
        "Equilibrio entre categorías",
        "De 0 a 1: un valor mayor indica mejor equilibrio entre detección y aciertos por categoría.",
        ".3f",
    ),
    "MAE": (
        "Error medio",
        "Cuánto se equivoca de media, en las unidades de lo que predices.",
        ",.2f",
    ),
    "RMSE": (
        "Impacto de fallos grandes",
        "Se expresa en las unidades de lo que predices; un valor menor es mejor.",
        ",.2f",
    ),
    "ruido": ("Casos sin grupo", "Proporción de registros que no se asignaron a un grupo.", ".1%"),
    "clusters": (
        "Grupos encontrados",
        "Cantidad de grupos, sin contar los casos sin grupo.",
        ".0f",
    ),
}


def readable_number(value, formatting=",.2f"):
    """Format a stored value using Spanish separators without scientific notation."""
    return format_number(value, formatting)


def session_context():
    return AgentController.context_for(st.session_state)


def defaults_for(tool, context, problems):
    return AgentController.default_arguments(context, tool, problems)


def _close_dialog():
    name = st.session_state.get("active_assistant")
    if name:
        if "agent_context" in st.session_state:
            AgentController(st.session_state.agent_context, name, state=st.session_state).close()
        st.session_state[f"agent_open_{name}"] = False


def _navigate(destination):
    target = next(item for item in NAV_ITEMS if item.slug == destination)
    st.session_state["assistant_navigation_notice"] = (
        f"Abrí {target.label}. Aquí puedes continuar con la sugerencia; no se ejecutó ningún cálculo ni se modificaron tus datos al cambiar de pantalla."
    )
    _close_dialog()
    navigate(target.route)


def _store(agent, response):
    agent.finish(response)
    st.session_state.work = agent.context.eda.df
    for experiment in agent.context.experiments:
        st.session_state.metrics[experiment.id] = experiment.metrics


def _execution_plan(context, response):
    """Present the actual validated pending calls, without making analytical decisions."""
    pending = context.pending.get(response.confirmation_id)
    if not pending:
        st.warning("Esta propuesta ya no está disponible. Vuelve a elegir una idea.")
        return
    with st.container(border=True):
        st.markdown("**Antes de continuar**")
        st.caption("Esto es lo que voy a hacer si das tu autorización:")
        for index, call in enumerate(pending.request.calls, 1):
            label = {
                "analysis.configure": "Actualizar las opciones del formulario. No entrenaré ningún modelo.",
                "cluster.train": "Calcular y guardar un agrupamiento.",
                "classification.train": "Entrenar y evaluar un modelo de categorías.",
                "regression.train": "Entrenar y evaluar un modelo de predicción numérica.",
                "eda.apply": "Aplicar la limpieza revisada y crear una versión restaurable de tus datos.",
                "eda.preview": "Calcular una vista previa sin cambiar los datos activos.",
                "cluster.elbow": "Calcular cómo cambia el agrupamiento al variar la cantidad de grupos.",
            }.get(call.tool, SPECS[call.tool].description)
            st.write(f"{index}. {label}")
            parameters = call.parameters
            columns = parameters.get("columns") or parameters.get("predictors")
            if columns:
                st.caption("Variables: " + ", ".join(map(str, columns)))
            if parameters.get("target"):
                st.caption(f"Qué se predice: {parameters['target']}")
            count = parameters.get("parameters", {}).get("n_clusters")
            if count:
                st.caption(f"Cantidad de grupos: {count}")
        st.caption("Te mostraré el resultado aquí al terminar. Puedes cancelar antes de empezar.")


def _execute_visible(agent, action):
    """Show real execution state, then retain its response through the next rerun."""
    with st.status("Trabajando en tu solicitud…", expanded=True) as progress:
        st.write("Ejecutando las operaciones autorizadas. No cierres este panel mientras termina.")
        try:
            response = action()
            _store(agent, response)
        except (ValueError, TypeError, KeyError, RuntimeError):
            progress.update(label="No se pudo completar la solicitud", state="error")
            st.error(
                "Ocurrió un problema al ejecutar la acción. Revisa la selección e inténtalo de nuevo; no se presenta un resultado como completado."
            )
            return False
        progress.update(
            label="Solicitud procesada",
            state="complete" if response.status == "success" else "error",
            expanded=True,
        )
    return True


def render_response(response: AgentResponse, context, key: str):
    """Render only everyday language and real result data in the normal experience."""
    if response.status in {"error", "blocked", "unavailable"}:
        st.error("No se pudo completar esta opción.")
        reasons = [output.error for output in response.tools if output.error]
        if not reasons:
            reasons = [response.message]
        for reason in dict.fromkeys(reasons):
            st.write(reason)
        st.info(
            "Revisa las variables seleccionadas y los requisitos indicados arriba. Después vuelve a elegir la idea. Tus resultados anteriores siguen disponibles."
        )
        return
    if response.status == "pending_confirmation":
        st.info(
            "Para obtener este resultado necesito realizar la acción descrita arriba. ¿Quieres continuar? Todavía no ha comenzado."
        )
        return
    completed = [output for output in response.tools if output.status == "success"]
    if completed:
        st.success(
            f"Listo: se completaron {len(completed)} de {len(response.tools)} acciones. Los resultados se muestran abajo."
        )
        for output in completed:
            tool = output.call.tool
            if tool == "analysis.configure":
                st.info(
                    "Actualicé las opciones del formulario. Cierra el asistente para revisarlas y pulsa Entrenar cuando quieras generar un modelo. Todavía no se entrenó ninguno."
                )
            elif tool.endswith(".train"):
                st.caption(
                    "El resultado quedó guardado en Experimentos y puede revisarse sin volver a entrenar."
                )
            elif tool == "eda.apply":
                st.info(
                    "El cambio quedó aplicado. La vista principal usa la nueva versión; puedes restaurar la anterior desde Preparación."
                )
        if len(completed) < len(response.tools):
            st.warning(
                "Algunas acciones no se completaron. Revisa los detalles antes de utilizar los resultados."
            )
    for index, output in enumerate(response.tools):
        if not output.result:
            continue
        result = output.result
        st.markdown("### Resultados de la revisión")
        st.caption(
            {
                "regression.train": "Comprobamos cuánto se acercan las predicciones a los valores reales usando datos reservados para la prueba.",
                "classification.train": "Comprobamos qué categorías reconoce y cuáles confunde en datos reservados para la prueba.",
                "model.metrics": "Estos son los resultados del modelo guardado; no se volvió a entrenar.",
            }.get(output.call.tool, SPECS[output.call.tool].description)
        )
        readable = [
            (name, value)
            for name, value in result.metrics.items()
            if name in METRICS and value is not None
        ]
        for start in range(0, min(len(readable), 4), 2):
            for column, (name, value) in zip(st.columns(2), readable[start : start + 2]):
                label, help_text, formatting = METRICS[name]
                column.metric(label, readable_number(value, formatting), help=help_text)
                column.caption(help_text)
        if result.metrics.get("MAE") is not None:
            configuration = result.configuration if isinstance(result, ExperimentResult) else {}
            saved = next(
                (
                    item
                    for item in context.experiments
                    if item.id == getattr(result, "source_experiment_id", None)
                ),
                None,
            )
            if saved is not None:
                configuration = saved.config
            target = configuration.get("target")
            target_text = f"«{target}»" if target else "la cantidad elegida"
            st.html(
                '<section class="atlas-assistant-recommendation"><span class="atlas-assistant-eyebrow">CÓMO LEER ESTE RESULTADO</span>'
                f"<h4>¿Cuánto se equivoca al predecir?</h4><p>Al predecir {escape(target_text)}, la diferencia absoluta media con el valor real fue de <strong>{readable_number(result.metrics['MAE'])} unidades</strong> en los datos de prueba. No significa que todos los casos tengan ese error ni indica si predice por encima o por debajo.</p></section>"
            )
            st.caption(
                "El indicador de fallos grandes da más peso a las equivocaciones importantes. En ambos indicadores, un valor menor es preferible al comparar modelos evaluados con los mismos datos y la misma variable. No se puede decidir si el error es aceptable sin conocer tu objetivo y las unidades."
            )
        additional = [
            {"Indicador": name, "Valor": value}
            for name, value in result.metrics.items()
            if name not in METRICS
        ]
        if additional:
            with st.expander("Ver todas las métricas (avanzado)"):
                data_table(
                    pd.DataFrame(additional), hide_index=True, use_container_width=True, target=st
                )
        if isinstance(result, ExperimentResult):
            st.caption(
                f"Modelo: {result.model.algorithm} · Tiempo: {result.seconds:.2f} s · Guardado en Experimentos"
            )
            experiment = next(
                (item for item in context.experiments if item.id == result.experiment_id), None
            )
            if experiment is not None:
                st.markdown("**Muestra de los resultados guardados**")
                data_table(experiment.predictions.head(50), use_container_width=True, target=st)
        if isinstance(result, AnalysisResult):
            titles = {
                "outliers": "Registros con valores poco habituales",
                "informacion": "Información de las columnas",
                "faltantes": "Datos faltantes por columna",
                "estadisticas": "Estadísticas calculadas",
                "correlacion": "Relaciones entre variables",
                "elbow": "Resultados por cantidad de grupos",
                "comparacion": "Comparación de alternativas",
                "seleccion": "Alternativa destacada",
                "datos": "Datos procesados",
            }
            for table_name, records in result.tables.items():
                if table_name == "frecuencias" or (
                    result.proposal_id and table_name in {"antes", "despues"}
                ):
                    continue
                st.markdown(
                    f"**{titles.get(table_name, table_name.replace('_', ' ').capitalize())}**"
                )
                if records:
                    frame = pd.DataFrame(records).rename(
                        columns={"indice_fila": "Registro / variable"}
                    )
                    data_table(frame, hide_index=True, use_container_width=True, target=st)
                    st.download_button(
                        "Descargar esta tabla",
                        frame.to_csv(index=False).encode("utf-8-sig"),
                        f"{table_name}.csv",
                        "text/csv",
                        key=f"table_{key}_{index}_{table_name}",
                    )
                    st.caption(
                        f"Muestra disponible: {len(frame):,} registros. Las tablas del asistente se limitan a 200 filas."
                    )
                else:
                    st.info(
                        "No se encontraron registros para esta revisión. No se eliminaron datos."
                    )
            if result.proposal_id:
                st.markdown("**Revisa el cambio propuesto**")
                before, after = st.columns(2)
                for column, label, table in [
                    (before, "Antes", "antes"),
                    (after, "Después", "despues"),
                ]:
                    column.caption(label)
                    data_table(
                        pd.DataFrame(result.tables.get(table, [])).head(10),
                        use_container_width=True,
                        target=column,
                    )
                if st.button(
                    "Aplicar este cambio", key=f"apply_preview_{key}_{index}", type="primary"
                ):
                    from atlas.core.schemas import ToolCall

                    agent = AgentController(context, response.agent, state=st.session_state)
                    _store(
                        agent,
                        agent.run(
                            agent.request(
                                "Aplicar cambio revisado",
                                [
                                    ToolCall(
                                        tool="eda.apply",
                                        parameters={"proposal_id": result.proposal_id},
                                    )
                                ],
                            )
                        ),
                    )
                    st.rerun()
            if result.tables.get("frecuencias"):
                st.caption("Ejemplos disponibles por categoría")
                data_table(
                    pd.DataFrame(result.tables["frecuencias"])
                    .head(20)
                    .rename(
                        columns={
                            "indice_fila": "Categoría",
                            "absoluta": "Cantidad",
                            "relativa": "Proporción",
                        }
                    ),
                    use_container_width=True,
                    target=st,
                )
        for image_id in result.visualizations:
            if image_id in context.artifacts:
                figure(context.artifacts[image_id], f"assistant_{key}_{index}_{image_id}")
    if response.recommendations:
        st.html(
            '<section class="atlas-assistant-recommendation">'
            '<span class="atlas-assistant-eyebrow">TU SIGUIENTE PASO</span>'
            "<h4>Lo que te recomiendo ahora</h4>"
            f"<p>{escape(response.recommendations[0].text)}</p></section>"
        )


def _technical_controls(agent, name):
    """Expert functionality exists only after an explicit request for technical details."""
    from atlas.core.schemas import ToolCall

    flow = agent.flow()
    with st.container(border=True, key=f"technical_content_{name}"):
        st.markdown("#### Detalles de tu análisis")
        st.caption(
            "Consulta qué se revisó y los resultados. Las acciones avanzadas son opcionales."
        )
        summary, history, tools, chat = st.tabs(
            ["Resumen", "Historial", "Acciones avanzadas", "Modelo local"]
        )
        with summary:
            response = flow.get("response")
            if not response:
                st.info("Elige primero una idea para ver aquí cómo se realizó la revisión.")
            else:
                status = {
                    "success": "Revisión completada",
                    "pending_confirmation": "Espera tu autorización",
                    "error": "No se pudo completar",
                    "blocked": "Revisa los requisitos",
                    "unavailable": "Opción no disponible",
                }.get(response.status, "Revisión registrada")
                st.markdown(f"**{status}**")
                for output in response.tools:
                    st.markdown(
                        f"**{SPECS[output.call.tool].description if output.call.tool in SPECS else 'Acción registrada'}**"
                    )
                    if output.result and output.result.metrics:
                        data_table(
                            pd.DataFrame(
                                [
                                    {
                                        "Indicador": METRICS.get(label, (label,))[0],
                                        "Valor": str(value),
                                    }
                                    for label, value in output.result.metrics.items()
                                ]
                            ),
                            hide_index=True,
                            use_container_width=True,
                            target=st,
                        )
                st.caption("Los cálculos los realiza el framework sobre el dataset de esta sesión.")
        with tools:
            st.caption(
                "Elige una operación. Sus parámetros sugeridos se cargan automáticamente; revisa los requisitos antes de ejecutarla."
            )
            actions = sorted(agent.allowed_tools, key=lambda value: (value != "eda.profile", value))
            tool = st.selectbox(
                "¿Qué quieres revisar?",
                actions,
                key=f"tool_{name}",
                format_func=lambda value: SPECS[value].description,
            )
            with st.expander("Ajustar parámetros (opcional)"):
                st.caption(
                    "Puedes dejar los valores sugeridos. Esta edición avanzada utiliza formato JSON."
                )
                params = st.text_area(
                    "Parámetros JSON",
                    json.dumps(
                        defaults_for(tool, agent.context, agent.problems),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    key=f"args_{name}_{tool}",
                )
            if st.button("Preparar esta acción", key=f"quick_{name}", type="primary"):
                try:
                    response = agent.run(
                        agent.request(
                            SPECS[tool].description,
                            [ToolCall(tool=tool, parameters=json.loads(params))],
                        )
                    )
                    _store(agent, response)
                    st.rerun()
                except (ValueError, TypeError):
                    st.error("Revisa el formato de los parámetros.")
        with history:
            entries = agent.context.conversation.get(name, [])
            if entries:
                selected = st.selectbox(
                    "Registro de revisiones",
                    entries,
                    format_func=lambda entry: entry.operation,
                    key=f"history_{name}",
                )
                st.write(selected.operation)
                for recommendation in selected.recommendations:
                    st.write(recommendation.text)
                with st.expander("Ver registro completo"):
                    st.json(selected.model_dump(mode="json"))
            else:
                st.caption("Todavía no hay revisiones guardadas.")
        with chat:
            if not agent.provider.available:
                st.caption(
                    "La orientación local está disponible sin credenciales. Las preguntas libres necesitan un modelo de lenguaje opcional."
                )
            else:
                st.caption(f"Proveedor: {agent.provider.name}")
                if agent.configuration_error:
                    st.warning(agent.configuration_error)
                if st.button("Comprobar modelo local", key=f"probe_{name}"):
                    st.info(agent.provider_status()["message"])
                consent = st.checkbox(
                    "Permitir enviar mi pregunta y el contexto reducido al modelo configurado. No se enviarán filas ni tablas.",
                    key=f"consent_{name}",
                )
                question = st.text_area("Tu pregunta", key=f"question_{name}")
                if st.button("Enviar pregunta", key=f"ask_{name}") and question.strip():
                    _store(agent, agent.run(agent.request(question, consent=consent)))
                    st.rerun()
        response = flow.get("response")
        if response:
            with st.expander("Registro completo para depuración"):
                st.json(response.model_dump(mode="json"))
            st.download_button(
                "Descargar registro",
                response.model_dump_json(indent=2),
                "revision.json",
                "application/json",
                key=f"response_{name}",
            )
            for index, output in enumerate(response.tools):
                if isinstance(output.result, AnalysisResult):
                    for table_name, records in output.result.tables.items():
                        st.caption(table_name)
                        data_table(pd.DataFrame(records), use_container_width=True, target=st)
                    if output.result.proposal_id and st.button(
                        "Preparar aplicación de la propuesta", key=f"apply_proposal_{name}_{index}"
                    ):
                        _store(
                            agent,
                            agent.run(
                                agent.request(
                                    "Aplicar propuesta",
                                    [
                                        ToolCall(
                                            tool="eda.apply",
                                            parameters={"proposal_id": output.result.proposal_id},
                                        )
                                    ],
                                )
                            ),
                        )
                        st.rerun()


@st.dialog("Asistente ATLAS", width="small", on_dismiss=_close_dialog)
def assistant_dialog(name):
    title, description, icon = ASSISTANTS[name]
    heading, close = st.columns([5, 1])
    with heading:
        st.markdown(
            f'<div class="atlas-assistant-heading"><img alt="" src="{icon_mask(icon).replace("black", "%23331832")}"><h3>{escape(title)}</h3></div>',
            unsafe_allow_html=True,
        )
    if close.button("Cerrar", key=f"close_{name}"):
        _close_dialog()
        st.rerun()
    st.caption(description)
    if "eda" not in st.session_state:
        st.info("Primero necesitamos cargar tus datos para comenzar.")
        if st.button("Cargar datos", key=f"assistant_load_{name}", type="primary"):
            _navigate("data")
            st.rerun()
        return
    context = session_context()
    agent = AgentController(context, name, state=st.session_state)
    st.html(
        '<div class="atlas-assistant-dataset">'
        "<span>DATASET ACTIVO</span> "
        f"<strong>{escape(st.session_state.get('dataset_name', 'conjunto activo'))}</strong></div>"
    )
    flow = agent.flow()
    with st.container(key=f"panel_{name}"):
        if flow["phase"] == "ideas":
            overview = agent.overview()
            st.html(
                '<div class="atlas-assistant-welcome">'
                f"<p>{escape(overview.welcome)}</p></div>"
                '<div class="atlas-assistant-section"><span class="atlas-assistant-eyebrow">RECOMENDACIONES PARA TI</span>'
                "<h4>¿Por dónde quieres empezar?</h4><p>Elige una idea y te acompaño paso a paso.</p></div>"
            )
            for idea in overview.ideas:
                with st.container(border=True, key=f"assistant_idea_{name}_{idea.id}"):
                    text, action = st.columns([3, 1])
                    text.markdown(f"**{idea.title}**")
                    text.caption(idea.description)
                    if action.button(
                        "Explorar",
                        key=f"idea_{name}_{idea.id}",
                        use_container_width=True,
                        type="primary",
                    ):
                        with st.spinner("Revisando esta idea…"):
                            response = agent.choose(idea.id)
                            if response:
                                _store(agent, response)
                        st.rerun()
        else:
            idea = flow.get("idea")
            if idea:
                st.html(
                    '<section class="atlas-assistant-selected">'
                    '<span class="atlas-assistant-eyebrow">IDEA QUE ESTAMOS EXPLORANDO</span>'
                    f"<h4>{escape(idea.title)}</h4><p>{escape(idea.expected)}</p></section>"
                )
                if not flow.get("response") or flow["response"].status == "pending_confirmation":
                    st.caption(idea.effect)
                with st.expander("¿Por qué esta idea?"):
                    st.write(idea.why)
                if idea.destination and st.button(
                    "Continuar en la pantalla", key=f"continue_page_{name}", type="primary"
                ):
                    _navigate(idea.destination)
                    st.rerun()
            response = flow.get("response")
            if response:
                render_response(response, context, f"{name}_{response.request_id}")
                if response.confirmation_id and response.confirmation_id in context.pending:
                    _execution_plan(context, response)
                    confirm, cancel = st.columns(2)
                    if confirm.button(
                        "Continuar", key=f"confirm_{name}", type="primary"
                    ) and _execute_visible(agent, lambda: agent.confirm(response.confirmation_id)):
                        st.rerun()
                    if cancel.button("Cancelar", key=f"cancel_{name}"):
                        agent.cancel(response.confirmation_id)
                        flow.pop("response", None)
                        flow["phase"] = "explanation"
                        st.rerun()
            elif (
                idea and idea.goal and st.button("Continuar", key=f"resume_{name}", type="primary")
            ):
                if _execute_visible(agent, lambda: agent.run(agent.request(f"guided:{idea.goal}"))):
                    st.rerun()
            if st.button("Ver otras ideas", key=f"back_{name}"):
                agent.back()
                st.rerun()
        if st.toggle("Ver detalles técnicos", key=f"assistant_technical_{name}"):
            _technical_controls(agent, name)


def render_agent_panel(
    name: str, *, show_button: bool = True, button_only: bool = False, label: str | None = None
):
    """Entry point stays compatible with existing pages; UI renders as one dialog."""
    if show_button and st.button(
        label or f"Usar {ASSISTANTS[name][0].lower()}", key=f"open_{name}"
    ):
        previous = st.session_state.get("active_assistant")
        if previous and previous != name:
            _close_dialog()
        if "eda" in st.session_state:
            AgentController(session_context(), name, state=st.session_state).open()
        else:
            st.session_state[f"agent_open_{name}"] = True
            st.session_state.active_assistant = name
    if button_only or not st.session_state.get(f"agent_open_{name}", False):
        return
    st.session_state.dialog_rendered_this_run = True
    assistant_dialog(name)
