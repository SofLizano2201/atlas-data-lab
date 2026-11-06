"""Readable autonomous workflow results; all decisions and calculations stay outside the view."""

from html import escape

import streamlit as st

from atlas.views.components.tables import data_table
from atlas.views.components.ui import figure, format_number
from atlas.views.utils.state import run_autonomous


def render_autonomous_results():
    with st.container(border=True, key="workflow_autonomous"):
        st.subheader("Tu análisis automático")
        run = st.session_state.get("autonomous_runs", {}).get(
            st.session_state.get("autonomous_active")
        )
        if run is None:
            st.caption(
                "ATLAS puede generar gráficos, probar predicciones y grupos, comparar sus resultados y preparar un informe. El trabajo manual sigue disponible."
            )
            if st.button("Analizar todo automáticamente", type="primary"):
                run_autonomous()
                st.rerun()
            return
        st.markdown(
            '<div class="atlas-auto-goal"><span>QUÉ ANALIZAMOS</span>'
            f"<strong>Predecir {escape(str(run.target))}</strong>"
            "<p>"
            + (
                "ATLAS sugirió esta columna. Si no es lo que necesitas, puedes cambiarla al final de este panel."
                if run.provisional
                else "Elegiste esta columna para orientar las predicciones."
            )
            + "</p></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{len(run.experiments)} alternativas revisadas · {len(run.figures)} gráficos disponibles"
        )
        overview, charts, decisions = st.tabs(
            ["Qué encontré", "Ver los gráficos", "Por qué tomé estas decisiones"]
        )
        with overview:
            if run.errors:
                st.warning(
                    "El recorrido terminó con algunas operaciones pendientes. Puedes consultar las razones en «Por qué tomé estas decisiones»; los resultados disponibles siguen siendo utilizables dentro de sus límites."
                )
            for comparison in run.comparisons:
                result = next(
                    item for item in st.session_state.experiments if item.id == comparison["best"]
                )
                render_recommendation(result, comparison, run)
            if not run.comparisons:
                st.warning(
                    "No hubo suficientes resultados compatibles para recomendar un modelo. Revisa las razones abajo; los gráficos y el modo manual siguen disponibles."
                )
            st.markdown("**Conserva tus resultados**")
            st.download_button(
                "Descargar el informe completo",
                run.report,
                "atlas_analisis_automatico.html",
                "text/html",
                type="primary",
            )
            st.caption(
                "El informe HTML incluye gráficos interactivos y funciona sin conexión. Los experimentos también están disponibles en Experimentos."
            )
        with charts:
            names = list(run.figures)
            if names:
                labels = {
                    "confusion": "Dónde acierta y se equivoca la predicción",
                    "roc": "Capacidad para distinguir categorías",
                    "actual_predicted": "Valores reales frente a predicciones",
                    "residuals": "Errores de las predicciones",
                    "cluster_sizes": "Cuántos registros hay en cada grupo",
                    "clusters": "Cómo se ven los grupos en una proyección",
                    "comparison_clustering": "Comparar las alternativas de agrupamiento",
                    "comparison_clasificacion": "Comparar las predicciones de categorías",
                    "comparison_regresion": "Comparar las predicciones numéricas",
                }
                selected = st.selectbox(
                    "¿Qué resultado quieres explorar?",
                    names,
                    format_func=lambda name: labels.get(
                        name,
                        "Exploración · "
                        + name.removeprefix("chart:")
                        .replace("box:", "valores poco habituales · ")
                        .replace("correlation", "relaciones entre variables")
                        .replace("scatter", "relación entre dos variables")
                        .replace("missing", "información faltante"),
                    ),
                )
                figure(run.figures[selected], f"autonomous_{run.key}_{selected}")
        with decisions:
            st.caption(
                f"Tiempo de ejecución: {format_number(run.seconds)} segundos. Las explicaciones corresponden a los resultados guardados."
            )
            for decision in run.decisions:
                st.write(decision["reason"])
                if decision.get("features"):
                    st.caption("Variables elegidas: " + ", ".join(decision["features"]))
            for warning in run.warnings:
                st.warning(warning)
            if run.errors:
                st.warning(
                    f"{len(run.errors)} operaciones no pudieron completarse. El resto de los resultados se conserva."
                )
                with st.expander("Revisar qué no se pudo hacer"):
                    for error in run.errors:
                        st.write(error)
        with st.expander("Quiero orientar el análisis a otro objetivo"):
            target = st.selectbox("¿Qué columna necesitas predecir?", list(st.session_state.eda.df))
            st.caption(
                "Se conservarán los experimentos anteriores. ATLAS probará alternativas para este objetivo, sin cambiar tus datos."
            )
            if st.button("Analizar con este objetivo"):
                run_autonomous(target)
                st.rerun()


def toggle_result_chart(key):
    """Show stored diagnostics without running another analysis."""
    st.session_state[key] = not st.session_state.get(key, False)


def render_recommendation(result, comparison, run):
    """Present stored evidence with a clear hierarchy and ordinary-language metric descriptions."""
    clustering = comparison["problem"] == "clustering"
    names = {
        "logistic": "Regresión logística",
        "ridge": "Regresión con regularización",
        "tree": "Árbol de decisión",
        "forest": "Bosque de árboles",
        "kmeans": "K-Means",
        "agglomerative": "Agrupamiento por niveles",
    }
    model_name = names.get(result.name, result.name)
    if clustering:
        heading, kicker = "Para descubrir grupos similares", "OTRA FORMA DE EXPLORAR TUS DATOS"
        count = result.metrics.get("clusters")
        value = str(int(count)) if count is not None else "Sin dato"
        label = "Grupos encontrados"
        reading = "Esta alternativa separó los registros en esos grupos. Es una propuesta para explorar, no una clasificación definitiva."
        reason = "Fue la alternativa con mayor separación media según la silueta, entre los agrupamientos probados."
        graph_name, action = "clusters", "Ver cómo se distribuyen los grupos"
    else:
        heading, kicker = "Para hacer predicciones", "RECOMENDACIÓN PRINCIPAL"
        reason = "Fue la mejor alternativa en la validación con los datos de entrenamiento. Los datos reservados para la comprobación final no se usaron para elegirla."
        if "MAE" in result.metrics:
            value = format_number(result.metrics["MAE"])
            label = "Diferencia media al predecir"
            reading = f"En los registros reservados para comprobar el modelo, las predicciones se apartaron en promedio {value} unidades del valor real de «{run.target}». Algunos errores pueden ser mayores. Menor diferencia es mejor."
        elif "Accuracy" in result.metrics:
            value = format_number(result.metrics["Accuracy"] * 100, ",.1f") + " %"
            label = "Predicciones correctas en la prueba"
            reading = "Porcentaje de aciertos en los registros reservados para comprobar el modelo. Revisa también en qué categorías se equivoca; este dato por sí solo no cuenta toda la historia."
        else:
            value, label, reading = (
                "Disponible",
                "Resultado guardado",
                "Consulta la comparación para ver las medidas disponibles.",
            )
        graph_name = "actual_predicted" if result.problem == "regresion" else "confusion"
        action = "Ver las predicciones y sus errores"
    with st.container(border=True, key=f"auto_recommendation_{result.id}"):
        st.markdown(
            f'<article class="atlas-auto-recommendation {"grouping" if clustering else "prediction"}">'
            f'<div class="atlas-auto-kicker">{escape(kicker)}</div><h3>{escape(heading)}</h3>'
            f'<div class="atlas-auto-model"><span>Alternativa recomendada</span><strong>{escape(model_name)}</strong></div>'
            f'<div class="atlas-auto-evidence"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
            f'<p class="atlas-auto-reading">{escape(reading)}</p></article>',
            unsafe_allow_html=True,
        )
        if graph_name in run.figures:
            key = f"auto_chart_open_{result.id}"
            st.button(
                "Ocultar gráfico" if st.session_state.get(key) else action,
                key=f"auto_chart_button_{result.id}",
                on_click=toggle_result_chart,
                args=(key,),
                type="primary" if not clustering else "secondary",
            )
            if st.session_state.get(key):
                figure(run.figures[graph_name], f"auto_inline_{result.id}")
        with st.expander("Por qué se recomienda y qué alternativas se probaron"):
            st.write(reason)
            st.caption(
                "La recomendación se limita a las alternativas evaluadas y a ese criterio. No garantiza el mejor modelo posible para cualquier uso."
            )
            data_table(comparison["table"], use_container_width=True, target=st)
