"""Friendly presentation of stored experiments; no training or metric calculation."""

from html import escape
from math import isfinite

import pandas as pd
import streamlit as st

from atlas.controllers.comparison_controller import ComparisonController
from atlas.controllers.report_controller import export_frame
from atlas.views.components.layout import page_header
from atlas.views.components.sidebar import NAV_ITEMS, navigate
from atlas.views.components.tables import data_table
from atlas.views.components.ui import figure, format_number
from atlas.views.pages.modeling import experiment_downloads

PROBLEMS = {
    "clasificacion": "Predecir categorías",
    "regresion": "Predecir cantidades",
    "clustering": "Encontrar grupos",
}
MODELS = {
    "logistic": "Clasificación logística",
    "linear_multiple": "Regresión lineal",
    "ridge": "Regresión Ridge",
    "forest": "Bosque de árboles",
    "tree": "Árbol de decisión",
    "kmeans": "K-Means",
    "knn": "Vecinos cercanos",
    "gradient_boosting": "Árboles con aprendizaje secuencial",
    "agglomerative": "Agrupamiento jerárquico",
    "dbscan": "Grupos por densidad",
}
METRIC_COPY = {
    "Accuracy": (
        "Aciertos globales",
        ".1%",
        "Parte de las predicciones de prueba que fueron correctas. No muestra por sí sola los errores de cada categoría.",
    ),
    "F1_macro": (
        "Equilibrio entre categorías",
        ".3f",
        "De 0 a 1. Da el mismo peso a cada categoría; un valor mayor es preferible según este criterio.",
    ),
    "Balanced_accuracy": (
        "Detección equilibrada",
        ".1%",
        "Tiene en cuenta cada categoría, aunque algunas tengan pocos ejemplos.",
    ),
    "MAE": (
        "Error medio",
        ",.2f",
        "Diferencia absoluta media con el valor real, en las unidades de lo que predices. Menor es preferible.",
    ),
    "RMSE": (
        "Impacto de fallos grandes",
        ",.2f",
        "Da más peso a las equivocaciones grandes. Está en las mismas unidades del objetivo; menor es preferible.",
    ),
    "R2": (
        "Ajuste del modelo (R²)",
        ".3f",
        "1 corresponde a un ajuste perfecto; puede ser negativo. No garantiza que el modelo sea útil en tu caso.",
    ),
    "clusters": (
        "Grupos encontrados",
        ",.0f",
        "Cantidad de grupos obtenidos. Más grupos no significa un mejor resultado.",
    ),
    "Silhouette": (
        "Separación de los grupos",
        ".3f",
        "Entre −1 y 1. Un valor mayor indica grupos más separados según esta medida, sin demostrar su utilidad práctica.",
    ),
    "ruido": (
        "Casos sin grupo",
        ".1%",
        "Registros que el algoritmo dejó sin asignar. No significa que deban eliminarse.",
    ),
}


def experiments_page() -> None:
    """Review evidence, predictions and next actions without exposing implementation IDs."""
    page_header("history")
    experiments = st.session_state.experiments
    st.info(
        "Cada experimento es una prueba que ya realizaste. Aquí puedes entender sus resultados y decidir qué hacer después, sin volver a entrenar."
    )
    if not experiments:
        st.markdown("### Todavía no hay pruebas guardadas")
        st.write(
            "Elige qué quieres hacer. Al crear un modelo o un agrupamiento, su resultado aparecerá aquí."
        )
        for column, slug, label in zip(
            st.columns(3),
            ["classification", "regression", "clustering"],
            ["Predecir una categoría", "Predecir una cantidad", "Buscar grupos"],
        ):
            destination = next(item for item in NAV_ITEMS if item.slug == slug)
            column.button(
                label, key=f"history_start_{slug}", on_click=navigate, args=(destination.route,)
            )
        return
    labels = [
        "Todas las pruebas",
        *dict.fromkeys(PROBLEMS.get(item.problem, item.problem) for item in experiments),
    ]
    numbers = {item.id: index for index, item in enumerate(experiments, 1)}
    with st.container(border=True, key="history_selection_card"):
        st.subheader("1. Encuentra la prueba que quieres revisar")
        st.caption("Consulta lo que ya calculaste. No necesitas volver a entrenar.")
        selected_problem = st.selectbox(
            "¿Qué tipo de resultados quieres revisar?", labels, key="history_problem"
        )
        filtered = [
            item
            for item in experiments
            if selected_problem == labels[0]
            or PROBLEMS.get(item.problem, item.problem) == selected_problem
        ]
        result = st.selectbox(
            "Elige una prueba guardada",
            filtered,
            index=len(filtered) - 1,
            format_func=lambda item: (
                f"Prueba {numbers[item.id]} · {MODELS.get(item.name, item.name)} · {item.config.get('target') or 'Agrupamiento'}"
            ),
            key="history_experiment",
        )
        st.caption(
            f"{len(filtered)} pruebas disponibles · Mostrando la prueba {numbers[result.id]}."
        )
    target = result.config.get("target") or "Grupos de registros similares"
    st.html(
        '<div class="atlas-result-context">'
        f"<div><span>QUÉ SE BUSCABA</span><strong>{escape(PROBLEMS.get(result.problem, result.problem))}</strong></div>"
        f"<div><span>RESULTADO PARA</span><strong>{escape(str(target))}</strong></div>"
        f"<div><span>TIEMPO DE CÁLCULO</span><strong>{format_number(result.seconds)} s</strong></div></div>"
    )
    active = st.session_state.eda
    if result.dataset_id not in {
        active.dataset_version.dataset_id,
        active.dataset_version.content_hash,
    }:
        st.warning(
            "Esta prueba corresponde a otros datos o a una versión anterior. Sus cifras no describen necesariamente el dataset activo."
        )
    st.markdown("### ¿Qué nos dice esta prueba?")
    metrics = result.metrics_for_context()
    keys = {
        "clasificacion": ["Accuracy", "F1_macro", "Balanced_accuracy"],
        "regresion": ["MAE", "RMSE", "R2"],
        "clustering": ["clusters", "Silhouette", "ruido"],
    }.get(result.problem, [])
    cards = []
    for key in keys:
        value = metrics.get(key)
        if value is None or not isfinite(value):
            continue
        title, formatting, description = METRIC_COPY[key]
        cards.append(
            f'<article class="atlas-result-card"><span>{escape(title)}</span><strong>{format_number(value, formatting)}</strong><p>{escape(description)}</p></article>'
        )
    if cards:
        st.html('<div class="atlas-result-grid">' + "".join(cards) + "</div>")
    else:
        st.info(
            "No hay indicadores resumidos disponibles para esta prueba. Puedes consultar las cifras registradas en los detalles avanzados."
        )
    st.caption(
        "Son resultados calculados de esta prueba. No bastan para declarar un modelo mejor para todos los usos."
    )
    predictions, charts, details = st.tabs(
        [
            "Ver los resultados registro por registro",
            "Entender con un gráfico",
            "Detalles avanzados",
        ]
    )
    with predictions:
        st.write(
            "Estos son los registros evaluados, no necesariamente todas las filas del dataset. En modelos predictivos se muestran los datos reservados para la prueba."
        )
        preview = result.predictions.head(100).rename(
            columns={
                "real": "Valor real",
                "predicho": "Predicción",
                "prediccion": "Predicción",
                "cluster": "Grupo asignado",
            }
        )
        data_table(preview, use_container_width=True, target=st)
        st.caption(
            f"Vista previa de {len(preview):,} registros; esta prueba tiene {len(result.predictions):,} resultados guardados. La descarga incluye todos los resultados disponibles."
        )
        with st.container(border=True, key="history_download_card"):
            st.markdown("#### Lleva contigo los resultados")
            st.caption(
                f"Archivo CSV con los {len(result.predictions):,} registros evaluados de la prueba {numbers[result.id]}. Incluye todas las filas guardadas, no solo la vista previa."
            )
            st.download_button(
                "Descargar los resultados de esta prueba",
                export_frame(result.predictions),
                f"resultados_{numbers[result.id]}.csv",
                "text/csv",
                key=f"history_predictions_{result.id}",
                use_container_width=True,
            )
    with charts:
        if result.problem == "clustering":
            st.write(
                "Abre Clustering para revisar los tamaños, las características y la representación de los grupos guardados."
            )
        else:
            chart_kind = "confusion" if result.problem == "clasificacion" else "actual_predicted"
            st.caption(
                "La diagonal muestra los aciertos; fuera de ella están las confusiones entre categorías."
                if result.problem == "clasificacion"
                else "Cuanto más cerca estén los puntos de la línea de referencia, más próximas son las predicciones a los valores reales."
            )
            if st.button(
                "Mostrar cómo acertó y dónde se equivocó", key=f"history_chart_{result.id}"
            ):
                try:
                    st.session_state.graphs[f"history_{result.id}"] = st.session_state.models[
                        result.id
                    ].plot_result(result, chart_kind)
                except (ValueError, TypeError, KeyError) as exc:
                    st.warning(f"No se pudo mostrar este gráfico: {exc}")
            if f"history_{result.id}" in st.session_state.graphs:
                figure(st.session_state.graphs[f"history_{result.id}"], f"history_plot_{result.id}")
    with details:
        st.caption(
            "Información para reproducir la prueba. No necesitas cambiarla para consultar los resultados."
        )
        st.write(f"Identificador: {result.id} · Versión del modelo: {result.model_version}")
        data_table(
            pd.DataFrame([{"Indicador": key, "Valor": value} for key, value in metrics.items()]),
            hide_index=True,
            use_container_width=True,
            target=st,
        )
        with st.expander("Ver configuración completa"):
            st.json(dict(result.config))
        experiment_downloads(result)
    groups = ComparisonController.groups(experiments)
    compatible = next(
        (group for group in groups.values() if any(item.id == result.id for item in group)), []
    )
    destination = next(
        item
        for item in NAV_ITEMS
        if item.slug
        == {
            "clasificacion": "classification",
            "regresion": "regression",
            "clustering": "clustering",
        }[result.problem]
    )
    comparison = next(item for item in NAV_ITEMS if item.slug == "experiments")

    def open_comparison():
        """Carry the reviewed cohort into the comparison view without training."""
        cohort = next(
            key for key, group in groups.items() if any(item.id == result.id for item in group)
        )
        st.session_state["comparison_cohort"] = cohort
        st.session_state[f"comparison_models_{cohort}"] = [item.id for item in compatible]
        navigate(comparison.route)

    with st.container(border=True, key="history_next_actions"):
        st.subheader("Elige tu siguiente paso")
        st.caption("Puedes seguir explorando sin perder esta prueba.")
        with st.container(border=True, key="history_compare_action"):
            st.markdown("#### ¿Quieres saber qué alternativa funciona mejor?")
            st.write(
                f"Compara {len(compatible)} pruebas que comparten datos, variables y evaluación. Elige el criterio que importa para tu objetivo."
                if len(compatible) >= 2
                else "Necesitas otra prueba con los mismos datos, variables y evaluación. Crea una alternativa antes de comparar."
            )
            st.button(
                "Comparar mis alternativas",
                key="history_compare",
                type="primary",
                disabled=len(compatible) < 2,
                on_click=open_comparison,
                use_container_width=True,
            )
        with st.container(border=True, key="history_return_action"):
            st.markdown("#### ¿Quieres probar otra opción?")
            st.caption(
                f"Vuelve a {destination.label} para revisar sus opciones. Abrir el módulo no inicia un entrenamiento."
            )
            st.button(
                f"Abrir {destination.label}",
                key="history_return",
                on_click=navigate,
                args=(destination.route,),
                use_container_width=True,
            )
