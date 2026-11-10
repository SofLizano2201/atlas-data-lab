"""Compact exploratory workspace using existing scientific chart methods."""

import streamlit as st

from atlas.controllers.visual_dashboard_controller import VisualDashboardController
from atlas.views.components.ui import figure


def render_visual_dashboard():
    frame = st.session_state.eda.df
    numeric = list(frame.select_dtypes(include="number").columns)
    if not numeric:
        st.info("Selecciona un dataset con variables numéricas para abrir el panel visual.")
        return
    groups = [None] + [c for c in frame if 1 < frame[c].nunique() <= 8]
    variable, category, method = st.columns([1, 1, 1])
    x = variable.selectbox("Variable principal", numeric, key="visual_primary")
    group = category.selectbox(
        "Categoría", groups, format_func=lambda c: c or "Sin agrupar", key="visual_group"
    )
    correlation = method.selectbox(
        "Correlación", ["pearson", "spearman", "kendall"], key="visual_correlation"
    )

    def draw(kind, title, key, **parameters):
        try:
            figure(
                VisualDashboardController.chart(st.session_state, kind, **parameters),
                key,
                title=title,
            )
        except ValueError as exc:
            st.info(str(exc))

    distribution, box, matrix = st.columns(3, gap="small")
    with distribution:
        draw("histogram", "Distribución de la variable", "visual_histogram", x=x)
    with box:
        draw("box", "Distribución por categoría", "visual_box", x=x, color=group)
    with matrix:
        draw(
            "correlation",
            "Matriz de correlación",
            "visual_matrix",
            columns=numeric[:8],
            method=correlation,
        )
        if len(numeric) > 8:
            st.caption(
                "Primeras ocho variables numéricas. La vista detallada permite cambiar la selección."
            )
    missing, relationship = st.columns([1, 2], gap="small")
    with missing:
        draw("missing_summary", "Valores faltantes", "visual_missing")
    with relationship:
        horizontal, vertical = st.columns(2)
        sx = horizontal.selectbox("Dispersión · variable X", numeric, key="visual_scatter_x")
        sy = vertical.selectbox(
            "Dispersión · variable Y", numeric, index=min(1, len(numeric) - 1), key="visual_scatter_y"
        )
        draw("scatter", "Relación entre variables", "visual_scatter", x=sx, y=sy, color=group)
