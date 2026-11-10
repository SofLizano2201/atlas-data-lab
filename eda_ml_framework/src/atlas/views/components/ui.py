"""Spanish Streamlit views: rendering and input collection only."""

from hashlib import sha256
from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from atlas.controllers.charts import chart_frame
from atlas.views.components.chart_theme import chart_config, chart_title, copy_chart, present_chart
from atlas.views.components.tables import data_table


def figure(fig: Any, key: str, title: str | None = None) -> None:
    """Provide figure using validated inputs."""
    if isinstance(fig, go.Figure):
        title = title or chart_title(fig)
        fig = present_chart(fig)
        with st.container(border=True, key=f"chart_card_{key}"):
            preferences = st.session_state.setdefault("chart_preferences", {}).setdefault(key, {})
            heading, tools = st.columns([9, 1])
            heading.html(f'<h3 class="atlas-chart-title">{escape(title)}</h3>')
            selectable = any(
                trace.type in {"scatter", "scattergl"} and trace.mode and "markers" in trace.mode
                for trace in fig.data
            )
            names = list(
                dict.fromkeys(
                    trace.name for trace in fig.data if trace.name and trace.showlegend is not False
                )
            )
            with tools.popover("", icon=":material/more_vert:", help="Opciones del gráfico"):
                if st.button(
                    "Abrir gráfico grande",
                    key=f"chart_fullscreen_{key}",
                    disabled=st.session_state.get("dialog_rendered_this_run", False),
                ):
                    enlarged_chart(fig, key)
                st.caption(
                    "Arrastra para acercarte; haz doble clic para restablecer. La leyenda permite ocultar y mostrar series."
                )
                zoom = st.checkbox(
                    "Zoom con rueda", value=preferences.get("zoom", False), key=f"chart_zoom_{key}"
                )
                inspect = (
                    st.checkbox(
                        "Seleccionar puntos",
                        value=preferences.get("inspect", False),
                        key=f"chart_inspect_{key}",
                        disabled=not selectable,
                    )
                    and selectable
                )
                enlarged = st.checkbox(
                    "Vista ampliada",
                    value=preferences.get("large", False),
                    key=f"chart_large_{key}",
                )
                selected_names = names
                if len(names) > 1:
                    state_key = f"chart_series_{key}"
                    if preferences.get("available_series") != names:
                        preferences["series"] = names
                        if state_key in st.session_state:
                            st.session_state[state_key] = names
                    preferences["available_series"] = names
                    if state_key in st.session_state:
                        st.session_state[state_key] = [
                            name for name in st.session_state[state_key] if name in names
                        ]
                    selected_names = st.multiselect(
                        "Series o grupos visibles",
                        names,
                        default=[
                            name for name in preferences.get("series", names) if name in names
                        ],
                        key=state_key,
                    )
                preferences.update(
                    zoom=zoom, inspect=inspect, large=enlarged, series=selected_names
                )
            display = copy_chart(fig)
            if enlarged:
                display.update_layout(height=760)
            if inspect:
                display.update_layout(dragmode="select")
            for trace in display.data:
                if trace.name in names and trace.name not in selected_names:
                    trace.visible = "legendonly"
            revision = sha256(display.to_json().encode()).hexdigest()[:12]
            event = st.plotly_chart(
                display,
                use_container_width=True,
                theme=None,
                key=f"plot_{key}_{revision}",
                on_select="rerun" if inspect else "ignore",
                selection_mode=("points", "box", "lasso"),
                config=chart_config(zoom),
            )
            caption = (display.layout.meta or {}).get("atlas_caption", "")
            if caption:
                st.caption(caption)
            if inspect:
                points = list(event.selection.points) if event and event.selection else []
                st.session_state.setdefault("chart_selections", {})[key] = points
                if points:
                    st.caption("Observaciones seleccionadas · valores reales del gráfico")
                    data_table(points, use_container_width=True, target=st)
                else:
                    st.caption(
                        "Dibuja un rectángulo o lazo sobre los puntos para inspeccionarlos. La selección no cambia los datos ni entrena modelos."
                    )
            with st.expander("Descargar", expanded=False):
                st.caption(
                    "PNG: usa la cámara en la barra del gráfico. HTML conserva zoom y leyendas. CSV contiene los datos de las series elegidas en las opciones; en histogramas y cajas son observaciones originales, no intervalos calculados por el navegador."
                )
                st.download_button(
                    "Descargar datos del gráfico (CSV)",
                    chart_frame(display).to_csv(index=False).encode("utf-8-sig"),
                    f"{key}.csv",
                    "text/csv",
                    key=f"chart_data_{key}",
                )
                if st.checkbox("Preparar archivo interactivo", key=f"chart_export_{key}"):
                    st.download_button(
                        "Descargar gráfico HTML",
                        display.to_html(include_plotlyjs=True)
                        .replace("<body>", f"<body><h1>{escape(title)}</h1>")
                        .replace("</body>", f"<p>{escape(caption)}</p></body>"),
                        f"{key}.html",
                        "text/html",
                        key=f"download_{key}",
                    )
    else:
        raise TypeError(
            "Este resultado no es un gráfico interactivo compatible. Vuelve a generar el gráfico con la versión actual de ATLAS."
        )


def format_number(value, formatting=",.2f"):
    """Display stored numbers with Spanish separators, preserving metric units."""
    return format(value, formatting).translate(str.maketrans({",": ".", ".": ","}))


@st.dialog("Vista ampliada del gráfico", width="large")
def enlarged_chart(fig, key):
    st.markdown(f"**{(fig.layout.meta or {}).get('atlas_display_title', chart_title(fig))}**")
    display = copy_chart(fig)
    display.update_layout(height=720)
    st.plotly_chart(
        display,
        use_container_width=True,
        theme=None,
        key=f"expanded_plot_{key}",
        config=chart_config(),
    )
    caption = (display.layout.meta or {}).get("atlas_caption")
    if caption:
        st.caption(caption)
