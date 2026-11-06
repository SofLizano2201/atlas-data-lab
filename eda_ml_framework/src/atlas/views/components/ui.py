"""Spanish Streamlit views: rendering and input collection only."""

from hashlib import sha256
from html import escape
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from atlas.controllers.charts import chart_frame, style
from atlas.views.components.tables import data_table


def figure(fig: Any, key: str) -> None:
    """Provide figure using validated inputs."""
    if isinstance(fig, go.Figure):
        fig = style(go.Figure(fig))
        with st.container(border=True, key=f"chart_card_{key}"):
            preferences = st.session_state.setdefault("chart_preferences", {}).setdefault(key, {})
            actions, tools = st.columns([1, 1.4])
            if actions.button(
                "Abrir gráfico grande",
                key=f"chart_fullscreen_{key}",
                disabled=st.session_state.get("dialog_rendered_this_run", False),
                help="Abre una vista amplia sin modificar el análisis. Dentro de un asistente puedes aumentar la altura desde las opciones.",
            ):
                enlarged_chart(fig, key)
            selectable = any(
                trace.type in {"scatter", "scattergl"} and trace.mode and "markers" in trace.mode
                for trace in fig.data
            )
            names = list(
                dict.fromkeys(
                    trace.name for trace in fig.data if trace.name and trace.showlegend is not False
                )
            )
            with tools.popover("Explorar y descargar"):
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
            display = go.Figure(fig)
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
                config={
                    "displaylogo": False,
                    "scrollZoom": zoom,
                    "responsive": True,
                    "toImageButtonOptions": {
                        "format": "png",
                        "scale": 2,
                        "filename": "atlas_grafico",
                    },
                },
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
            with st.expander("Descargar este gráfico o sus datos"):
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
                        display.to_html(include_plotlyjs=True).replace(
                            "</body>", f"<p>{escape(caption)}</p></body>"
                        ),
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
    display = go.Figure(fig)
    display.update_layout(height=720)
    st.plotly_chart(
        display,
        use_container_width=True,
        theme=None,
        key=f"expanded_plot_{key}",
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "scale": 2},
        },
    )
    caption = (display.layout.meta or {}).get("atlas_caption")
    if caption:
        st.caption(caption)
