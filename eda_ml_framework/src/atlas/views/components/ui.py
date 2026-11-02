"""Spanish Streamlit views: rendering and input collection only."""

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st


def figure(fig: Any, key: str) -> None:
    """Provide figure using validated inputs."""
    if isinstance(fig, go.Figure):
        with st.popover("Opciones del gráfico"):
            zoom = st.checkbox("Zoom con rueda", key=f"chart_zoom_{key}")
            inspect = st.checkbox(
                "Seleccionar puntos",
                key=f"chart_inspect_{key}",
                disabled=not any(trace.type in {"scatter", "scattergl"} for trace in fig.data),
            )
            enlarged = st.checkbox("Vista ampliada", key=f"chart_large_{key}")
        display = go.Figure(fig)
        if enlarged:
            display.update_layout(height=760)
        event = st.plotly_chart(
            display,
            use_container_width=True,
            theme=None,
            key=f"plot_{key}",
            on_select="rerun" if inspect else "ignore",
            selection_mode=("points", "box", "lasso"),
            config={
                "displaylogo": False,
                "scrollZoom": zoom,
                "toImageButtonOptions": {"format": "png", "scale": 2},
            },
        )
        if inspect and event and event.selection.points:
            st.caption("Puntos seleccionados · coordenadas reales mostradas por el gráfico")
            st.dataframe(event.selection.points, use_container_width=True)
        st.download_button(
            "Descargar gráfico HTML",
            fig.to_html(include_plotlyjs=True),
            f"{key}.html",
            "text/html",
            key=f"download_{key}",
        )
    else:
        native = getattr(fig, "figure", getattr(fig, "fig", fig))
        st.pyplot(native)
        buffer = BytesIO()
        native.savefig(buffer, format="png", bbox_inches="tight")
        st.download_button(
            "Descargar gráfico PNG",
            buffer.getvalue(),
            f"{key}.png",
            "image/png",
            key=f"download_{key}",
        )
        plt.close(native)
