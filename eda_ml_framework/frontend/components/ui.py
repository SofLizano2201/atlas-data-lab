"""Presentación de figuras y descargas con claves estables."""

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st


def figure(fig: Any, key: str) -> None:
    """Renderizar y permitir exportar la figura realmente calculada."""
    if isinstance(fig, go.Figure):
        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None,
            key=f"plot_{key}",
            config={
                "displaylogo": False,
                "scrollZoom": False,
                "toImageButtonOptions": {"format": "png", "scale": 2},
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
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
