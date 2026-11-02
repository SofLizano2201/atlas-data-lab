"""Punto de entrada: streamlit run app.py."""

from base64 import b64encode
from importlib.resources import files

import streamlit as st

from atlas.views.components.sidebar import render_sidebar, render_sidebar_status
from atlas.views.pages.experiments import experiments_page
from atlas.views.pages.modeling import clustering_page, comparison_page, supervised_page
from atlas.views.pages.reports import reports_page
from atlas.views.pages.workspace import cleaning_page, data_page, eda_page, overview
from atlas.views.utils.state import initialize


def main() -> None:
    """Enrutar páginas y manejar errores comprensibles sin ocultar fallos de desarrollo."""
    st.set_page_config(page_title="Atlas · Data Lab", layout="wide")
    resources = files("atlas.views")
    css = resources.joinpath("styles", "theme.css")
    tokens = resources.joinpath("styles", "tokens.css")
    composition = resources.joinpath("styles", "mockups.css").read_text(encoding="utf-8")
    landscape = resources.joinpath("assets", "atlas-landscape.svg")
    landscape_uri = "data:image/svg+xml;base64," + b64encode(landscape.read_bytes()).decode("ascii")
    composition = composition.replace("__ATLAS_LANDSCAPE__", landscape_uri)
    workspace_style = resources.joinpath("styles", "workspace.css").read_text(encoding="utf-8")
    st.html(
        f"<style>{css.read_text(encoding='utf-8')}\n{tokens.read_text(encoding='utf-8')}\n{composition}\n{workspace_style}</style>",
    )
    initialize()
    page, sidebar_status = render_sidebar()
    if page not in {"🏠 Inicio", "📂 Datos"} and "eda" not in st.session_state:
        st.title("Tu análisis comienza con datos")
        st.info("Abre Datos y confirma un archivo o un ejemplo para continuar.")
        render_sidebar_status(sidebar_status)
        return
    try:
        if page == "🏠 Inicio":
            overview()
        elif page == "📂 Datos":
            data_page()
        elif page == "🔎 Exploración":
            eda_page()
        elif page == "🧹 Preparación":
            cleaning_page()
        elif page == "🧩 Clustering":
            clustering_page()
        elif page == "🎯 Clasificación":
            supervised_page("clasificacion")
        elif page == "📈 Regresión":
            supervised_page("regresion")
        elif page == "⚖️ Experimentos":
            comparison_page()
        elif page == "experiment_history":
            experiments_page()
        else:
            reports_page()
    except (ValueError, TypeError, KeyError, ImportError, OSError, RuntimeError) as exc:
        st.error(f"No se pudo completar la operación: {exc}")
        st.caption("Revisa las variables seleccionadas, el tipo de datos y los parámetros.")
    render_sidebar_status(sidebar_status)


if __name__ == "__main__":
    main()
