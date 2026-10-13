"""Punto de entrada: streamlit run app.py."""

from pathlib import Path

import streamlit as st

from frontend.components.sidebar import render_sidebar, render_sidebar_status
from frontend.pages.modeling import clustering_page, comparison_page, supervised_page
from frontend.pages.workspace import cleaning_page, data_page, downloads, eda_page, overview
from frontend.utils.state import initialize


def main() -> None:
    """Enrutar páginas y manejar errores comprensibles sin ocultar fallos de desarrollo."""
    st.set_page_config(page_title="Atlas · Data Lab", page_icon="🔬", layout="wide")
    css = Path(__file__).parent / "frontend" / "styles" / "theme.css"
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
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
        else:
            st.title("Reportes / Compartir resultados")
            downloads(st.session_state.eda)
    except (ValueError, TypeError, KeyError, ImportError, OSError, RuntimeError) as exc:
        st.error(f"No se pudo completar la operación: {exc}")
        st.caption("Revisa las variables seleccionadas, el tipo de datos y los parámetros.")
    render_sidebar_status(sidebar_status)


if __name__ == "__main__":
    main()
