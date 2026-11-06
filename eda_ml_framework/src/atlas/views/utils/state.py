"""Private Streamlit state; dataset workflows live in services/controllers."""

import streamlit as st

from atlas.controllers.automatic_eda_controller import AutomaticEDAController
from atlas.controllers.dataset_controller import DatasetController


@st.cache_data
def example(name):
    return DatasetController.example(name)


def activate(df, name, autonomous=False):
    with st.status("Analizando tus datos…", expanded=True) as status:

        def progress(operation):
            messages = {
                "profile": "Revisando tamaño y calidad de los datos…",
                "descriptive_statistics": "Calculando estadísticas de las variables…",
                "outliers_iqr": "Revisando valores poco habituales…",
                "correlation_pearson": "Explorando relaciones entre variables…",
            }
            if operation in messages:
                status.write(messages[operation])

        st.session_state.eda_progress_callback = progress
        try:
            DatasetController.activate(st.session_state, df, name)
            execution = st.session_state.automatic_eda[
                AutomaticEDAController.key(st.session_state.eda)
            ]
            status.update(
                label="No se pudo completar la exploración inicial. Revisa EDA."
                if execution.status == "failed"
                else "Exploración inicial disponible. Consulta sus resultados y límites en EDA.",
                state="error" if execution.status == "failed" else "complete",
                expanded=False,
            )
        finally:
            st.session_state.pop("eda_progress_callback", None)
    if autonomous:
        run_autonomous()


def run_autonomous(target=None):
    """Show progress while the controller executes the explicitly authorized workflow."""
    from atlas.controllers.autonomous_controller import AutonomousController

    with st.status("ATLAS está analizando tus datos de principio a fin…", expanded=True) as status:
        run = AutonomousController.run(st.session_state, target=target, progress=status.write)
        status.update(
            label="Análisis automático disponible: revisa resultados y límites abajo.",
            state="error" if run.status == "failed" else "complete",
            expanded=False,
        )


def initialize():
    DatasetController.upgrade(st.session_state)
    for key, value in {
        "experiments": [],
        "models": {},
        "metrics": {},
        "graphs": {},
        "preprocessing": {},
        "selected_variables": [],
        "pending": None,
        "problem": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value
