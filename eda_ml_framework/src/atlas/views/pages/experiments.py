"""Session experiment history; no training or statistical computations in the view."""

import streamlit as st

from atlas.views.components.layout import page_header
from atlas.views.pages.modeling import experiment_downloads


def experiments_page() -> None:
    page_header("history")
    experiments = st.session_state.experiments
    if not experiments:
        st.info(
            "Aún no hay experimentos. Entrena un modelo o ejecuta un agrupamiento para guardar sus resultados."
        )
        return
    result = st.selectbox(
        "Experimento guardado",
        experiments,
        format_func=lambda item: f"{item.name} · {item.problem} · {item.id}",
        key="history_experiment",
    )
    st.caption(
        f"Dataset {result.dataset_id[:12]} · cohorte {result.cohort[:12]} · versión {result.model_version}"
    )
    details, metrics = st.columns(2)
    with details.container(border=True):
        st.subheader("Configuración reproducible")
        st.json(dict(result.config))
    with metrics.container(border=True):
        st.subheader("Métricas calculadas")
        st.json(result.metrics_for_context())
    experiment_downloads(result)
