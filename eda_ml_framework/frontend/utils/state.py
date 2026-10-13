"""Estado privado por sesión; únicamente ejemplos públicos usan caché."""

import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, make_blobs

from framework import AnalisisDatosExploratorio


@st.cache_data
def example(name: str) -> pd.DataFrame:
    """Datasets públicos de ejemplo; no requiere descarga de internet."""
    if name == "Clusters sintéticos":
        x, _ = make_blobs(n_samples=180, centers=3, n_features=4, random_state=42)
        return pd.DataFrame(x, columns=[f"variable_{i + 1}" for i in range(4)])
    loader = {"Iris": load_iris, "Diabetes": load_diabetes, "Cáncer de mama": load_breast_cancer}[
        name
    ]
    return loader(as_frame=True).frame


def activate(df: pd.DataFrame, name: str) -> None:
    """Activar datos confirmados e invalidar resultados de otro dataset."""
    eda = AnalisisDatosExploratorio(df)
    for key in ["new_predictions", "representation"]:
        st.session_state.pop(key, None)
    st.session_state.update(
        original=df.copy(deep=True),
        eda=eda,
        dataset_name=name,
        work=eda.df,
        experiments=[],
        models={},
        metrics={},
        graphs={},
        pending=None,
        preprocessing={},
        selected_variables=[],
        problem=None,
    )


def initialize() -> None:
    """Inicializar sin sobrescribir estado en reruns."""
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
