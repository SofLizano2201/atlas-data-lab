"""Private Streamlit state; dataset workflows live in services/controllers."""

import streamlit as st

from atlas.controllers.dataset_controller import DatasetController


@st.cache_data
def example(name):
    return DatasetController.example(name)


def activate(df, name):
    DatasetController.activate(st.session_state, df, name)


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
