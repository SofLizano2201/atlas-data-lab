"""Frontend styling must preserve scientific evidence and report source figures."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atlas.models.eda import ExploratoryDataAnalysis
from atlas.views.components.chart_theme import chart_title, present_chart


@pytest.mark.parametrize("kind", ["histogram", "scatter", "correlation"])
def test_frontend_preserves_scientific_traces_and_source(kind):
    eda = ExploratoryDataAnalysis(
        pd.DataFrame(
            {
                "x": range(20),
                "y": np.arange(20) ** 2,
                "constant": [1] * 20,
            }
        )
    )
    source = eda.chart(kind, x="x", y="y" if kind == "scatter" else None)
    before = source.to_json()
    display = present_chart(source)
    assert source.to_json() == before
    assert display.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert "<br>" not in chart_title(source)
    assert not display.layout.title.text
    assert len(display.data) == len(source.data)
    for original, styled in zip(source.data, display.data, strict=True):
        for field in ["x", "y", "z", "customdata", "xbins", "ybins", "histnorm", "histfunc"]:
            if hasattr(original, field):
                a, b = getattr(original, field), getattr(styled, field)
                if hasattr(a, "to_plotly_json"):
                    assert a.to_plotly_json() == b.to_plotly_json()
                else:
                    np.testing.assert_equal(a, b)
        if original.type == "heatmap":
            assert original.hovertemplate == styled.hovertemplate
    assert source.layout.coloraxis.cmin == display.layout.coloraxis.cmin
    assert source.layout.coloraxis.cmax == display.layout.coloraxis.cmax


def test_comparison_keeps_ranking_errors_and_axis_ranges():
    source = go.Figure(
        go.Bar(
            x=[0.8, 0.9],
            y=["Modelo A", "Modelo B"],
            orientation="h",
            error_x={"array": [0.02, 0.04]},
            text=[0.8, 0.9],
        )
    )
    source.update_xaxes(range=[0, 1])
    source.update_layout(
        title="Comparación de modelos", meta={"atlas_caption": "Prueba independiente"}
    )
    display = present_chart(source)
    assert display.data[0].x == source.data[0].x
    assert display.data[0].y == source.data[0].y
    assert display.data[0].error_x == source.data[0].error_x
    assert display.layout.xaxis.range == (0, 1)
    assert display.layout.meta["atlas_caption"] == "Prueba independiente"


def test_bubble_and_continuous_color_encodings_are_not_replaced():
    source = go.Figure(
        go.Scatter(
            x=[1, 2],
            y=[3, 4],
            mode="markers",
            marker={
                "size": [10, 25],
                "color": [1.0, 2.0],
                "coloraxis": "coloraxis",
                "opacity": 0.4,
            },
        )
    )
    display = present_chart(source)
    assert display.data[0].marker.size == source.data[0].marker.size
    assert display.data[0].marker.color == source.data[0].marker.color
    assert display.data[0].marker.opacity == 0.4


def test_dashboard_reuses_existing_charts_and_invalidates_by_data_version():
    from atlas.controllers.visual_dashboard_controller import VisualDashboardController

    eda = ExploratoryDataAnalysis(pd.DataFrame({"x": range(20), "y": range(20)}))
    state = {"eda": eda}
    source = VisualDashboardController.chart(state, "histogram", x="x")
    assert VisualDashboardController.chart(state, "histogram", x="x") is source
    assert len(state["visual_dashboard_cache"]) == 1
    before = source.to_json()
    present_chart(source)
    assert source.to_json() == before
    eda.apply(eda.df.iloc[:15], "subset", confirm=True)
    assert VisualDashboardController.chart(state, "histogram", x="x") is not source
    assert len(source.data[0].x) == 20


def test_visual_dashboard_renders_five_real_charts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string("""
import streamlit as st
import pandas as pd
from atlas.models.eda import ExploratoryDataAnalysis
from atlas.views.components.visual_dashboard import render_visual_dashboard
if "eda" not in st.session_state:
    st.session_state.eda = ExploratoryDataAnalysis(pd.DataFrame({
        "precio": range(30), "area": range(30), "zona": ["Norte", "Sur"] * 15}))
render_visual_dashboard()
""").run(timeout=30)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 5
    app.selectbox(key="visual_group").set_value("zona").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 5
    assert len(app.session_state["eda"].df) == 30
