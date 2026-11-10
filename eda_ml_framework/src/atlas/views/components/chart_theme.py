"""Presentation-only Plotly styling. Never mutate a scientific/source figure."""

import re
from copy import deepcopy
from html import unescape

import plotly.graph_objects as go

DISPLAY_COLORS = [
    "#1B5299",
    "#62BFCE",
    "#A15CC1",
    "#C4AFD1",
    "#E6D4A0",
    "#507FB5",
    "#185563",
    "#331832",
]


def chart_title(source):
    text = re.sub(r"<br\s*/?>", " ", source.layout.title.text or "Análisis de datos")
    return unescape(re.sub(r"<[^>]+>", "", text))


def copy_chart(source):
    # Figure.to_dict() can encode arrays as binary dictionaries in Plotly 6.
    # Trace contracts retain arrays for selection and the existing CSV exporter.
    return go.Figure(
        data=[deepcopy(trace.to_plotly_json()) for trace in source.data],
        layout=deepcopy(source.layout.to_plotly_json()),
        frames=[deepcopy(frame.to_plotly_json()) for frame in source.frames],
    )


def present_chart(source):
    """Keep trace data, domains, ranges, masks and scientific annotations intact."""
    display = copy_chart(source)
    title = chart_title(source)
    metadata = dict(display.layout.meta or {})
    metadata["atlas_display_title"] = title
    has_legend = any(t.name and t.showlegend is not False for t in display.data)
    display.update_layout(
        title_text=None,
        meta=metadata,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": "#331832"},
        height=320,
        autosize=True,
        margin={"l": 24, "r": 24, "t": 16, "b": 32},
        barcornerradius=2,
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
            "font": {"size": 10},
            "bgcolor": "rgba(0,0,0,0)",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        hoverlabel={
            "bgcolor": "#FFFCF7",
            "bordercolor": "#A68AB4",
            "font": {"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": "#331832"},
            "namelength": -1,
        },
        modebar={"bgcolor": "rgba(255,255,255,.9)", "color": "#694D75", "activecolor": "#1B5299"},
        separators=",.",
        dragmode="zoom",
        uirevision=title,
    )
    display.update_xaxes(
        automargin=True,
        title_standoff=16,
        tickfont_size=10,
        showline=True,
        linecolor="#B7B2B9",
        showgrid=False,
        zeroline=False,
        ticks="outside",
        ticklen=4,
        tickcolor="#DDE5ED",
    )
    display.update_yaxes(
        automargin=True,
        title_standoff=16,
        tickfont_size=10,
        showline=True,
        linecolor="#B7B2B9",
        showgrid=False,
        gridcolor="#E5EBF1",
        griddash="dot",
        gridwidth=1,
        zeroline=False,
        ticks="",
    )
    if has_legend and display.layout.showlegend is not False:
        display.update_layout(margin_r=90)
    groups = {}
    for trace in display.data:
        marker = getattr(trace, "marker", None)
        color = getattr(marker, "color", None)
        # Keep continuous color scales, bubble sizes and special noise/mask encodings.
        if (
            marker is not None
            and isinstance(color, str)
            and not getattr(marker, "coloraxis", None)
            and color.lower() != "#827887"
        ):
            groups.setdefault(color, DISPLAY_COLORS[len(groups) % len(DISPLAY_COLORS)])
            marker.color = groups[color]
        if trace.type == "histogram":
            trace.marker.line.update(color="#FFFFFF", width=1)
            if len(display.data) == 1:
                trace.marker.color = "#1B5299"
                trace.opacity = 1
            display.update_layout(bargap=0.05)
            # Full numeric labels avoid ambiguous 5k/10k abbreviations.
            if trace.orientation != "h":
                display.update_xaxes(tickformat=",~g", nticks=7)
                display.update_yaxes(nticks=6)
            if not trace.hovertemplate:
                # Plotly computes bins; do not substitute counts or change binning.
                trace.hovertemplate = "%{x}<br>Valor: %{y}<extra>%{fullData.name}</extra>"
        elif trace.type in {"scatter", "scattergl"} and trace.mode and "markers" in trace.mode:
            # Preserve continuous colors, bubble sizes, symbols and opacity encodings.
            trace.marker.line.update(width=0.3, color="#FFFFFF")
        elif trace.type == "bar":
            trace.marker.line.update(width=0)
        elif trace.type == "heatmap":
            trace.update(xgap=1, ygap=1)
    heatmaps = [t for t in display.data if t.type == "heatmap"]
    if heatmaps:
        count = len(heatmaps[0].z) if heatmaps[0].z is not None else 0
        display.update_layout(height=max(320, min(1000, count * 26 + 120)))
        display.update_xaxes(showgrid=False, ticks="", tickangle=-35)
        display.update_yaxes(showgrid=False, ticks="")
        display.update_coloraxes(
            colorbar={"thickness": 10, "len": 0.8, "outlinewidth": 0, "tickfont": {"size": 11}}
        )
        if display.layout.coloraxis.cmin == -1 and display.layout.coloraxis.cmax == 1:
            display.update_coloraxes(colorscale=[[0, "#A15CC1"], [0.5, "#FFFCF7"], [1, "#1B5299"]])
    # Preserve space allocated to scientific subplot grids and long rankings.
    if len(list(display.select_xaxes())) > 1:
        display.update_layout(height=source.layout.height or 620)
    if any(t.type == "bar" and t.orientation == "h" for t in display.data):
        rows = max(
            (len(t.y) for t in display.data if t.type == "bar" and t.y is not None), default=0
        )
        display.update_layout(height=max(300, min(1100, 110 + rows * 30)))
    return display


def chart_config(scroll_zoom=False):
    return {
        "displaylogo": False,
        "displayModeBar": "hover",
        "responsive": True,
        "scrollZoom": scroll_zoom,
        "modeBarButtonsToRemove": ["autoScale2d", "toggleSpikelines"],
        "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "atlas_grafico"},
    }
