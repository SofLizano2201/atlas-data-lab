"""Scientific model implementation, independent of agents and Streamlit."""

from textwrap import wrap
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from atlas.core.exceptions import FrameworkError

PALETTE = [
    "#1B5299",
    "#694D75",
    "#9FC2CC",
    "#331832",
    "#507FB5",
    "#9672A4",
    "#185563",
    "#54416A",
    "#3668A6",
    "#B09BBA",
    "#557D88",
    "#4A2F50",
    "#244776",
    "#83628F",
    "#BDD5DB",
    "#482445",
    "#7097C7",
    "#B29BBE",
    "#3C6570",
    "#6F527D",
]
SEQUENTIAL = [[0, "#F1ECCE"], [0.35, "#9FC2CC"], [1, "#1B5299"]]
DIVERGING = [[0, "#1B5299"], [0.5, "#F1ECCE"], [1, "#694D75"]]
NOISE_COLOR = "#827887"
MAX_VISUAL_POINTS = 10000
SYMBOLS = [
    "circle",
    "diamond",
    "square",
    "cross",
    "triangle-up",
    "x",
    "star",
    "triangle-down",
    "circle-open",
    "diamond-open",
    "square-open",
    "triangle-up-open",
]


def categorical_colors(values: Any, max_categories: int = 12) -> dict[str, str]:
    """Provide categorical colors using validated inputs."""
    labels = sorted(set(map(str, values)))
    if len(labels) > max_categories:
        raise FrameworkError(
            f"Demasiadas categorías de color (máximo {max_categories}). Filtra o selecciona otra variable."
        )
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


def note(fig: go.Figure, text: str) -> go.Figure:
    """Provide note using validated inputs."""
    metadata = dict(fig.layout.meta or {})
    existing = metadata.get("atlas_caption", "")
    metadata["atlas_caption"] = (
        (existing + " " + text).strip() if text not in existing else existing
    )
    fig.update_layout(meta=metadata)
    return fig


def style(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Provide style using validated inputs."""
    heading = (title or fig.layout.title.text or "Análisis de datos").replace("<br>", " ")
    title_lines = wrap(heading, 30)
    legend_entries = {
        trace.name for trace in fig.data if trace.name and trace.showlegend is not False
    }
    legend_space = max(100, 70 + 22 * ((len(legend_entries) + 2) // 3))
    fig.update_layout(
        template=go.layout.Template(layout={"colorway": PALETTE, "plot_bgcolor": "#F5F2F8"}),
        title={
            "text": "<br>".join(title_lines),
            "x": 0,
            "xanchor": "left",
            "y": 0.95,
            "yanchor": "top",
            "font": {"size": 20, "family": "Manrope, Inter, sans-serif", "color": "#331832"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F5F2F8",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": "#331832"},
        colorway=PALETTE,
        modebar={"bgcolor": "#FFFFFF", "color": "#694D75", "activecolor": "#1B5299"},
        height=max(500, 300 + 55 + 24 * len(title_lines) + legend_space),
        autosize=True,
        hovermode="closest",
        dragmode="zoom",
        uirevision=heading,
        margin={"l": 65, "r": 30, "t": 55 + 24 * len(title_lines), "b": legend_space},
        legend={
            "title_text": "",
            "orientation": "h",
            "x": 0,
            "y": -0.20,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 12},
        },
        hoverlabel={
            "font_size": 13,
            "bgcolor": "#331832",
            "font_color": "#F1ECCE",
            "bordercolor": "#694D75",
        },
        separators=",.",
        scene={
            "bgcolor": "#FDFBF4",
            "xaxis": {
                "gridcolor": "rgba(159,194,204,0.33)",
                "backgroundcolor": "#F5F2F8",
                "showbackground": True,
            },
            "yaxis": {
                "gridcolor": "rgba(159,194,204,0.33)",
                "backgroundcolor": "#F5F2F8",
                "showbackground": True,
            },
            "zaxis": {
                "gridcolor": "rgba(159,194,204,0.33)",
                "backgroundcolor": "#F5F2F8",
                "showbackground": True,
            },
        },
    )
    fig.update_xaxes(
        automargin=True,
        title_standoff=16,
        showgrid=False,
        zeroline=False,
        linecolor="#9FC2CC",
        tickfont_size=12,
    )
    fig.update_yaxes(
        automargin=True,
        title_standoff=16,
        gridcolor="rgba(105,77,117,.13)",
        zeroline=False,
        tickfont_size=12,
    )
    histograms = [trace for trace in fig.data if trace.type == "histogram"]
    if len(histograms) == 1:
        histograms[0].marker.color = "#694D75"
    for i, trace in enumerate(fig.data):
        if trace.type in {"scatter", "scattergl"}:
            if trace.mode and "markers" in trace.mode:
                trace.marker.update(
                    size=trace.marker.size if trace.marker.size is not None else 7,
                    opacity=trace.marker.opacity if trace.marker.opacity is not None else 0.72,
                    line={"width": 0.45, "color": "#694D75"},
                )
            if trace.mode and "lines" in trace.mode:
                if trace.line.width is None:
                    trace.line.width = 2.5
                if trace.line.dash is None:
                    trace.line.dash = [
                        "solid",
                        "dash",
                        "dot",
                        "dashdot",
                        "longdash",
                        "longdashdot",
                    ][i % 6]
        if trace.type == "bar":
            trace.marker.line.update(width=0.5, color="#F1ECCE")
        if trace.type == "heatmap":
            trace.update(xgap=2, ygap=2)
        if not trace.hovertemplate and trace.type in {"scatter", "scattergl", "bar"}:
            trace.hovertemplate = "%{x}<br>%{y}<extra>%{fullData.name}</extra>"
    return fig


def chart_frame(fig: go.Figure, visible_only: bool = True) -> pd.DataFrame:
    """Export actual trace inputs; histogram/box/violin rows are observations, not bins."""
    rows = []
    for index, trace in enumerate(fig.data):
        if visible_only and trace.visible in {False, "legendonly"}:
            continue
        if isinstance(trace.meta, dict) and trace.meta.get("atlas_role") == "mask":
            continue
        label = trace.name or f"Serie {index + 1}"
        if trace.type == "heatmap":
            for i, values in enumerate(trace.z):
                for j, value in enumerate(values):
                    rows.append(
                        {
                            "Serie": label,
                            "Tipo": trace.type,
                            "X": trace.x[j] if trace.x is not None else j,
                            "Y": trace.y[i] if trace.y is not None else i,
                            "Valor": value,
                        }
                    )
        else:
            axes = {axis: getattr(trace, axis, None) for axis in ("x", "y", "z")}
            if trace.type == "pie":
                axes = {"Categoría": trace.labels, "Valor": trace.values}
            length = max((len(values) for values in axes.values() if values is not None), default=0)
            for point in range(length):
                row = {"Serie": label, "Tipo": trace.type, "Punto": point}
                for axis, values in axes.items():
                    if values is not None and point < len(values):
                        row[axis.upper() if axis in {"x", "y", "z"} else axis] = values[point]
                if trace.customdata is not None and point < len(trace.customdata):
                    row["Contexto"] = str(trace.customdata[point])
                rows.append(row)
    return pd.DataFrame(rows)


def diagnostic_bars(table: pd.DataFrame, kind: str, limit: int = 15) -> go.Figure:
    """Render existing coefficients/importances without fitting or recomputing them."""
    if not 1 <= limit <= 50 or table.empty:
        raise FrameworkError("Selecciona entre 1 y 50 variables con resultados disponibles.")
    if kind == "importance":
        shown = table.nlargest(limit, "importancia").sort_values("importancia")
        fig = style(
            px.bar(
                shown,
                x="importancia",
                y="variable",
                orientation="h",
                error_x="std" if "std" in shown else None,
                color_discrete_sequence=PALETTE,
                labels={"variable": "Variable", "importancia": "Cambio en puntuación al permutar"},
            ),
            "¿Qué variables afectan más a las predicciones?",
        )
        return note(
            fig,
            "Importancia calculada sobre datos de prueba. Las barras de error muestran variabilidad de las permutaciones, no intervalos de confianza. No uses esta prueba para seleccionar variables y después reportar el mismo resultado como independiente.",
        )
    if kind != "coefficients":
        raise FrameworkError("Selecciona coeficientes o importancia de variables.")
    data = (
        table.rename_axis("Variable")
        .reset_index()
        .melt(id_vars="Variable", var_name="Componente", value_name="Coeficiente")
    )
    selected = (
        data.assign(magnitude=data.Coeficiente.abs())
        .nlargest(limit, "magnitude")
        .sort_values("Coeficiente")
    )
    selected["Signo"] = np.where(selected.Coeficiente >= 0, "Positivo", "Negativo")
    fig = style(
        px.bar(
            selected,
            x="Coeficiente",
            y="Variable",
            orientation="h",
            color="Signo",
            pattern_shape="Componente",
            color_discrete_map={"Positivo": PALETTE[0], "Negativo": PALETTE[1]},
            hover_data=["Componente"],
        ),
        "¿En qué dirección se relaciona cada variable con la predicción?",
    )
    fig.add_vline(x=0, line_color=PALETTE[3], line_width=1)
    return note(
        fig,
        "Coeficientes del modelo entrenado después del preprocesamiento. Sus magnitudes dependen de las unidades y del escalamiento; no prueban causalidad.",
    )


def plot(
    df: pd.DataFrame,
    kind: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    **params: Any,
) -> Any:
    """Provide plot using validated inputs."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise FrameworkError(
            "No hay registros para representar. Carga datos antes de crear el gráfico."
        )
    if kind == "missing_summary":
        table = pd.DataFrame(
            {
                "Variable": df.columns,
                "Faltantes": df.isna().sum().values,
                "Porcentaje": df.isna().mean().values * 100,
            }
        ).sort_values("Porcentaje")
        fig = px.bar(
            table,
            x="Porcentaje",
            y="Variable",
            orientation="h",
            text="Porcentaje",
            custom_data=["Faltantes"],
            color="Porcentaje",
            color_continuous_scale=SEQUENTIAL,
            range_color=[0, 100],
        )
        style(fig, "¿Dónde falta información?")
        fig.update_traces(
            texttemplate="%{x:.1f}%",
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Faltantes: %{customdata[0]:,}<br>Porcentaje: %{x:.2f}%<extra></extra>",
        )
        fig.update_layout(
            height=max(400, min(1100, len(table) * 28 + 160)),
            showlegend=False,
            coloraxis_showscale=False,
        )
        fig.update_xaxes(
            title="Porcentaje de registros sin información",
            range=[0, max(5, min(110, table.Porcentaje.max() * 1.15))],
        )
        fig.update_yaxes(title="")
        return note(
            fig,
            "Las variables con más información faltante aparecen arriba. Se revisaron todos los registros; no se eliminó ni completó ningún valor.",
        )
    if kind == "missing":
        fig = px.imshow(
            df.head(1000).isna().astype(int).T,
            aspect="auto",
            zmin=0,
            zmax=1,
            color_continuous_scale=[
                [0, "#F1ECCE"],
                [0.499, "#F1ECCE"],
                [0.5, "#1B5299"],
                [1, "#1B5299"],
            ],
            labels={"x": "Fila", "y": "Variable", "color": "Estado"},
        )
        fig.update_coloraxes(colorbar={"tickvals": [0, 1], "ticktext": ["Presente", "Faltante"]})
        style(fig, "Patrón de valores faltantes")
        return note(fig, f"{min(1000, len(df)):,} de {len(df):,} filas · azul = faltante")
    if kind in {"heatmap", "correlation"}:
        numeric = df.select_dtypes(include=np.number)
        selected = params.get("columns")
        if selected is not None:
            if not selected or any(column not in numeric for column in selected):
                raise FrameworkError("Selecciona al menos una columna numérica para la matriz.")
            numeric = numeric[list(dict.fromkeys(selected))]
        if numeric.empty:
            raise FrameworkError("La correlación requiere variables numéricas.")
        method = params.get("method", params.get("metodo", "pearson"))
        if method not in {"pearson", "spearman", "kendall"}:
            raise FrameworkError("Selecciona Pearson, Spearman o Kendall.")
        matrix = numeric.iloc[:, :30].corr(method=method)
        fig = px.imshow(
            matrix,
            color_continuous_scale=DIVERGING,
            zmin=-1,
            zmax=1,
            color_continuous_midpoint=0,
            text_auto=".2f" if len(matrix) <= 12 else False,
            labels={"color": "Correlación"},
            aspect="auto",
        )
        style(fig, f"Matriz de correlación · {method.title()}")
        if matrix.isna().any().any():
            fig.add_heatmap(
                z=np.where(matrix.isna(), 1, np.nan),
                x=matrix.columns,
                y=matrix.index,
                colorscale=[[0, "#D9D0DD"], [1, "#D9D0DD"]],
                showscale=False,
                text=np.where(matrix.isna(), "N/D", ""),
                texttemplate="%{text}" if len(matrix) <= 12 else None,
                hovertemplate="%{y} / %{x}<br>No calculable: datos insuficientes o constantes<extra></extra>",
                meta={"atlas_role": "mask"},
            )
        fig.update_xaxes(tickangle=-40)
        fig.data[0].hovertemplate = "%{y} / %{x}<br>Correlación: %{z:.3f}<extra></extra>"
        fig.update_layout(height=max(520, min(1000, len(matrix) * 30 + 200)))
        return note(
            fig,
            "−1: negativa · 0: sin asociación lineal/monótona · +1: positiva. No implica causalidad. N/D: no calculable."
            if numeric.shape[1] <= 30
            else "Se muestran las primeras 30 variables numéricas.",
        )
    if kind == "pairplot":
        columns = list(df.select_dtypes(include=np.number).columns[:6])
        if not columns:
            raise FrameworkError("Se necesitan variables numéricas para comparar relaciones.")
        data = df.sample(min(500, len(df)), random_state=42)
        groups = (
            data[color].astype("string").fillna("(Sin dato)")
            if color
            else pd.Series("Observaciones", index=data.index)
        )
        colors = categorical_colors(groups)
        fig = make_subplots(
            rows=len(columns), cols=len(columns), horizontal_spacing=0.04, vertical_spacing=0.05
        )
        for row, vertical in enumerate(columns, 1):
            for col, horizontal in enumerate(columns, 1):
                for label, series_color in colors.items():
                    subset = data[groups == label]
                    common = {
                        "name": label,
                        "legendgroup": label,
                        "showlegend": row == 1 and col == 1,
                    }
                    if row == col:
                        trace = go.Histogram(
                            x=subset[horizontal], marker_color=series_color, opacity=0.65, **common
                        )
                    else:
                        trace = go.Scattergl(
                            x=subset[horizontal],
                            y=subset[vertical],
                            mode="markers",
                            marker={"color": series_color, "size": 4, "opacity": 0.55},
                            customdata=subset.index.astype(str),
                            hovertemplate=f"{horizontal}: %{{x}}<br>{vertical}: %{{y}}<br>Registro: %{{customdata}}<extra>{label}</extra>",
                            **common,
                        )
                    fig.add_trace(trace, row=row, col=col)
                if row == len(columns):
                    fig.update_xaxes(title_text=horizontal, row=row, col=col)
                if col == 1:
                    fig.update_yaxes(title_text=vertical, row=row, col=col)
        style(fig, "¿Qué relaciones aparecen entre las variables?")
        fig.update_layout(height=max(620, len(columns) * 190), barmode="overlay")
        return note(
            fig,
            f"Muestra reproducible de {len(data)} de {len(df)} registros; hasta seis variables. Diagonal: distribución de cada variable.",
        )
    if (
        x is None
        or x not in df
        or (y is not None and y not in df)
        or (color is not None and color not in df)
    ):
        raise FrameworkError("Seleccione variables existentes para el gráfico.")
    if kind in {"histogram", "box", "violin", "kde", "qq", "outliers"} and (
        not pd.api.types.is_numeric_dtype(df[x])
    ):
        raise FrameworkError("Este gráfico requiere una variable numérica.")
    if kind in {"kde", "qq"}:
        values = df[x].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 3 or values.nunique() < 2:
            raise FrameworkError("Se requieren 3+ valores finitos y variación.")
        if kind == "qq":
            (theoretical, observed), (slope, intercept, _) = stats.probplot(values)
            positions = np.linspace(
                0, len(values) - 1, min(MAX_VISUAL_POINTS, len(values)), dtype=int
            )
            fig = go.Figure(
                go.Scattergl(
                    x=theoretical[positions],
                    y=observed[positions],
                    mode="markers",
                    name="Valores observados",
                    hovertemplate="Cuantil normal: %{x:.3f}<br>Valor observado: %{y}<extra></extra>",
                )
            )
            fig.add_scatter(
                x=[theoretical[0], theoretical[-1]],
                y=[slope * theoretical[0] + intercept, slope * theoretical[-1] + intercept],
                mode="lines",
                name="Referencia normal",
                line={"color": PALETTE[1], "dash": "dash"},
            )
            style(fig, f"¿La distribución se aproxima a una normal? · {x}")
            fig.update_xaxes(title="Cuantiles teóricos de la normal")
            fig.update_yaxes(title=f"Cuantiles observados · {x}")
            return note(
                fig,
                f"Se muestran {len(positions):,} de {len(values):,} cuantiles; el ajuste usa todos los valores finitos. Las desviaciones respecto a la línea muestran diferencias de forma; este gráfico no sustituye una prueba de hipótesis.",
            )
        groups = (
            df[color].astype("string").fillna("(Sin dato)")
            if color
            else pd.Series("Observaciones", index=df.index)
        )
        colors = categorical_colors(groups)
        fig = go.Figure()
        skipped = []
        for label, series_color in colors.items():
            subset = df.loc[groups == label, x].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset) < 3 or subset.nunique() < 2:
                skipped.append(label)
                continue
            estimate = stats.gaussian_kde(subset)
            bandwidth = float(estimate.factor * subset.std())
            grid = np.linspace(subset.min() - 3 * bandwidth, subset.max() + 3 * bandwidth, 256)
            fig.add_scatter(
                x=grid,
                y=estimate(grid),
                name=label,
                mode="lines",
                fill="tozeroy",
                line={"color": series_color},
                opacity=0.65,
                hovertemplate=f"{x}: %{{x}}<br>Densidad estimada: %{{y:.4f}}<extra>{label}</extra>",
            )
        if not fig.data:
            raise FrameworkError(
                "No hay grupos con suficientes valores y variación para estimar densidad."
            )
        style(fig, f"¿Cómo se distribuyen los valores? · {x}")
        fig.update_xaxes(title=x)
        fig.update_yaxes(title="Densidad estimada")
        return note(
            fig,
            "Cada curva estima una distribución con área 1; no representa cantidades de registros."
            + (
                " Grupos omitidos por datos insuficientes: " + ", ".join(skipped) if skipped else ""
            ),
        )
    if kind in {"count", "bar", "pie"}:
        values = df[x].astype("string").fillna("(Sin dato)")
        count = "_atlas_count" + ("_" if x == "_atlas_count" else "")
        counts = values.value_counts().rename_axis(x).reset_index(name=count)
        if kind == "pie":
            if len(counts) > 8:
                raise FrameworkError("Gráfico circular limitado a 8 categorías; use barras.")
            fig = px.pie(
                counts,
                names=x,
                values=count,
                color=x,
                color_discrete_map=categorical_colors(counts[x]),
                hole=0.35,
            )
            fig.update_traces(
                textinfo="percent",
                hovertemplate="%{label}<br>%{value:,} registros (%{percent})<extra></extra>",
            )
            return style(fig, f"Proporción por {x}")
        shown = counts.head(30).sort_values(count, ascending=True)
        kwargs = (
            {"color": x, "color_discrete_map": categorical_colors(shown[x])}
            if len(shown) <= 12
            else {}
        )
        fig = px.bar(
            shown,
            y=x,
            x=count,
            orientation="h",
            text=count,
            labels={count: "Registros"},
            color_discrete_sequence=PALETTE,
            **kwargs,
        )
        style(fig, f"Registros por {x}")
        fig.update_layout(showlegend=False, height=max(440, min(1100, len(shown) * 30 + 170)))
        fig.update_xaxes(rangemode="tozero")
        if len(counts) > 30:
            note(
                fig,
                f"30 de {len(counts)} categorías · {int(shown[count].sum()):,} de {len(df):,} registros",
            )
        return fig
    kind = "box" if kind == "outliers" else kind
    functions = {
        "histogram": px.histogram,
        "box": px.box,
        "violin": px.violin,
        "scatter": px.scatter,
        "line": px.line,
    }
    if kind not in functions:
        raise FrameworkError("Tipo de gráfico desconocido.")
    if kind in {"scatter", "line"} and y is None:
        raise FrameworkError("Seleccione también una variable Y.")
    maximum = params.get("max_points", MAX_VISUAL_POINTS)
    if not isinstance(maximum, int) or not 100 <= maximum <= 50000:
        raise FrameworkError("El límite visual admite entre 100 y 50.000 puntos.")
    data = (
        df.sample(maximum, random_state=42)
        if len(df) > maximum and kind not in {"histogram"}
        else df
    ).copy()
    data_index = "_atlas_record"
    while data_index in data:
        data_index += "_"
    data[data_index] = data.index.astype(str)
    kwargs: dict[str, Any] = {
        "x": x,
        "color": color,
        "color_discrete_sequence": PALETTE,
        "color_continuous_scale": SEQUENTIAL,
    }
    if not color:
        kwargs["color_discrete_sequence"] = [
            {"histogram": "#694D75", "violin": "#9672A4", "scatter": "#331832"}.get(
                kind, PALETTE[0]
            )
        ]
    if kind == "histogram":
        bins = params.get("nbins", 30)
        if not isinstance(bins, int) or not 5 <= bins <= 150:
            raise FrameworkError("El histograma admite de 5 a 150 intervalos.")
        kwargs["nbins"] = bins
    facet = params.get("facet")
    if facet:
        if facet not in data or data[facet].nunique(dropna=False) > 6:
            raise FrameworkError("La faceta debe existir y tener como máximo seis categorías.")
        kwargs["facet_col"] = facet
        kwargs["facet_col_wrap"] = 3
    for axis in ("x", "y"):
        if params.get(f"log_{axis}"):
            variable = x if axis == "x" else y
            if (
                variable is None
                or not pd.api.types.is_numeric_dtype(data[variable])
                or (data[variable].dropna() <= 0).any()
            ):
                raise FrameworkError(
                    "La escala logarítmica requiere valores numéricos estrictamente positivos."
                )
            kwargs[f"log_{axis}"] = True
    if color and (not pd.api.types.is_numeric_dtype(df[color]) or df[color].nunique() <= 12):
        group = "_atlas_color"
        while group in data.columns:
            group += "_"
        data[group] = data[color].astype("string").fillna("(Sin dato)").astype(str)
        kwargs.update(color=group, labels={group: color})
        kwargs["color_discrete_map"] = categorical_colors(
            df[color].astype("string").fillna("(Sin dato)")
        )
        kwargs["category_orders"] = {group: sorted(kwargs["color_discrete_map"])}
        if kind == "histogram":
            kwargs.update(
                pattern_shape=group, pattern_shape_sequence=["", "/", ".", "x", "+", "-", "|", "\\"]
            )
        if kind == "scatter":
            kwargs.update(symbol=group, symbol_sequence=SYMBOLS)
        if kind == "line":
            kwargs["line_dash"] = group
    elif color and kind != "scatter":
        raise FrameworkError(
            "Este gráfico requiere hasta 12 categorías de color. Para color continuo use dispersión."
        )
    if y is not None and kind in {"scatter", "line"}:
        kwargs["y"] = y
    if kind in {"box", "violin"}:
        kwargs.pop("x")
        kwargs["y"] = x
        kwargs["points"] = "all" if params.get("show_points") else "outliers"
        if kind == "violin":
            kwargs["box"] = True
    if kind == "line":
        data = data.sort_values(x, kind="stable")
    if kind != "scatter":
        kwargs.pop("color_continuous_scale")
    else:
        kwargs["custom_data"] = [data_index]
    titles = {
        "histogram": f"Distribución de {x}",
        "box": f"Distribución y valores extremos · {x}",
        "violin": f"Forma de la distribución · {x}",
        "scatter": f"{y} frente a {x}",
        "line": f"{y} según {x}",
    }
    fig = style(functions[kind](data, **kwargs), titles[kind])
    if kind == "scatter":
        # Original source indices remain inspectable after visual sampling.
        for trace in fig.data:
            trace.hovertemplate = f"{x}: %{{x}}<br>{y}: %{{y}}<br>Registro: %{{customdata[0]}}<extra>%{{fullData.name}}</extra>"
    if kind == "histogram":
        fig.update_layout(barmode="group", yaxis_title="Frecuencia (registros)", bargap=0.1)
        fig.update_traces(opacity=0.9, bingroup="atlas_histogram")
        fig.update_traces(marker_line_color="#694D75", marker_line_width=0.6)
        fig.update_traces(
            hovertemplate=f"{x}: %{{x}}<br>Frecuencia: %{{y:,}} registros<extra>%{{fullData.name}}</extra>"
        )
    guides = {
        "histogram": "La altura de cada barra indica cuántos registros caen en un intervalo. Cambiar los intervalos cambia el detalle visual, no los datos.",
        "box": "La caja muestra el 50% central y su línea indica la mediana. Los puntos señalados son valores poco habituales, no necesariamente errores.",
        "violin": "Las zonas anchas representan mayor concentración de valores. La caja interior ayuda a localizar la mediana y el rango central.",
        "scatter": "Cada punto es un registro. Una relación visual entre variables no demuestra causalidad.",
        "line": "Las observaciones se unen en orden de X; no se han sumado ni promediado registros con el mismo valor.",
    }
    note(fig, guides[kind])
    if len(df) > len(data):
        note(
            fig,
            f"Muestra visual reproducible: {len(data):,} de {len(df):,} registros. No cambia las estadísticas del dataset completo.",
        )
    return fig


def before_after(before: pd.DataFrame, after: pd.DataFrame, column: str) -> go.Figure:
    """Provide before after using validated inputs."""
    data = pd.concat(
        [before[[column]].assign(estado="Antes"), after[[column]].assign(estado="Después")]
    )
    fig = style(
        px.histogram(
            data,
            x=column,
            color="estado",
            barmode="overlay",
            histnorm="percent",
            opacity=0.6,
            color_discrete_map={"Antes": PALETTE[0], "Después": PALETTE[1]},
            category_orders={"estado": ["Antes", "Después"]},
        ),
        f"Antes y después · {column}",
    )
    fig.update_layout(yaxis_title="Porcentaje dentro de cada versión (%)")
    fig.update_traces(bingroup="atlas_distribution")
    return note(
        fig,
        f"Antes: {len(before):,} filas · Después: {len(after):,} filas · intervalos compartidos",
    )
