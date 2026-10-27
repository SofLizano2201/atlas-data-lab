"""Estándar visual Atlas: escalas explícitas, categorías distinguibles y figuras legibles."""

from textwrap import wrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from .exceptions import FrameworkError

# Colores de marca y variaciones oscuras: crema se reserva para fondos y puntos medios.
PALETTE = ["#1B5299", "#694D75", "#287C87", "#331832", "#507FB5", "#9672A4", "#185563", "#54416A"]
SEQUENTIAL = [[0, "#F1ECCE"], [0.35, "#9FC2CC"], [1, "#1B5299"]]
DIVERGING = [[0, "#1B5299"], [0.5, "#F1ECCE"], [1, "#694D75"]]
NOISE_COLOR = "#827887"
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

matplotlib.rcParams.update(
    {
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "figure.facecolor": "#FBF9EF",
        "axes.facecolor": "#FBF9EF",
        "text.color": "#331832",
        "axes.labelcolor": "#331832",
        "xtick.color": "#331832",
        "ytick.color": "#331832",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 160,
    }
)


def categorical_colors(values: Any) -> dict[str, str]:
    """Mapa reproducible para una misma colección, sin colores de fondo como categorías."""
    labels = sorted(set(map(str, values)))
    if len(labels) > 12:
        raise FrameworkError(
            "Demasiadas categorías de color (máximo 12). Filtra o selecciona otra variable."
        )
    return {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(labels)}


def note(fig: go.Figure, text: str) -> go.Figure:
    """Advertencia visible también en la exportación HTML."""
    fig.update_layout(margin_t=140)
    fig.add_annotation(
        text="<br>".join(wrap(text, 58)),
        x=0,
        y=1.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "#694D75"},
    )
    return fig


def style(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Tema común sin sobrescribir escalas, rangos ni codificación semántica."""
    heading = title or fig.layout.title.text or "Análisis de datos"
    fig.update_layout(
        template="plotly_white",
        title={
            "text": "<br>".join(wrap(heading, 45)),
            "x": 0,
            "xanchor": "left",
            "y": 0.95,
            "yanchor": "top",
            "font": {"size": 18},
        },
        paper_bgcolor="#FBF9EF",
        plot_bgcolor="#FBF9EF",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": "#331832"},
        colorway=PALETTE,
        height=520,
        autosize=True,
        margin={"l": 65, "r": 35, "t": 85, "b": 110},
        legend={
            "title_text": "",
            "orientation": "h",
            "x": 0,
            "y": -0.22,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 12},
        },
        hoverlabel={"font_size": 13},
        separators=",.",
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
    # Diferenciar líneas además de su color. Las referencias conservan su estilo explícito.
    for i, trace in enumerate(fig.data):
        if trace.type in {"scatter", "scattergl"}:
            if trace.mode and "markers" in trace.mode:
                trace.marker.update(size=7, opacity=0.78, line={"width": 0.5, "color": "#FBF9EF"})
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
    return fig


def plot(
    df: pd.DataFrame,
    tipo: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    **params: Any,
) -> Any:
    """Construir gráficos con títulos, unidades implícitas en nombres y límites explícitos."""
    if tipo == "missing":
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
    if tipo in {"heatmap", "correlation"}:
        numeric = df.select_dtypes(include=np.number)
        if numeric.empty:
            raise FrameworkError("La correlación requiere variables numéricas.")
        method = params.get("metodo", "pearson")
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
            )
        fig.update_xaxes(tickangle=-40)
        fig.update_layout(height=max(520, min(1000, len(matrix) * 30 + 200)))
        return note(
            fig,
            "−1: negativa · 0: sin asociación lineal/monótona · +1: positiva. No implica causalidad. N/D: no calculable."
            if numeric.shape[1] <= 30
            else "Se muestran las primeras 30 variables numéricas.",
        )
    if tipo == "pairplot":
        columns = list(df.select_dtypes(include=np.number).columns[:6])
        if not columns:
            raise FrameworkError("Pairplot requiere variables numéricas.")
        data = df.sample(min(500, len(df)), random_state=42).copy()
        grouping = {}
        if color is not None:
            if color not in data:
                raise FrameworkError("La variable de color no existe.")
            group = "_atlas_group"
            while group in data:
                group += "_"
            data[group] = data[color].astype("string").fillna("(Sin dato)").astype(str)
            grouping = {
                "hue": group,
                "palette": categorical_colors(data[group]),
                "markers": ["o", "D", "s", "P", "^", "X", "*", "v", "<", ">", "p", "h"][
                    : data[group].nunique()
                ],
            }
        grid = sns.pairplot(
            data,
            vars=columns,
            **grouping,
            diag_kind="hist",
            plot_kws={"alpha": 0.6, "s": 18, **({} if grouping else {"color": PALETTE[0]})},
            diag_kws={} if grouping else {"color": PALETTE[0]},
        )
        grid.figure.suptitle(
            f"Relaciones entre variables · {len(data)} filas · hasta 6 variables", y=1.02
        )
        for legend in grid.figure.legends:
            legend.set_title(color or "")
        return grid
    if (
        x is None
        or x not in df
        or (y is not None and y not in df)
        or (color is not None and color not in df)
    ):
        raise FrameworkError("Seleccione variables existentes para el gráfico.")
    if tipo in {
        "histogram",
        "box",
        "violin",
        "kde",
        "qq",
        "outliers",
    } and not pd.api.types.is_numeric_dtype(df[x]):
        raise FrameworkError("Este gráfico requiere una variable numérica.")
    if tipo in {"kde", "qq"}:
        values = df[x].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 3 or values.nunique() < 2:
            raise FrameworkError("Se requieren 3+ valores finitos y variación.")
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        if tipo == "qq":
            stats.probplot(values, plot=ax)
            for line, line_color in zip(ax.lines, PALETTE):
                line.set_color(line_color)
            ax.set(
                xlabel="Cuantiles teóricos de la normal",
                ylabel=f"Cuantiles observados · {x}",
                title=f"Q-Q normal · {x}",
            )
        else:
            if color:
                sample = df[[x]].copy()
                group = "_atlas_group"
                while group in sample:
                    group += "_"
                sample[group] = df[color].astype("string").fillna("(Sin dato)").astype(str)
                sample[x] = sample[x].replace([np.inf, -np.inf], np.nan)
                valid = sample.groupby(group)[x].agg(["count", "nunique"])
                valid = valid.index[(valid["count"] >= 3) & (valid["nunique"] > 1)]
                sample = sample[sample[group].isin(valid)]
                if sample.empty:
                    raise FrameworkError(
                        "KDE por grupo requiere 3+ valores y variación en cada grupo mostrado."
                    )
                sns.kdeplot(
                    data=sample,
                    x=x,
                    hue=group,
                    palette=categorical_colors(sample[group]),
                    common_norm=False,
                    ax=ax,
                    fill=True,
                    alpha=0.25,
                )
                if ax.get_legend() is not None:
                    ax.get_legend().set_title(color)
                ax.text(
                    0,
                    -0.22,
                    "Cada curva tiene área 1; grupos insuficientes se omiten.",
                    transform=ax.transAxes,
                    fontsize=9,
                )
            else:
                sns.kdeplot(values, ax=ax, color=PALETTE[0], fill=True, alpha=0.3)
            ax.set(xlabel=x, ylabel="Densidad estimada", title=f"Distribución suavizada · {x}")
        ax.grid(alpha=0.15)
        return fig
    if tipo in {"count", "bar", "pie"}:
        values = df[x].astype("string").fillna("(Sin dato)")
        count = "_atlas_count" + ("_" if x == "_atlas_count" else "")
        counts = values.value_counts().rename_axis(x).reset_index(name=count)
        if tipo == "pie":
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
    kind = "box" if tipo == "outliers" else tipo
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
    data = df.head(10000).copy()
    kwargs: dict[str, Any] = {
        "x": x,
        "color": color,
        "color_discrete_sequence": PALETTE,
        "color_continuous_scale": SEQUENTIAL,
    }
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
    if kind == "line":
        data = data.sort_values(x, kind="stable")
    # Solo scatter admite color continuo; los otros constructores no aceptan esa opción.
    if kind != "scatter":
        kwargs.pop("color_continuous_scale")
    titles = {
        "histogram": f"Distribución de {x}",
        "box": f"Distribución y valores extremos · {x}",
        "violin": f"Forma de la distribución · {x}",
        "scatter": f"{y} frente a {x}",
        "line": f"{y} según {x}",
    }
    fig = style(functions[kind](data, **kwargs), titles[kind])
    if kind == "histogram":
        fig.update_layout(barmode="group", yaxis_title="Frecuencia (registros)", bargap=0.1)
        fig.update_traces(opacity=0.9, bingroup="atlas_histogram")
    if len(df) > len(data):
        note(fig, f"Se muestran las primeras {len(data):,} de {len(df):,} filas.")
    return fig


def before_after(before: pd.DataFrame, after: pd.DataFrame, column: str) -> go.Figure:
    """Histogramas porcentuales con límites de intervalos comunes, sin ocultar cambios de tamaño."""
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
