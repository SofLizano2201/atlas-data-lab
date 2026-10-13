"""Figuras interactivas y estadísticas retornadas como objetos."""

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from scipy import stats

from .exceptions import FrameworkError

PALETTE = ["#1B5299", "#694D75", "#9FC2CC", "#331832", "#F1ECCE"]

# Figuras estadísticas para servidor y exportación: no requieren Tk ni escritorio.
matplotlib.use("Agg")


def style(fig: go.Figure) -> go.Figure:
    """Aplicar tema consistente a una figura Plotly."""
    fig.update_layout(
        paper_bgcolor="#F1ECCE",
        plot_bgcolor="#F1ECCE",
        font_color="#331832",
        colorway=PALETTE,
        margin={"l": 30, "r": 30, "t": 40, "b": 30},
    )
    return fig


def plot(
    df: pd.DataFrame,
    tipo: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    **params: Any,
) -> Any:
    """Catálogo de gráficos EDA; limitar matrices y muestras costosas."""
    if tipo == "missing":
        return style(
            px.imshow(
                df.head(1000).isna().astype(int).T,
                aspect="auto",
                color_continuous_scale=PALETTE[:3],
            )
        )
    if tipo in {"heatmap", "correlation"}:
        return style(
            px.imshow(
                df.select_dtypes(include=np.number).corr(method=params.get("metodo", "pearson")),
                color_continuous_scale=PALETTE[:3],
                zmin=-1,
                zmax=1,
            )
        )
    if tipo == "pairplot":
        columns = list(df.select_dtypes(include=np.number).columns[:8])
        if not columns:
            raise FrameworkError("Pairplot requiere variables numéricas.")
        return sns.pairplot(df[columns].sample(min(500, len(df)), random_state=42))
    if x is None or x not in df:
        raise FrameworkError("Seleccione una variable para el gráfico.")
    if tipo in {"histogram", "box", "violin", "kde", "qq", "outliers"} and not (
        pd.api.types.is_numeric_dtype(df[x])
    ):
        raise FrameworkError("Este gráfico requiere una variable numérica.")
    if tipo in {"kde", "qq"}:
        values = df[x].dropna()
        if len(values) < 3 or values.nunique() < 2:
            raise FrameworkError("Se requieren 3+ valores y variación.")
        fig, ax = plt.subplots()
        if tipo == "qq":
            stats.probplot(values, plot=ax)
        else:
            sns.kdeplot(values, ax=ax, color=PALETTE[0])
        return fig
    if tipo in {"count", "bar", "pie"}:
        counts = df[x].astype(str).value_counts().head(30).rename_axis(x).reset_index(name="n")
        if tipo == "pie":
            if df[x].nunique() > 8:
                raise FrameworkError("Gráfico circular limitado a 8 categorías; use barras.")
            return style(px.pie(counts, names=x, values="n", color_discrete_sequence=PALETTE))
        return style(px.bar(counts, x=x, y="n", color_discrete_sequence=PALETTE))
    if tipo == "outliers":
        tipo = "box"
    functions = {
        "histogram": px.histogram,
        "box": px.box,
        "violin": px.violin,
        "scatter": px.scatter,
        "line": px.line,
    }
    if tipo not in functions:
        raise FrameworkError("Tipo de gráfico desconocido.")
    kwargs: dict[str, Any] = {"x": x, "color": color, "color_discrete_sequence": PALETTE}
    if y:
        kwargs["y"] = y
    if tipo in {"box", "violin"}:
        kwargs.pop("x")
        kwargs["y"] = x
    if tipo in {"scatter", "line"} and not y:
        raise FrameworkError("Seleccione también una variable Y.")
    return style(functions[tipo](df.head(10000), **kwargs))


def before_after(before: pd.DataFrame, after: pd.DataFrame, column: str) -> go.Figure:
    """Comparación de distribuciones de una variable transformada."""
    data = pd.concat(
        [before[[column]].assign(estado="Antes"), after[[column]].assign(estado="Después")]
    )
    return style(
        px.histogram(
            data,
            x=column,
            color="estado",
            barmode="overlay",
            opacity=0.6,
            color_discrete_sequence=PALETTE,
        )
    )
