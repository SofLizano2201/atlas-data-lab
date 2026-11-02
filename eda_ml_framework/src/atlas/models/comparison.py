"""Comparison operations reuse existing cohort validation and recorded metrics."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .experiment import compare
from .visualization import PALETTE, style


class ExperimentComparison:
    compare = staticmethod(compare)

    @staticmethod
    def groups(experiments):
        """Expose compatible cohorts before a user selects comparison candidates."""
        groups = {}
        for result in experiments:
            key = f"{result.problem} · {result.dataset_id[:8]} · {result.cohort[:8]}"
            groups.setdefault(key, []).append(result)
        return groups

    @staticmethod
    def chart(experiments, metric, kind="ranking"):
        """Render stored metrics only; never retrain or invent stability evidence."""
        table = compare(experiments).reset_index()
        if metric not in table or not pd.api.types.is_numeric_dtype(table[metric]):
            raise ValueError("Selecciona una métrica numérica disponible.")
        if kind == "cost":
            return style(
                px.scatter(
                    table,
                    x="segundos",
                    y=metric,
                    color="modelo",
                    symbol="modelo",
                    hover_name="id",
                    color_discrete_sequence=PALETTE,
                    labels={"segundos": "Tiempo total de entrenamiento (s)", "modelo": "Modelo"},
                ),
                f"Rendimiento y coste · {metric}",
            )
        if kind == "runtime":
            return style(
                px.bar(
                    table,
                    x="id",
                    y="segundos",
                    color="modelo",
                    pattern_shape="modelo",
                    color_discrete_sequence=PALETTE,
                    labels={
                        "id": "Experimento",
                        "segundos": "Tiempo total (s)",
                        "modelo": "Modelo",
                    },
                ),
                "Tiempo de ejecución registrado",
            )
        if kind == "cv":
            if len({e.config.get("cv") for e in experiments}) > 1:
                raise ValueError(
                    "Usa el mismo número de folds para comparar distribuciones de validación."
                )
            rows = [
                {"experiment": e.id, "model": e.name, "fold": fold + 1, "score": value}
                for e in experiments
                for fold, value in enumerate(e.details.get("cv_scores", []))
            ]
            if not rows:
                raise ValueError(
                    "Estos experimentos no guardan puntuaciones por fold. No se simula una distribución ni estabilidad."
                )
            return style(
                px.box(
                    pd.DataFrame(rows),
                    x="model",
                    y="score",
                    color="model",
                    points="all",
                    hover_data=["experiment", "fold"],
                    color_discrete_sequence=PALETTE,
                    labels={
                        "model": "Modelo",
                        "score": "Puntuación CV (RMSE negativo en regresión)",
                    },
                ),
                "Variación entre folds · no equivale a estabilidad entre semillas",
            )
        if kind in {"roc", "pr"}:
            if any(e.problem != "clasificacion" for e in experiments):
                raise ValueError("ROC y Precision-Recall requieren clasificación.")
            figure = go.Figure()
            for result in experiments:
                points = result.details.get(kind)
                if points is None:
                    raise ValueError(
                        "La superposición requiere clasificación binaria con puntuaciones válidas en todos los modelos."
                    )
                horizontal, vertical = (
                    (points[0], points[1]) if kind == "roc" else (points[1], points[0])
                )
                figure.add_scatter(
                    x=horizontal, y=vertical, mode="lines", name=f"{result.name} · {result.id}"
                )
            style(
                figure,
                "Curvas ROC comparadas" if kind == "roc" else "Curvas Precision-Recall comparadas",
            )
            figure.update_xaxes(
                title="Tasa de falsos positivos" if kind == "roc" else "Recall", range=[0, 1]
            )
            figure.update_yaxes(
                title="Tasa de verdaderos positivos" if kind == "roc" else "Precision", range=[0, 1]
            )
            return figure
        return style(
            px.bar(
                table.sort_values(metric),
                y="id",
                x=metric,
                color="modelo",
                pattern_shape="modelo",
                orientation="h",
                color_discrete_sequence=PALETTE,
                labels={"id": "Experimento", "modelo": "Modelo"},
            ),
            f"Comparación registrada · {metric}",
        )

    @staticmethod
    def rank(experiments, metric, direction):
        table = compare(experiments)
        if (
            direction not in {"min", "max"}
            or metric not in table
            or not pd.api.types.is_numeric_dtype(table[metric])
        ):
            raise ValueError("Select an available metric and min/max direction.")
        if metric.startswith("CV") and len({e.config.get("cv") for e in experiments}) > 1:
            raise ValueError(
                "Las métricas de validación cruzada requieren el mismo número de folds para esta comparación."
            )
        values = table[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            raise ValueError("The selected metric has no finite values.")
        best = values.idxmin() if direction == "min" else values.idxmax()
        return table, best, float(values.loc[best])

    @staticmethod
    def summary(experiments):
        return {
            "count": len(experiments),
            "average_seconds": float(np.mean([e.seconds for e in experiments]))
            if experiments
            else 0.0,
        }
