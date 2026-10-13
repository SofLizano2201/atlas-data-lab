"""Resultados independientes de la interfaz y exportaciones seguras."""

from dataclasses import dataclass, field
from html import escape
from io import BytesIO
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from .exceptions import FrameworkError


@dataclass
class Experiment:
    """Una ejecución con métricas, configuración y partición identificables."""

    name: str
    problem: str
    metrics: dict[str, float]
    seconds: float
    model: Any
    predictions: pd.DataFrame
    config: dict[str, Any]
    cohort: str
    details: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex[:10])

    def model_bytes(self) -> bytes:
        """Serializar solo para exportación; nunca se cargan modelos externos."""
        buffer = BytesIO()
        payload = (
            self.model
            if self.problem != "clustering"
            else {
                "modelo": self.model,
                "preprocessor": self.details["preprocessor"],
                "config": self.config,
                "problema": self.problem,
            }
        )
        joblib.dump(payload, buffer)
        return buffer.getvalue()


def compare(experiments: list[Experiment]) -> pd.DataFrame:
    """Comparar ejecuciones sobre exactamente la misma cohorte y problema."""
    if not experiments:
        raise FrameworkError("Seleccione al menos un experimento.")
    if len({(e.problem, e.cohort) for e in experiments}) != 1:
        raise FrameworkError("Compare solo el mismo dataset, variables y partición de prueba.")
    return pd.DataFrame(
        [{"id": e.id, "modelo": e.name, "segundos": e.seconds, **e.metrics} for e in experiments]
    ).set_index("id")


def export_frame(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    """Exportar datos; neutralizar fórmulas de hojas de cálculo en textos."""
    safe = df.copy()
    for c in safe.select_dtypes(include=["object", "string"]).columns:
        safe[c] = safe[c].map(
            lambda v: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
        )
    if fmt == "csv":
        return safe.to_csv(index=False).encode("utf-8-sig")
    if fmt == "json":
        return df.to_json(orient="records", date_format="iso").encode()
    buffer = BytesIO()
    if fmt == "xlsx":
        safe.to_excel(buffer, index=False)
    elif fmt == "parquet":
        df.to_parquet(buffer, index=False)
    else:
        raise FrameworkError("Formato de exportación no admitido.")
    return buffer.getvalue()


def html_report(title: str, tables: dict[str, pd.DataFrame], note: str = "") -> str:
    """Generar un reporte HTML autónomo con contenido escapado."""
    body = "".join(f"<h2>{escape(k)}</h2>{v.to_html(escape=True)}" for k, v in tables.items())
    return (
        "<!doctype html><html lang='es'><meta charset='utf-8'><title>"
        + escape(title)
        + "</title><style>body{font:16px system-ui;background:#F1ECCE;color:#331832;"
        "margin:3rem}table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid"
        " #9FC2CC}h2{color:#1B5299}</style><h1>"
        + escape(title)
        + "</h1><p>"
        + escape(note)
        + "</p>"
        + body
        + "</html>"
    )
