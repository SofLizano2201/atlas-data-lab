"""Scientific model implementation, independent of agents and Streamlit."""

from dataclasses import dataclass, field
from html import escape
from io import BytesIO
from math import isfinite
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from atlas.core.compatibility import CompatibleConfiguration
from atlas.core.exceptions import FrameworkError

from .provenance import fingerprint, runtime_versions


@dataclass
class Experiment:
    """Define the Experiment contract."""

    def get_metrics(self) -> dict[str, float]:
        """Return a defensive copy of previously calculated metrics."""
        return dict(self.metrics)

    def metrics_for_context(self) -> dict[str, float | None]:
        """Serialize recorded metrics without exporting NaN or infinity to a provider."""
        return {
            key: float(value) if isfinite(float(value)) else None
            for key, value in self.metrics.items()
        }

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
    dataset_id: str = ""
    model_version: int = 1
    library_versions: dict[str, str] = field(default_factory=runtime_versions)

    def __post_init__(self):
        self.config = CompatibleConfiguration(self.config)

    def record_provenance(self, df: pd.DataFrame, number: int) -> None:
        """Provide record provenance using validated inputs."""
        self.dataset_id = fingerprint(df)
        self.model_version = number
        self.config["dataset_id"] = self.dataset_id
        self.config["model_version"] = self.model_version
        self.config["library_versions"] = self.library_versions
        for name in ["x_train", "x_test"]:
            if name in self.details:
                self.config[f"{name}_index"] = [str(i) for i in self.details[name].index]

    def model_bytes(self) -> bytes:
        """Provide model bytes using validated inputs."""
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
    """Compare recorded experiments only when their problem and cohort agree."""
    if not experiments:
        raise FrameworkError("Seleccione al menos un experimento.")
    from .methodological_validation import MethodologicalValidation

    MethodologicalValidation.enforce(MethodologicalValidation.comparison(experiments))
    if len({(e.problem, e.cohort) for e in experiments}) != 1:
        raise FrameworkError("Compare solo el mismo dataset, variables y partición de prueba.")
    return pd.DataFrame(
        [{"id": e.id, "modelo": e.name, "segundos": e.seconds, **e.metrics} for e in experiments]
    ).set_index("id")


def export_frame(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    """Provide export frame using validated inputs."""
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


TABLE_CSS = "table{border-collapse:separate;border-spacing:0;background:#FFFFFF;border:1px solid #DFE4EB;border-radius:12px;overflow:hidden;font-size:14px}td,th{padding:10px 14px;border:0;border-bottom:1px solid #DFE4EB;text-align:left}thead th{background:#EAE4F0;color:#331832}tbody tr:nth-child(even){background:#F1F5F9}tbody tr:hover{background:#E7EEF7}tbody th{color:#694D75;font-weight:600}"


def html_report(title: str, tables: dict[str, pd.DataFrame], note: str = "") -> str:
    """Provide html report using validated inputs."""
    body = "".join((f"<h2>{escape(k)}</h2>{v.to_html(escape=True)}" for k, v in tables.items()))
    return (
        "<!doctype html><html lang='es'><meta charset='utf-8'><title>"
        + escape(title)
        + "</title><style>body{font:16px system-ui;background:#F1ECCE;color:#331832;margin:3rem}h2{color:#1B5299}"
        + TABLE_CSS
        + "</style><h1>"
        + escape(title)
        + "</h1><p>"
        + escape(note)
        + "</p>"
        + body
        + "</html>"
    )
