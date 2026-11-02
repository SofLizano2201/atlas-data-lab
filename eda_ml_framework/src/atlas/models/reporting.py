"""Scientific model implementation, independent of agents and Streamlit."""

from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from io import BytesIO
from typing import Any
from uuid import uuid4

import pandas as pd

from .eda import ExploratoryDataAnalysis
from .experiment import Experiment, html_report

SECTIONS = {
    "overview": "Información del dataset",
    "statistics": "Estadísticas",
    "quality": "Calidad de datos",
    "cleaning": "Historial de limpieza",
    "experiments": "Modelos y métricas",
    "decisions": "Decisiones de los agentes",
}


@dataclass
class ReportDocument:
    """Define the ReportDocument contract."""

    id: str
    title: str
    format: str
    data: bytes
    dataset_hash: str
    sections: list[str]
    created: str


def report_tables(
    eda: ExploratoryDataAnalysis,
    experiments: list[Experiment],
    decisions: list[Any],
    sections: list[str],
) -> dict[str, pd.DataFrame]:
    """Provide report tables using validated inputs."""
    if not sections or any(section not in SECTIONS for section in sections):
        raise ValueError("Seleccione secciones válidas para el reporte.")
    tables = {}
    for section in dict.fromkeys(sections):
        if section == "overview":
            table = eda.information()
        elif section == "statistics":
            table = eda.summary()
        elif section == "quality":
            table = eda.missing_values()
        elif section == "cleaning":
            table = eda.history
        elif section == "experiments":
            table = pd.DataFrame(
                [
                    {
                        "id": e.id,
                        "tipo": e.problem,
                        "algoritmo": e.name,
                        "dataset": e.dataset_id,
                        "version_modelo": e.model_version,
                        "segundos": e.seconds,
                        "configuracion": str(e.config),
                        **e.metrics,
                    }
                    for e in experiments
                ]
            )
        else:
            table = pd.DataFrame([d.model_dump() for d in decisions])
        tables[SECTIONS[section]] = table
    return tables


def _pdf(title: str, tables: dict[str, pd.DataFrame], note: str) -> bytes:
    """Provide pdf using validated inputs."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle

    buffer = BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=title,
    )
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#331832")
    styles["Heading2"].textColor = colors.HexColor("#1B5299")
    cell_style = ParagraphStyle(
        "AtlasCell", fontName="Helvetica", fontSize=8, leading=10, alignment=TA_LEFT, wordWrap="CJK"
    )

    def cell(value: Any) -> Any:
        text = str(value)
        return Paragraph(escape(text[:1500]) + ("…" if len(text) > 1500 else ""), cell_style)

    story = [
        Paragraph(escape(title), styles["Title"]),
        Paragraph(escape(note), styles["Normal"]),
        Spacer(1, 0.5 * cm),
    ]
    for name, original in tables.items():
        story.append(Paragraph(escape(name), styles["Heading2"]))
        if original.empty:
            story.append(Paragraph("Sin registros en esta sección.", styles["Normal"]))
            continue
        table = original.head(200).copy()
        if not isinstance(table.index, pd.RangeIndex):
            table.insert(0, "variable", table.index.astype(str))
        if len(original) > 200:
            story.append(
                Paragraph(
                    "Vista limitada a las primeras 200 filas; exporte CSV/Excel para consultar los datos completos.",
                    styles["Normal"],
                )
            )
        columns = list(table.columns)
        groups = (
            [columns]
            if len(columns) <= 5
            else [[columns[0], *columns[i : i + 4]] for i in range(1, len(columns), 4)]
        )
        for group in groups:
            rows = [[cell(c) for c in group]] + [
                [cell(v) for v in row] for row in table[group].itertuples(index=False, name=None)
            ]
            rendered = LongTable(
                rows, colWidths=[doc.width / len(group)] * len(group), repeatRows=1, hAlign="LEFT"
            )
            rendered.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#9FC2CC")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#694D75")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([rendered, Spacer(1, 0.4 * cm)])

    def footer(canvas: Any, document: Any) -> None:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#694D75"))
        canvas.drawString(doc.leftMargin, 0.8 * cm, "ATLAS / EDA + ML")
        canvas.drawRightString(page[0] - doc.rightMargin, 0.8 * cm, f"Página {document.page}")

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def generate_report(
    eda: ExploratoryDataAnalysis,
    experiments: list[Experiment],
    decisions: list[Any],
    sections: list[str],
    fmt: str = "html",
    title: str = "Atlas · Informe de análisis",
) -> ReportDocument:
    """Provide generate report using validated inputs."""
    if fmt not in {"html", "pdf"} or not 1 <= len(title) <= 160:
        raise ValueError("Formato o título de reporte inválido.")
    tables = report_tables(eda, experiments, decisions, sections)
    created = datetime.now(UTC).isoformat()
    note = f"Dataset: {eda.dataset_version.content_hash}; versión: {eda.dataset_version.version}. Generado: {created}. Las tablas PDF se limitan a 200 filas y 1500 caracteres por celda. Los gráficos interactivos se exportan por separado desde sus pantallas. Los experimentos de diferentes cohortes se listan, no se ordenan como una comparación."
    content = (
        html_report(title, tables, note).encode() if fmt == "html" else _pdf(title, tables, note)
    )
    return ReportDocument(
        id=uuid4().hex,
        title=title,
        format=fmt,
        data=content,
        dataset_hash=eda.dataset_version.content_hash,
        sections=list(sections),
        created=created,
    )
