"""Reportes reales, historial reproducible y contenido escapado."""

from io import BytesIO

import pytest
from pypdf import PdfReader

from framework import AnalisisDatosExploratorio, Regresion
from framework.reporting import generate_report


@pytest.mark.parametrize("fmt", ["html", "pdf"])
def test_reports(mixed, fmt):
    eda = AnalisisDatosExploratorio(mixed)
    report = generate_report(
        eda,
        [],
        [],
        ["overview", "statistics", "quality", "cleaning", "experiments", "decisions"],
        fmt,
    )
    assert report.dataset_hash == eda.dataset_version.content_hash
    assert len(report.data) > 1000
    if fmt == "pdf":
        assert report.data.startswith(b"%PDF-") and b"%%EOF" in report.data
        reader = PdfReader(BytesIO(report.data))
        assert len(reader.pages) >= 1
        assert "Atlas" in reader.pages[0].extract_text()
    else:
        assert b"<html" in report.data


def test_report_errors(mixed):
    eda = AnalisisDatosExploratorio(mixed)
    for sections in [[], ["unimplemented"]]:
        with pytest.raises(ValueError):
            generate_report(eda, [], [], sections)
    with pytest.raises(ValueError):
        generate_report(eda, [], [], ["quality"], "pickle")


def test_report_with_experiment(regression):
    engine = Regresion(regression)
    result = engine.entrenar("ridge", "target", ["a", "b"], cv=2)
    report = generate_report(engine, [result], [], ["experiments"], "pdf")
    assert report.data.startswith(b"%PDF")
    text = " ".join(p.extract_text() for p in PdfReader(BytesIO(report.data)).pages)
    assert "ridge" in text and "RMSE" in text
