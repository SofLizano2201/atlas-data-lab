"""QA de reportes con datos públicos; ejecutar con el extra pdf_qa instalado."""

from pathlib import Path

import pymupdf
from pypdf import PdfReader
from sklearn.datasets import load_diabetes, load_iris

from framework import AnalisisDatosExploratorio, Regresion
from framework.reporting import generate_report


def main() -> None:
    """Generar, reabrir y renderizar un PDF de prueba, sin incluir datos del usuario."""
    output = Path(__file__).resolve().parents[1] / "artifacts" / "reports"
    output.mkdir(parents=True, exist_ok=True)
    iris = AnalisisDatosExploratorio(load_iris(as_frame=True).frame)
    data = load_diabetes(as_frame=True).frame.head(80)
    regression = Regresion(data)
    trained = regression.entrenar("ridge", "target", list(data.columns[:4]), cv=2)
    report = generate_report(
        iris,
        [trained],
        [],
        ["overview", "statistics", "quality", "cleaning", "experiments", "decisions"],
        "pdf",
    )
    path = output / "atlas-validation.pdf"
    path.write_bytes(report.data)
    reader = PdfReader(path)
    text = " ".join(page.extract_text() for page in reader.pages)
    assert "ridge" in text and "RMSE" in text and "Atlas" in text
    with pymupdf.open(path) as document:
        for page in document:
            for block in page.get_text("blocks"):
                x0, y0, x1, y1 = block[:4]
                assert (
                    x0 >= 0 and y0 >= 0 and x1 <= page.rect.width + 1 and y1 <= page.rect.height + 1
                )
        for i in sorted({0, len(document) - 1}):
            document[i].get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(
                output / f"page-{i + 1}.png"
            )
    print(f"PDF reabierto, texto verificado y {len(reader.pages)} páginas dentro de sus límites.")


if __name__ == "__main__":
    main()
