"""Generar evidencia visual local con cálculos reales y capturarla en Edge headless."""

import base64
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, make_blobs
from websockets.sync.client import connect

from framework import Clasificacion, Clustering, Regresion
from framework.visualization import before_after, plot


def main() -> None:
    """Galería de datos públicos/sintéticos; no accede a sesiones de usuarios."""
    output = Path(__file__).resolve().parents[1] / "artifacts" / "charts"
    output.mkdir(parents=True, exist_ok=True)
    iris = load_iris(as_frame=True).frame
    iris["Especie"] = iris.target.map(dict(enumerate(load_iris().target_names)))
    features = list(iris.columns[:4])
    classifier = Clasificacion(iris)
    classification = classifier.entrenar("logistic", "target", features, cv=3)
    diabetes = load_diabetes(as_frame=True).frame
    regressor = Regresion(diabetes)
    regression = regressor.entrenar("linear_multiple", "target", list(diabetes.columns[:-1]), cv=3)
    blobs, _ = make_blobs(n_samples=160, centers=3, cluster_std=0.7, random_state=42)
    cluster = Clustering(pd.DataFrame(blobs, columns=["x", "y"]))
    grouping = cluster.entrenar("dbscan", ["x", "y"], eps=0.28, min_samples=5)
    representation = cluster.reducir(["x", "y"])
    figures = [
        plot(iris, "scatter", features[0], features[2], "Especie"),
        plot(iris, "histogram", features[0], color="Especie"),
        plot(iris[features], "correlation"),
        classifier.grafico_resultado(classification, "confusion_normalized"),
        classifier.grafico_resultado(classification, "roc"),
        classifier.grafico_resultado(classification, "pr"),
        regressor.grafico_resultado(regression),
        regressor.grafico_resultado(regression, "residuals"),
        cluster.visualizar_representacion(representation["datos"], grouping.details["labels"]),
        cluster.silhouette_plot(grouping),
        plot(iris, "count", "Especie"),
        before_after(iris, iris[iris[features[0]] < 7], features[0]),
    ]
    for kind in ["qq", "kde", "pairplot", "box", "violin"]:
        figures.append(
            plot(
                iris[features[:2] + ["Especie"]],
                kind,
                features[0],
                color="Especie" if kind != "qq" else None,
            )
        )
    figures.extend(
        [
            plot(iris, "missing_summary"),
            cluster.dendrogram(["x", "y"]),
            classifier.grafico_resultado(classification, "confusion_normalized"),
            classifier.grafico_resultado(classification, "pr"),
            regressor.grafico_resultado(regression, "histogram"),
            regressor.grafico_resultado(regression, "qq"),
        ]
    )
    content = [
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Atlas · Verificación de gráficos</title>",
        (
            "<style>body{margin:0;padding:24px;background:#F1ECCE;color:#331832;font-family:Segoe UI,sans-serif}"
            "h1{font-size:24px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}"
            "section{min-width:0;background:#FBF9EF;border:1px solid #9FC2CC;border-radius:10px;padding:12px}"
            "@media(max-width:800px){body{padding:8px}main{grid-template-columns:1fr}}</style>"
        ),
        "<h1>ATLAS · Revisión de gráficos calculados</h1><p>Iris, Diabetes y agrupamiento sintético. Sin datos privados.</p><main>",
    ]
    for index, fig in enumerate(figures):
        content.append(
            f"<section id='chart-{index}'>"
            + fig.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"displaylogo": False, "responsive": True},
            )
            + "</section>"
        )
    content.append("</main>")
    gallery = output / "gallery.html"
    gallery.write_text("".join(content), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            [{"title": fig.layout.title.text, "traces": len(fig.data)} for fig in figures],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    capture(gallery, output, len(figures))
    print(
        f"Verificados {len(figures)} gráficos Plotly interactivos, incluidos gráficos estadísticos. {gallery}"
    )


def capture(gallery: Path, output: Path, count: int) -> None:
    """Capturas de escritorio, diagnóstico y móvil con el Edge ya instalado."""
    browser = Path(os.environ["PROGRAMFILES(X86)"]) / "Microsoft/Edge/Application/msedge.exe"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="atlas-charts-", ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                gallery.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            for _ in range(100):
                try:
                    with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                        page = next(p for p in json.load(response) if p.get("type") == "page")
                    break
                except (OSError, StopIteration):
                    time.sleep(0.2)
            else:
                raise RuntimeError("Edge no inició para la revisión de gráficos.")
            with connect(page["webSocketDebuggerUrl"], max_size=30 * 1024**2) as ws:
                sequence = 0

                def command(method, params=None):
                    nonlocal sequence
                    sequence += 1
                    ws.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
                    while True:
                        message = json.loads(ws.recv(timeout=40))
                        if message.get("id") == sequence:
                            if "error" in message:
                                raise RuntimeError(message["error"])
                            return message.get("result", {})

                def evaluate(expression):
                    result = command(
                        "Runtime.evaluate", {"expression": expression, "returnByValue": True}
                    )
                    if "exceptionDetails" in result:
                        raise RuntimeError(result["exceptionDetails"])
                    return result["result"].get("value")

                command(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False},
                )
                for _ in range(100):
                    if evaluate(
                        f"[...document.querySelectorAll('.js-plotly-plot')].filter(p=>p._fullLayout).length==={count}"
                    ):
                        break
                    time.sleep(0.2)
                else:
                    raise AssertionError("No se renderizaron todos los gráficos.")
                for name, target, width in [
                    ("desktop", 0, 1440),
                    ("matrices", 2, 1440),
                    ("modelos", 6, 1440),
                    ("mobile", 0, 390),
                ]:
                    command(
                        "Emulation.setDeviceMetricsOverride",
                        {
                            "width": width,
                            "height": 1000,
                            "deviceScaleFactor": 1,
                            "mobile": width < 800,
                        },
                    )
                    evaluate(f"document.getElementById('chart-{target}').scrollIntoView()")
                    time.sleep(0.7)
                    command("Page.bringToFront")
                    png = command(
                        "Page.captureScreenshot",
                        {"format": "png", "captureBeyondViewport": False, "optimizeForSpeed": True},
                    )
                    (output / f"{name}.png").write_bytes(base64.b64decode(png["data"]))
                command("Browser.close")
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
