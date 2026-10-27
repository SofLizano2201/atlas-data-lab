"""QA local con Edge headless y CDP; no instala dependencias de navegador."""

import argparse
import base64
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect


def main() -> None:
    """Verificar tamaños reales y navegación, y guardar capturas de las tres vistas."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--all-pages", action="store_true")
    parser.add_argument(
        "--agents", action="store_true", help="Capturar además los paneles abiertos."
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("Puerto local inválido.")
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / ("atlas-mockups" if args.all_pages else "sidebar")
    output.mkdir(parents=True, exist_ok=True)
    browser = (
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe"
    )
    if not browser.is_file():
        raise RuntimeError("Esta comprobación local requiere Microsoft Edge en Windows.")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(
        prefix="atlas-sidebar-", ignore_cleanup_errors=True
    ) as profile:
        process = subprocess.Popen(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-features=PaintHolding",
                "--no-first-run",
                "--no-default-browser-check",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                f"http://127.0.0.1:{args.port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            targets = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                        targets = json.load(response)
                    if any(t.get("type") == "page" for t in targets):
                        break
                except OSError:
                    time.sleep(0.2)
            page = next(t for t in targets if t.get("type") == "page")
            with connect(page["webSocketDebuggerUrl"], max_size=20 * 1024**2) as ws:
                request_id = 0

                def command(method, params=None):
                    nonlocal request_id
                    request_id += 1
                    ws.send(
                        json.dumps({"id": request_id, "method": method, "params": params or {}})
                    )
                    while True:
                        result = json.loads(ws.recv(timeout=40))
                        if result.get("id") == request_id:
                            if "error" in result:
                                raise RuntimeError(result["error"])
                            return result.get("result", {})

                def evaluate(expression):
                    result = command(
                        "Runtime.evaluate", {"expression": expression, "returnByValue": True}
                    )
                    if "exceptionDetails" in result:
                        raise RuntimeError(result["exceptionDetails"])
                    return result["result"].get("value")

                def wait_for(expression):
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline:
                        if evaluate(expression):
                            return
                        time.sleep(0.15)
                    screenshot("failure")
                    (output / "failure-dom.html").write_text(
                        evaluate("document.querySelector('[data-testid=stSidebar]').outerHTML"),
                        encoding="utf-8",
                    )
                    raise AssertionError(f"Estado no alcanzado: {expression}")

                def screenshot(name):
                    if name != "failure":
                        wait_for(
                            "document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='notRunning'"
                        )
                    command("Page.bringToFront")
                    image = command(
                        "Page.captureScreenshot",
                        {"format": "png", "captureBeyondViewport": False, "optimizeForSpeed": True},
                    )
                    (output / f"{name}.png").write_bytes(base64.b64decode(image["data"]))

                command(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": 1672, "height": 941, "deviceScaleFactor": 1, "mobile": False},
                )
                wait_for("Boolean(document.querySelector('.st-key-atlas_nav_data button'))")
                wait_for(
                    "document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().width===260"
                )
                time.sleep(0.3)
                screenshot("expanded")
                (output / "geometry.json").write_text(
                    evaluate(
                        """JSON.stringify(['.st-key-atlas_sidebar_footer','.st-key-atlas_nav_home button'].map(selector=>{let chain=[];for(let p=document.querySelector(selector);p&&p.closest('[data-testid=stSidebar]');p=p.parentElement){let s=getComputedStyle(p),r=p.getBoundingClientRect();chain.push({tag:p.tagName,testid:p.dataset.testid,classes:p.className,x:r.x,y:r.y,width:r.width,height:r.height,display:s.display,gap:s.gap,padding:s.padding,minHeight:s.minHeight,marginTop:s.marginTop})}return chain}))"""
                    ),
                    encoding="utf-8",
                )
                evaluate("document.querySelector('.st-key-atlas_nav_data button').click()")
                wait_for("document.body.innerText.includes('Importación de datos')")
                wait_for(
                    "[...document.querySelectorAll('[role=tab]')].some(b=>b.innerText==='Dataset de ejemplo')"
                )
                evaluate(
                    "[...document.querySelectorAll('[role=tab]')].find(b=>b.innerText==='Dataset de ejemplo').click()"
                )
                wait_for(
                    "[...document.querySelectorAll('button')].some(b=>b.innerText==='Activar ejemplo')"
                )
                evaluate(
                    "[...document.querySelectorAll('button')].find(b=>b.innerText==='Activar ejemplo').click()"
                )
                wait_for("document.querySelector('.atlas-dataset-name')?.innerText==='Iris'")
                time.sleep(0.5)
                screenshot("dataset")
                cards = evaluate("""[...document.querySelectorAll('.atlas-metric-item')].map(card=>{
                    const rect=card.getBoundingClientRect();
                    return {height:rect.height,top:rect.top,
                        valueTop:card.querySelector('.atlas-card-value').getBoundingClientRect().top,
                        detailTop:card.querySelector('.atlas-metric-detail').getBoundingClientRect().top,
                        overflow:card.scrollWidth>card.clientWidth};})""")
                assert len(cards) == 3, cards
                assert max(c["height"] for c in cards) - min(c["height"] for c in cards) < 1, cards
                assert max(c["valueTop"] for c in cards) - min(c["valueTop"] for c in cards) < 1, (
                    cards
                )
                assert not any(c["overflow"] for c in cards), cards
                (output / "metric-cards.json").write_text(json.dumps(cards), encoding="utf-8")
                evaluate(
                    "document.querySelector('.atlas-metrics-row').scrollIntoView({block:'center'})"
                )
                time.sleep(0.2)
                screenshot("metric-cards")
                evaluate("document.querySelector('[data-testid=stMain]').scrollTo(0,0)")
                if args.all_pages:
                    routes = [
                        ("home", "Framework de análisis de datos", None),
                        ("data", "Importación de datos", None),
                        ("exploration", "Exploración de datos", "EDAAgent"),
                        ("preparation", "Limpieza y transformación", "EDAAgent"),
                        ("clustering", "Clustering y agrupamiento", "ClusteringAgent"),
                        (
                            "classification",
                            "ANÁLISIS DE CLASIFICACIÓN",
                            "ClassificationAgent",
                        ),
                        ("regression", "ANÁLISIS DE REGRESIÓN", "RegressionAgent"),
                        (
                            "experiments",
                            "Comparador de modelos",
                            "ModelComparisonAgent",
                        ),
                        ("reports", "Reportes y exportación", None),
                    ]
                    for slug, heading, agent in routes:
                        evaluate(
                            f"document.querySelector('.st-key-atlas_nav_{slug} button').click()"
                        )
                        wait_for(f"document.body.innerText.includes({json.dumps(heading)})")
                        time.sleep(0.6)
                        assert not evaluate(
                            "Boolean(document.querySelector('[data-testid=stException]'))"
                        )
                        screenshot(f"page-{slug}")
                        if agent and args.agents:
                            already_open = evaluate(
                                f"Boolean(document.querySelector('.st-key-panel_{agent}'))"
                            )
                            if not already_open:
                                evaluate(
                                    f"document.querySelector('.st-key-open_{agent} button').click()"
                                )
                            wait_for(f"Boolean(document.querySelector('.st-key-panel_{agent}'))")
                            time.sleep(0.5)
                            assert not evaluate(
                                "Boolean(document.querySelector('[data-testid=stException]'))"
                            )
                            if already_open:
                                # La captura anterior ya contiene este mismo panel; no hubo cambios visuales.
                                (output / f"agent-{slug}.png").write_bytes(
                                    (output / f"page-{slug}.png").read_bytes()
                                )
                            else:
                                screenshot(f"agent-{slug}")
                    print("Nueve pantallas capturadas sin errores de ejecución.", flush=True)
                evaluate("document.querySelector('.st-key-atlas_sidebar_toggle button').click()")
                wait_for(
                    "document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().width===72"
                )
                time.sleep(0.3)
                screenshot("compact")
                evaluate("document.querySelector('.st-key-atlas_nav_exploration button').click()")
                wait_for("document.body.innerText.includes('Exploración de datos')")
                evaluate("document.querySelector('.st-key-atlas_sidebar_toggle button').click()")
                wait_for(
                    "document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().width===260"
                )
                command(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True},
                )
                time.sleep(0.5)
                # Streamlit puede cerrar el menú automáticamente al entrar en vista móvil.
                evaluate(
                    "(()=>{let sidebar=document.querySelector('[data-testid=stSidebar]');if(sidebar.getBoundingClientRect().x>=0&&sidebar.getAttribute('aria-expanded')!=='false')return;let control=document.querySelector('[data-testid=stSidebarCollapsedControl], [data-testid=stExpandSidebarButton]');(control?.tagName==='BUTTON'?control:control?.querySelector('button'))?.click()})()"
                )
                wait_for(
                    "document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().x>=0"
                )
                time.sleep(0.5)
                screenshot("mobile")
                measurements = evaluate(
                    "JSON.stringify({sidebar:document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().width,viewport:innerWidth,documentWidth:document.documentElement.scrollWidth,buttonHeights:[...document.querySelectorAll('.st-key-atlas_sidebar_nav button')].map(b=>b.getBoundingClientRect().height).filter(h=>h>0)})"
                )
                (output / "measurements.json").write_text(measurements, encoding="utf-8")
                print("QA navegador: navegación, dataset y modos 260/72 px comprobados.")
                print(measurements)
                command("Browser.close")
        finally:
            try:
                with urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1) as response:
                    browser_ws = json.load(response)["webSocketDebuggerUrl"]
                with connect(browser_ws) as cleanup_ws:
                    cleanup_ws.send(json.dumps({"id": 1, "method": "Browser.close"}))
                process.wait(timeout=5)
            except (OSError, TimeoutError, subprocess.TimeoutExpired, WebSocketException):
                pass
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
