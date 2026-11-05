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
    parser.add_argument("--trained-results", action="store_true")
    parser.add_argument("--chart-interactions", action="store_true")
    parser.add_argument(
        "--agents", action="store_true", help="Capturar además los paneles abiertos."
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("Puerto local inválido.")
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / ("atlas-2.0" if args.all_pages else "sidebar")
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
                wait_for("Boolean(document.querySelector('.st-key-atlas_intelligence_panel'))")
                evaluate(
                    "document.querySelector('.atlas-page-title').scrollIntoView({block:'start'})"
                )
                screenshot("intelligence-desktop")
                panel_geometry = evaluate(
                    "(()=>{let p=document.querySelector('.st-key-atlas_intelligence_panel').getBoundingClientRect();return {x:p.x,width:p.width,right:p.right,viewport:innerWidth};})()"
                )
                assert (
                    panel_geometry["width"] > 200
                    and panel_geometry["right"] <= panel_geometry["viewport"] + 1
                )
                (output / "intelligence-geometry.json").write_text(
                    json.dumps(panel_geometry), encoding="utf-8"
                )
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
                        ("home", "ATLAS Dashboard", None),
                        ("data", "Importación de datos", None),
                        ("exploration", "Exploración EDA", "EDAAgent"),
                        ("preparation", "Limpieza y transformación", "EDAAgent"),
                        ("clustering", "Clustering y agrupamiento", "ClusteringAgent"),
                        (
                            "classification",
                            "Clasificación",
                            "ClassificationAgent",
                        ),
                        ("regression", "Regresión", "RegressionAgent"),
                        ("history", "Experimentos", "ModelComparisonAgent"),
                        (
                            "experiments",
                            "Comparador de modelos",
                            "ModelComparisonAgent",
                        ),
                        ("reports", "Reportes y exportación", None),
                    ]
                    for slug, heading, agent in routes:
                        evaluate(
                            "[...document.querySelectorAll('[data-testid=stDialog] button')].find(button=>button.innerText==='Cerrar')?.click()"
                        )
                        evaluate(
                            f"document.querySelector('.st-key-atlas_nav_{slug} button').click()"
                        )
                        wait_for(
                            f"document.querySelector('.atlas-page-title')?.textContent==={json.dumps(heading)}"
                        )
                        time.sleep(0.6)
                        assert not evaluate(
                            "Boolean(document.querySelector('[data-testid=stException]'))"
                        )
                        screenshot(f"page-{slug}")
                        if args.trained_results and slug in {"classification", "regression"}:
                            problem = "clasificacion" if slug == "classification" else "regresion"
                            evaluate(
                                f"document.querySelector('.st-key-train_{problem} button').click()"
                            )
                            wait_for("Boolean(document.querySelector('.atlas-result-grid'))")
                            assert not evaluate(
                                "Boolean(document.querySelector('[data-testid=stException]'))"
                            )
                            evaluate(
                                "document.querySelector('.atlas-result-context').scrollIntoView({block:'start'})"
                            )
                            time.sleep(0.5)
                            screenshot(f"results-{slug}")
                            if args.chart_interactions and slug == "regression":
                                card = (
                                    "[class*=st-key-chart_card_summary_][class*=_actual_predicted]"
                                )
                                evaluate(
                                    f"document.querySelector('{card}').scrollIntoView({{block:'center'}})"
                                )
                                evaluate(
                                    f"[...document.querySelectorAll('{card} button')].find(b=>b.innerText.includes('Explorar y descargar')).click()"
                                )
                                wait_for(
                                    "Boolean(document.querySelector('[class*=st-key-chart_inspect_summary_] input'))"
                                )
                                evaluate(
                                    "document.querySelector('[class*=st-key-chart_inspect_summary_] input').click()"
                                )
                                plot = f"document.querySelector('{card} .js-plotly-plot')"
                                wait_for(f"{plot}?._fullLayout?.dragmode==='select'")
                                evaluate(
                                    "[...document.querySelectorAll('button')].find(b=>b.innerText.includes('Explorar y descargar') && b.getAttribute('aria-expanded')==='true')?.click()"
                                )
                                evaluate(
                                    f"document.querySelector('{card}').scrollIntoView({{block:'center'}})"
                                )
                                bounds = evaluate(
                                    f"(()=>{{let p={plot},r=p.getBoundingClientRect(),x=p._fullLayout.xaxis,y=p._fullLayout.yaxis;return {{x0:r.x+x._offset+x._length*.15,x1:r.x+x._offset+x._length*.85,y0:r.y+y._offset+y._length*.15,y1:r.y+y._offset+y._length*.85}};}})()"
                                )
                                command(
                                    "Input.dispatchMouseEvent",
                                    {"type": "mouseMoved", "x": bounds["x0"], "y": bounds["y0"]},
                                )
                                command(
                                    "Input.dispatchMouseEvent",
                                    {
                                        "type": "mousePressed",
                                        "x": bounds["x0"],
                                        "y": bounds["y0"],
                                        "button": "left",
                                        "clickCount": 1,
                                    },
                                )
                                for step in range(1, 8):
                                    command(
                                        "Input.dispatchMouseEvent",
                                        {
                                            "type": "mouseMoved",
                                            "x": bounds["x0"]
                                            + (bounds["x1"] - bounds["x0"]) * step / 7,
                                            "y": bounds["y0"]
                                            + (bounds["y1"] - bounds["y0"]) * step / 7,
                                            "button": "left",
                                            "buttons": 1,
                                        },
                                    )
                                command(
                                    "Input.dispatchMouseEvent",
                                    {
                                        "type": "mouseReleased",
                                        "x": bounds["x1"],
                                        "y": bounds["y1"],
                                        "button": "left",
                                        "clickCount": 1,
                                    },
                                )
                                wait_for(
                                    f"document.querySelector('{card}')?.innerText.includes('Observaciones seleccionadas')"
                                )
                                screenshot("chart-selection")
                                before_range = evaluate(f"{plot}._fullLayout.xaxis.range")
                                (output / "chart-toolbar.json").write_text(
                                    json.dumps(
                                        evaluate(
                                            f"[...{plot}.querySelectorAll('.modebar-btn')].map(a=>({{tag:a.tagName,title:a.getAttribute('data-title'),label:a.getAttribute('aria-label')}}))"
                                        )
                                    ),
                                    encoding="utf-8",
                                )
                                evaluate(
                                    f"[...{plot}.querySelectorAll('.modebar-btn')].find(a=>/zoom in|acercar|aumentar/i.test(a.getAttribute('data-title') || a.getAttribute('aria-label') || '')).click()"
                                )
                                wait_for(
                                    f"JSON.stringify({plot}._fullLayout.xaxis.range)!=={json.dumps(json.dumps(before_range, separators=(',', ':')))}"
                                )
                                evaluate(
                                    f"[...{plot}.querySelectorAll('.modebar-btn')].find(a=>/^pan$|desplaz/i.test(a.getAttribute('data-title') || a.getAttribute('aria-label') || '')).click()"
                                )
                                wait_for(f"{plot}._fullLayout.dragmode==='pan'")
                                command(
                                    "Browser.setDownloadBehavior",
                                    {"behavior": "allow", "downloadPath": str(output)},
                                )
                                started_download = time.time()
                                evaluate(
                                    f"[...{plot}.querySelectorAll('.modebar-btn')].find(a=>/download|descargar|png/i.test(a.getAttribute('data-title') || a.getAttribute('aria-label') || '')).click()"
                                )
                                for _ in range(60):
                                    downloaded = [
                                        path
                                        for path in output.glob("atlas_grafico*.png")
                                        if path.stat().st_mtime >= started_download
                                        and path.stat().st_size > 1000
                                    ]
                                    if downloaded:
                                        break
                                    time.sleep(0.15)
                                assert downloaded, "La descarga PNG no produjo un archivo real."
                                (output / "chart-interactions.json").write_text(
                                    json.dumps(
                                        {
                                            "selection": True,
                                            "zoom": True,
                                            "pan": True,
                                            "png": downloaded[0].name,
                                        }
                                    ),
                                    encoding="utf-8",
                                )
                        if agent and args.agents:
                            already_open = evaluate(
                                f"Boolean(document.querySelector('.st-key-panel_{agent}'))"
                            )
                            if not already_open:
                                evaluate(
                                    f"document.querySelector('.st-key-open_{agent} button').click()"
                                )
                            wait_for(f"Boolean(document.querySelector('.st-key-panel_{agent}'))")
                            (output / "dialog-desktop.json").write_text(
                                json.dumps(
                                    evaluate(
                                        "(()=>{let chain=[];for(let p=document.querySelector('[role=dialog]');p&&p!==document.body;p=p.parentElement){let r=p.getBoundingClientRect(),s=getComputedStyle(p);chain.push({tag:p.tagName,testid:p.dataset.testid,role:p.getAttribute('role'),classes:p.className,width:r.width,maxWidth:s.maxWidth,overflow:s.overflow});}return chain;})()"
                                    )
                                ),
                                encoding="utf-8",
                            )
                            time.sleep(0.5)
                            visible = evaluate(
                                "document.querySelector('[data-testid=stDialog]')?.innerText || ''"
                            )
                            assert not any(
                                text in visible
                                for text in [
                                    "pending_confirmation",
                                    "analysis.configure",
                                    "Parámetros JSON",
                                    "request_id",
                                ]
                            ), visible
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
                            if slug == "exploration":
                                evaluate(
                                    "document.querySelector('[class*=st-key-idea_EDAAgent_] button').click()"
                                )
                                wait_for(
                                    "Boolean(document.querySelector('.atlas-assistant-recommendation'))"
                                )
                                time.sleep(0.4)
                                screenshot("assistant-recommendation")
                                evaluate(
                                    "document.querySelector('.st-key-assistant_technical_EDAAgent input').click()"
                                )
                                wait_for(
                                    "Boolean(document.querySelector('.st-key-technical_content_EDAAgent'))"
                                )
                                assert not evaluate(
                                    "Boolean(document.querySelector('[data-testid=stException]'))"
                                )
                                evaluate(
                                    "document.querySelector('.st-key-technical_content_EDAAgent').scrollIntoView({block:'start'})"
                                )
                                screenshot("assistant-technical")
                                evaluate(
                                    "document.querySelector('.st-key-assistant_technical_EDAAgent input').click()"
                                )
                                evaluate(
                                    "document.querySelector('.st-key-back_EDAAgent button').click()"
                                )
                                wait_for(
                                    "Boolean(document.querySelector('.atlas-assistant-section'))"
                                )
                            evaluate(
                                f"document.querySelector('.st-key-close_{agent} button').click()"
                            )
                            wait_for("!document.querySelector('[data-testid=stDialog]')")
                    print(
                        f"{len(routes)} pantallas capturadas sin errores de ejecución.", flush=True
                    )
                if args.all_pages:
                    evaluate(
                        "document.querySelector('.st-key-atlas_nav_exploration button').click()"
                    )
                    wait_for(
                        "document.querySelector('.atlas-page-title')?.textContent==='Exploración EDA'"
                    )
                    wait_for(
                        "[...document.querySelectorAll('[role=tab]')].some(tab=>tab.innerText==='Distribuciones y relaciones')"
                    )
                    evaluate(
                        "[...document.querySelectorAll('[role=tab]')].find(tab=>tab.innerText==='Distribuciones y relaciones').click()"
                    )
                    time.sleep(0.4)
                    evaluate(
                        "[...document.querySelectorAll('[data-testid=stPlotlyChart]')].find(chart=>chart.offsetParent!==null)?.scrollIntoView({block:'center'})"
                    )
                    screenshot("chart-exploration")
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
                if args.agents and args.all_pages:
                    evaluate("document.querySelector('.st-key-open_EDAAgent button').click()")
                    wait_for("Boolean(document.querySelector('.st-key-panel_EDAAgent'))")
                    screenshot("mobile-assistant")
                    dialog_measurements = evaluate(
                        "(()=>{let element=document.querySelector('[role=dialog]')||document.querySelector('[data-testid=stDialog]');let rect=element.getBoundingClientRect();return {viewport:innerWidth,width:rect.width,x:rect.x,documentWidth:document.documentElement.scrollWidth};})()"
                    )
                    assert dialog_measurements["width"] <= dialog_measurements["viewport"] + 1
                    assert dialog_measurements["documentWidth"] == dialog_measurements["viewport"]
                    (output / "dialog-mobile.json").write_text(
                        json.dumps(dialog_measurements), encoding="utf-8"
                    )
                    evaluate("document.querySelector('.st-key-close_EDAAgent button').click()")
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
