"""QA local con Edge headless y CDP; no instala dependencias de navegador."""

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
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "sidebar"
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
                "--no-first-run",
                "--no-default-browser-check",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                "http://127.0.0.1:8501",
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
                        result = json.loads(ws.recv(timeout=20))
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
                    image = command("Page.captureScreenshot", {"format": "png"})
                    (output / f"{name}.png").write_bytes(base64.b64decode(image["data"]))

                command(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False},
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
                wait_for("document.body.innerText.includes('Datos / Punto de partida')")
                evaluate(
                    "[...document.querySelectorAll('button')].find(b=>b.innerText==='Activar ejemplo').click()"
                )
                wait_for("document.querySelector('.atlas-dataset-name')?.innerText==='Iris'")
                time.sleep(0.5)
                screenshot("dataset")
                evaluate("document.querySelector('.st-key-atlas_sidebar_toggle button').click()")
                wait_for(
                    "document.querySelector('[data-testid=stSidebar]').getBoundingClientRect().width===72"
                )
                time.sleep(0.3)
                screenshot("compact")
                evaluate("document.querySelector('.st-key-atlas_nav_exploration button').click()")
                wait_for("document.body.innerText.includes('Exploración / Entender primero')")
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
