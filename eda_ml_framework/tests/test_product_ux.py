"""User-oriented preparation, recoverable errors and report navigation."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from frontend.components.sidebar import NAV_ITEMS

APP = Path(__file__).parents[1] / "app.py"


def navigate(app, slug):
    item = next(item for item in NAV_ITEMS if item.slug == slug)
    app.sidebar.button(key=f"atlas_page_{item.slug}").click().run()


def test_preparation_error_is_recoverable_and_preserves_data():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    navigate(app, "data")
    next(box for box in app.selectbox if box.label == "Ejemplo").set_value("Diabetes").run()
    next(button for button in app.button if button.label == "Activar ejemplo").click().run()
    original = app.session_state["eda"].df
    navigate(app, "preparation")
    next(box for box in app.selectbox if box.label == "¿Qué necesitas preparar?").set_value(
        "Escalas y distribuciones"
    ).run()
    next(box for box in app.selectbox if box.label == "Operación").set_value("Logaritmo").run()
    next(
        button for button in app.button if button.label == "Previsualizar transformación"
    ).click().run()
    assert not app.exception
    assert any("Esta acción necesita una revisión" in item.value for item in app.markdown)
    assert app.session_state["eda"].df.equals(original)
    assert app.session_state["eda"].version_count == 1
    next(box for box in app.selectbox if box.label == "Operación").set_value("StandardScaler").run()
    next(
        button for button in app.button if button.label == "Previsualizar transformación"
    ).click().run()
    assert not app.exception
    assert app.session_state["pending"] is not None
    assert app.session_state["eda"].version_count == 1


def test_report_workflow_tabs_keep_generation_functional():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    navigate(app, "data")
    next(button for button in app.button if button.label == "Activar ejemplo").click().run()
    navigate(app, "reports")
    titles = [tab.label for tab in app.tabs]
    assert "1. Configurar el informe" in titles
    assert "2. Historial y descargas" in titles
    next(button for button in app.button if button.label == "Generar reporte").click().run()
    assert not app.exception
    assert len(app.session_state["reports"]) == 1
    assert any("Historial y descargas" in message.value for message in app.success)


def test_view_preferences_and_workflow_navigation_preserve_data():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    navigate(app, "data")
    next(button for button in app.button if button.label == "Activar ejemplo").click().run()
    original = app.session_state["eda"].df
    app.toggle(key="view_show_context").set_value(True).run()
    app.toggle(key="view_reduce_motion").set_value(True).run()
    app.button(key="flow_data_exploration").click().run()
    assert not app.exception
    assert app.session_state["navigation"] == "🔎 Exploración"
    assert app.session_state["view_show_context"]
    assert app.session_state["view_reduce_motion"]
    assert app.session_state["eda"].df.equals(original)
    assert not app.session_state["experiments"]
