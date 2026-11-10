"""Exercise the live research flow, including its explicit training consent."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_lab_end_to_end_requires_confirmation_and_exports():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.button(key="atlas_page_data").click().run()
    next(b for b in app.button if b.label == "Activar ejemplo").click().run()
    app.sidebar.button(key="atlas_page_lab").click().run()
    assert not app.exception
    assert not app.session_state["experiments"]
    app.selectbox(key="lab_template").set_value("Evaluar una transformación").run()
    next(n for n in app.number_input if n.label == "Repeticiones coordinadas").set_value(1).run()
    app.button(key="lab_prepare").click().run()
    assert not app.exception
    identifier = app.session_state["active_investigation"]
    assert app.session_state["scientific_investigations"][identifier]["status"] == "ready"
    assert app.button(key=f"lab_start_{identifier}").disabled
    assert not app.session_state["experiments"]
    app.checkbox(key=f"lab_consent_{identifier}").check().run()
    app.button(key=f"lab_start_{identifier}").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["scientific_investigations"][identifier]["status"] == "completed"
    assert len(app.session_state["experiments"]) == 2
    next(b for b in app.button if b.label == "Preparar archivos de esta investigación").click().run(
        timeout=60
    )
    assert not app.exception
    assert app.session_state["lab_exports"]["zip"].startswith(b"PK")
    next(
        b for b in app.button if b.label == "Preparar reproducción con un identificador nuevo"
    ).click().run()
    assert not app.exception
    new_id = app.session_state["active_investigation"]
    assert new_id != identifier
    assert app.session_state["scientific_investigations"][new_id]["status"] == "ready"
    assert len(app.session_state["experiments"]) == 2
