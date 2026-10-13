"""Integración real con Streamlit AppTest y estado entre interacciones."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from frontend.components.sidebar import NAV_ITEMS

APP = Path(__file__).resolve().parents[1] / "app.py"


def navigate(app, route):
    item = next(item for item in NAV_ITEMS if item.route == route)
    app.sidebar.button(key=f"atlas_page_{item.slug}").click().run()
    return app


def active_app():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    navigate(app, "📂 Datos")
    next(b for b in app.button if b.label == "Activar ejemplo").click().run()
    assert not app.exception
    return app


def test_home_navigation_and_state():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    next(b for b in app.button if b.label == "📂 Cargar dataset").click().run()
    assert app.session_state["navigation"] == "📂 Datos"
    app = active_app()
    for page in [
        "🏠 Inicio",
        "🔎 Exploración",
        "🧹 Preparación",
        "🧩 Clustering",
        "🎯 Clasificación",
        "📈 Regresión",
        "⚖️ Experimentos",
        "📤 Reportes",
    ]:
        navigate(app, page)
        assert not app.exception, page
        assert len(app.session_state["eda"].df) == 150


def test_train_from_frontend():
    app = active_app()
    navigate(app, "🎯 Clasificación")
    next(b for b in app.button if b.label == "Entrenar y evaluar").click().run(timeout=60)
    assert not app.exception
    assert len(app.session_state["experiments"]) == 1
    assert app.session_state["experiments"][0].metrics["Accuracy"] > 0.8
    navigate(app, "⚖️ Experimentos")
    assert len(app.session_state["experiments"]) == 1


def test_clean_confirmation():
    app = active_app()
    navigate(app, "🧹 Preparación")
    next(b for b in app.button if b.label == "Previsualizar transformación").click().run()
    assert app.session_state["eda"].versiones == 1
    next(b for b in app.button if b.label == "Confirmar estos cambios").click().run()
    assert not app.exception
    assert app.session_state["eda"].versiones == 2


def test_cluster_from_frontend():
    app = active_app()
    navigate(app, "🧩 Clustering")
    next(b for b in app.button if b.label == "Ejecutar agrupamiento").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["experiments"][0].problem == "clustering"
    next(b for b in app.button if b.label == "Visualizar grupos con PCA").click().run()
    assert not app.exception


def test_regression_from_frontend():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    navigate(app, "📂 Datos")
    next(s for s in app.selectbox if s.label == "Ejemplo").set_value("Diabetes").run()
    next(b for b in app.button if b.label == "Activar ejemplo").click().run()
    navigate(app, "📈 Regresión")
    next(b for b in app.button if b.label == "Entrenar y evaluar").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["experiments"][0].problem == "regresion"
    assert app.session_state["experiments"][0].metrics["RMSE"] > 0


def test_sidebar_compact_navigation_and_dataset():
    app = active_app()
    original = app.session_state["eda"].df.copy()
    app.session_state["dataset_name"] = "archivo_largo_<script>alert(1)</script>.csv"
    app.sidebar.button(key="atlas_toggle").click().run()
    assert not app.exception
    assert app.session_state["sidebar_compact"] is True
    assert len(app.sidebar.radio) == 0
    for item in NAV_ITEMS:
        assert app.sidebar.button(key=f"atlas_page_{item.slug}").label == item.label
    navigate(app, "🔎 Exploración")
    assert app.session_state["navigation"] == "🔎 Exploración"
    assert app.session_state["eda"].df.equals(original)
    html = "\n".join(m.value for m in app.sidebar.markdown)
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert "DATASET ACTIVO" in html
    app.sidebar.button(key="atlas_toggle").click().run()
    assert app.session_state["sidebar_compact"] is False
    assert app.session_state["navigation"] == "🔎 Exploración"
    assert not app.exception
