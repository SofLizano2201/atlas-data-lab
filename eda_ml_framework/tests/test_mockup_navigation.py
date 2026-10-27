"""Verificar interacciones nuevas del diseño, conservando datos y experimentos."""

from streamlit.testing.v1 import AppTest
from test_integration import APP, active_app, navigate

from frontend.components.sidebar import NAV_ITEMS


def test_header_search_changes_route_without_losing_dataset():
    app = active_app()
    navigate(app, "🏠 Inicio")
    original_hash = app.session_state["eda"].dataset_version.content_hash
    app.selectbox(key="atlas_search_home").set_value(NAV_ITEMS[4]).run()
    assert not app.exception
    assert app.session_state["navigation"] == NAV_ITEMS[4].route
    assert app.session_state["eda"].dataset_version.content_hash == original_hash


def test_model_card_updates_real_algorithm_selector():
    app = active_app()
    navigate(app, "🎯 Clasificación")
    app.button(key="choose_atlas_algorithms_clasificacion_knn").click().run()
    assert not app.exception
    assert app.multiselect(key="atlas_algorithms_clasificacion").value == ["knn"]
    assert len(app.session_state["experiments"]) == 0
    next(b for b in app.button if b.label == "Entrenar y evaluar").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["experiments"][0].name == "knn"


def test_cluster_card_updates_configuration_without_training():
    app = active_app()
    navigate(app, "🧩 Clustering")
    app.button(key="choose_atlas_cluster_algorithm_dbscan").click().run()
    assert not app.exception
    assert app.selectbox(key="atlas_cluster_algorithm").value == "dbscan"
    assert len(app.session_state["experiments"]) == 0


def test_welcome_search_works_without_dataset():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.selectbox(key="atlas_search_home").set_value(NAV_ITEMS[1]).run()
    assert not app.exception
    assert app.session_state["navigation"] == NAV_ITEMS[1].route


def test_home_agent_invoker_uses_active_dataset():
    app = active_app()
    navigate(app, "🏠 Inicio")
    app.button(key="open_EDAAgent").click().run()
    assert not app.exception
    assert app.session_state["agent_open_EDAAgent"]
    assert app.session_state["agent_context"].dataset.rows == 150


def test_report_card_applies_real_section_preset():
    app = active_app()
    navigate(app, "📤 Reportes")
    app.button(key="choose_atlas_report_type_cleaning").click().run()
    assert not app.exception
    selected = next(widget for widget in app.multiselect if widget.label == "Secciones del informe")
    assert selected.value == ["overview", "quality", "cleaning"]
    next(button for button in app.button if button.label == "Generar reporte").click().run()
    assert not app.exception
    assert app.session_state["reports"][0].sections == selected.value
