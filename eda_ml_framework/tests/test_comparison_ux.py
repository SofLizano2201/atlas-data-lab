"""Compare real saved evaluations through the Spanish guided interface."""

from test_integration import active_app, navigate


def test_single_result_explains_next_step_and_history_link():
    app = active_app()
    navigate(app, "🎯 Clasificación")
    next(b for b in app.button if b.label == "Crear modelo y revisar resultados").click().run(
        timeout=60
    )
    navigate(app, "⚖️ Experimentos")
    assert not app.exception
    assert any("al menos dos" in item.value for item in app.info)
    next(b for b in app.button if b.label == "Ir a mis experimentos").click().run()
    assert app.session_state["navigation"] == "experiment_history"


def test_saved_results_are_selected_and_changing_criterion_does_not_train():
    app = active_app()
    navigate(app, "🎯 Clasificación")
    for _ in range(2):
        next(b for b in app.button if b.label == "Crear modelo y revisar resultados").click().run(
            timeout=60
        )
    navigate(app, "experiment_history")
    app.button(key="history_compare").click().run()
    assert app.session_state["navigation"] == "⚖️ Experimentos"
    assert not app.exception
    models = next(box for box in app.multiselect if box.label == "Modelos que se mostrarán")
    assert len(models.value) == 2
    cohort = app.selectbox(key="comparison_cohort")
    assert "conjunto 1" in cohort.options[0]
    assert "clasificacion" not in cohort.options[0]
    assert app.session_state["comparison_selection"]["metric"] == "F1_macro"
    count = len(app.session_state["experiments"])
    next(box for box in app.selectbox if box.label == "¿Qué quieres comparar?").select(
        "Accuracy"
    ).run()
    assert not app.exception
    assert len(app.session_state["experiments"]) == count
    app.radio(key="comparison_export_format").set_value("csv").run()
    assert not app.exception
    assert app.session_state["comparison_export_format"] == "csv"
    assert len(app.session_state["experiments"]) == count
    assert any("Informe de la comparación" in heading.value for heading in app.markdown)
    models = next(box for box in app.multiselect if box.label == "Modelos que se mostrarán")
    models.set_value(models.value[:1]).run()
    assert not app.exception
    assert any("al menos dos" in item.value for item in app.info)

