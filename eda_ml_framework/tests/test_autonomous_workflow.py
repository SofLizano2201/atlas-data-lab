"""Real local autonomous executions, isolation, reproducibility and UI integration."""

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris

from atlas.controllers.automatic_eda_controller import EDALimits
from atlas.controllers.autonomous_controller import AutonomousController
from atlas.services.dataset_service import DatasetService


def state_for(frame):
    state = {"eda_limits": EDALimits(max_seconds=5, max_charts=2)}
    DatasetService.activate(state, frame, "example.csv")
    return state


@pytest.mark.parametrize(
    "loader,problem", [(load_iris, "clasificacion"), (load_diabetes, "regresion")]
)
def test_autonomous_trains_compares_and_reports_without_credentials(loader, problem):
    frame = loader(as_frame=True).frame
    state = state_for(frame)
    notices = []
    run = AutonomousController.run(state, progress=notices.append)
    assert run.status == "completed", run.errors
    assert len(run.experiments) == 7
    assert {item["problem"] for item in run.comparisons} == {problem, "clustering"}
    assert len({item["agent"] for item in run.decisions}) == 5
    assert "plotly" in run.report.lower() and "CV_media" in run.report
    assert notices and run.figures
    pd.testing.assert_frame_equal(state["eda"].df, frame)
    pd.testing.assert_frame_equal(state["original"], frame)
    selection = next(item for item in run.comparisons if item["problem"] == problem)
    candidates = [item for item in state["experiments"] if item.problem == problem]
    assert selection["best"] == max(candidates, key=lambda item: item.metrics["CV_media"]).id
    for item in candidates:
        assert item.config["target"] not in item.config["predictors"]
        assert set(item.details["x_train"].index).isdisjoint(item.details["x_test"].index)
        assert item.config["autonomous_run"] == run.key
    assert AutonomousController.run(state) is run
    assert len(state["experiments"]) == 7


def test_target_missing_not_imputed_and_identifiers_not_selected():
    frame = load_iris(as_frame=True).frame
    frame["id"] = range(len(frame))
    frame.loc[0, "target"] = float("nan")
    state = state_for(frame)
    run = AutonomousController.run(state, target="target")
    assert all(item.problem == "clustering" for item in state["experiments"])
    assert any("objetivo" in warning and "faltante" in warning for warning in run.warnings)
    assert all("id" not in decision.get("features", []) for decision in run.decisions)
    assert pd.isna(state["eda"].df.loc[0, "target"])


def test_budgets_invalid_target_and_new_dataset_reset():
    frame = load_iris(as_frame=True).frame
    state = state_for(frame)
    run = AutonomousController.run(state, max_seconds=0)
    assert not run.experiments and run.warnings
    invalid = AutonomousController.run(state, target="missing")
    assert invalid.status == "failed" and invalid.report
    DatasetService.activate(state, frame.iloc[:40], "other.csv")
    assert not state["autonomous_runs"] and state["autonomous_active"] is None


def test_autonomous_panel_executes_and_keeps_results_on_rerun():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from frontend.components.sidebar import NAV_ITEMS

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60).run()
    data = next(item for item in NAV_ITEMS if item.slug == "data")
    app.sidebar.button(key=f"atlas_page_{data.slug}").click().run()
    next(button for button in app.button if button.label == "Activar ejemplo").click().run()
    assert not app.exception
    next(
        button for button in app.button if button.label == "Analizar todo automáticamente"
    ).click().run(timeout=60)
    assert not app.exception
    assert len(app.session_state["experiments"]) == 7
    run = app.session_state["autonomous_runs"][app.session_state["autonomous_active"]]
    assert run.report and run.comparisons
    best = next(item for item in run.comparisons if item["problem"] == "clasificacion")["best"]
    app.button(key=f"auto_chart_button_{best}").click().run()
    assert not app.exception
    assert app.session_state[f"auto_chart_open_{best}"]
    assert len(app.session_state["experiments"]) == 7
    app.run()
    assert not app.exception and len(app.session_state["experiments"]) == 7
    next(box for box in app.selectbox if box.label == "¿Qué resultado quieres explorar?").set_value(
        "confusion"
    ).run()
    assert not app.exception and len(app.session_state["experiments"]) == 7
