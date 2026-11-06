"""Automatic analyses use real model outputs, cache by version and expose coverage."""

import pandas as pd
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    make_blobs,
    make_moons,
)

from atlas.controllers.automatic_eda_controller import AutomaticEDAController, EDALimits
from atlas.core.events import EventType
from atlas.models import ExploratoryDataAnalysis
from atlas.services.dataset_service import DatasetService


def test_confirmation_runs_eda_and_records_actual_missing_values():
    frame = pd.DataFrame({"price": [10.0, None, 30.0, 30.0], "region": ["a", "b", "a", "a"]})
    state = {"eda_limits": EDALimits(max_charts=0)}
    DatasetService.activate(state, frame, "mixed.csv")
    execution = state["automatic_eda"][AutomaticEDAController.key(state["eda"])]
    assert execution.profile["missing_cells"] == 1
    assert execution.profile["additional_duplicates"] == 1
    assert execution.operations["descriptive_statistics"]["value"].loc["price", "mediana"] == 30
    assert execution.operations["missing_values"]["value"].loc["price", "cantidad"] == 1
    assert execution.operations["hypothesis_tests"]["status"] == "skipped"
    assert execution.report and "Execution coverage" in execution.report
    assert EventType.DATASET_CONFIRMED in [
        event.type for event in state["workflow_events"].values()
    ]
    pd.testing.assert_frame_equal(state["eda"].df, frame)
    assert not state["experiments"]


def test_automatic_charts_exclude_ids_and_constants():
    frame = pd.DataFrame(
        {"id": range(30), "price": range(100, 130), "size": range(50, 80), "constant": [1] * 30}
    )
    state = {"eda_limits": EDALimits(max_charts=6)}
    DatasetService.activate(state, frame, "example.csv")
    result = state["automatic_eda"][AutomaticEDAController.key(state["eda"])]
    assert "chart:id" not in result.operations
    assert "chart:constant" not in result.operations
    assert result.operations["chart:price"]["value"].data[0].marker.color == "#694D75"
    correlation = result.operations["chart:correlation"]["value"]
    assert "id" not in correlation.data[0].x and "constant" not in correlation.data[0].x
    assert "id" in result.operations["schema"]["value"].index


def test_version_cache_and_invalidation(monkeypatch):
    model = ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, 2.0, 4.0]}))
    calls = []
    original = model.statistics

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "statistics", counted)
    state = {"eda": model}
    first = AutomaticEDAController.run(state, EDALimits(max_charts=0))
    assert AutomaticEDAController.run(state) is first and len(calls) == 1
    model.apply(model.df.iloc[:2], "subset", confirm=True)
    assert AutomaticEDAController.run(state, EDALimits(max_charts=0)).key != first.key
    assert len(calls) == 2


def test_resource_limits_are_explicit_and_charts_label_samples():
    state = {"eda": ExploratoryDataAnalysis(pd.DataFrame({"x": range(1100), "y": range(1100)}))}
    result = AutomaticEDAController.run(
        state, EDALimits(max_chart_points=100, max_charts=1, max_kendall_rows=10)
    )
    assert result.operations["correlation_kendall"]["status"] == "skipped"
    assert result.operations["chart:x"]["sampled"]
    assert result.profile["rows"] == 1100
    assert result.operations["descriptive_statistics"]["value"].loc["x", "n"] == 1100
    assert result.status == "partial"


def test_failure_is_local_and_no_numeric_analysis_is_invented(monkeypatch):
    model = ExploratoryDataAnalysis(pd.DataFrame({"kind": ["a", "b", "a"]}))

    def failure(*args, **kwargs):
        raise ValueError("deliberate test failure")

    monkeypatch.setattr(model, "rare_categories", failure)
    result = AutomaticEDAController.run({"eda": model}, EDALimits(max_charts=0))
    assert result.operations["numeric_analysis"]["status"] == "skipped"
    assert result.operations["rare_categories:kind"]["status"] == "failed"
    assert result.operations["frequencies:kind"]["value"].loc["a", "absoluta"] == 2


@pytest.mark.parametrize("loader", [load_iris, load_wine, load_breast_cancer, load_diabetes])
def test_representative_datasets(loader):
    frame = loader(as_frame=True).frame
    result = AutomaticEDAController.run(
        {"eda": ExploratoryDataAnalysis(frame)}, EDALimits(max_charts=0, max_numeric=5)
    )
    assert result.profile["rows"] == len(frame)
    assert result.operations["descriptive_statistics"]["value"].loc[
        frame.columns[0], "media"
    ] == pytest.approx(frame.iloc[:, 0].mean())


@pytest.mark.parametrize("generator", [make_blobs, make_moons])
def test_synthetic_patterns(generator):
    values, _ = generator(n_samples=80, random_state=42)
    result = AutomaticEDAController.run(
        {
            "eda": ExploratoryDataAnalysis(
                pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])
            )
        },
        EDALimits(max_charts=0),
    )
    assert result.operations["outliers_iqr"]["status"] == "completed"
