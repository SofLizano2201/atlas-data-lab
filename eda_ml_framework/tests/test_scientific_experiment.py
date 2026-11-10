"""Real controlled runs, paired uncertainty and reproducible portable manifests."""

import json
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine

from atlas.agents.factory import AgentFactory
from atlas.controllers.experiment_lab_controller import ExperimentLabController as Lab
from atlas.core.exceptions import FrameworkError
from atlas.core.session import AgentContext
from atlas.models import Clustering, ExploratoryDataAnalysis
from atlas.models.scientific_experiment import (
    ExperimentalAlternative as Alternative,
)
from atlas.models.scientific_experiment import (
    ScientificExperiment as Science,
)
from atlas.models.scientific_experiment import (
    ScientificExperimentDefinition as Definition,
)
from atlas.services.experiment_service import ExperimentService
from atlas.services.report_service import ReportService


def state_for(frame=None):
    return {
        "eda": ExploratoryDataAnalysis(
            frame if frame is not None else load_wine(as_frame=True).data
        ),
        "experiments": [],
        "models": {},
        "metrics": {},
    }


def design(state, **changes):
    defaults = {
        "name": "Efecto de la escala",
        "question": "¿La escala mejora la silueta?",
        "hypothesis": "Estandarizar mejora la silueta bajo estas condiciones.",
        "template": "preprocessing",
        "task": "clustering",
        "dataset": state["eda"].dataset_version,
        "predictors": list(state["eda"].df.columns[:4]),
        "alternatives": [
            Alternative(
                name="A",
                algorithm="kmeans",
                scaler="none",
                parameters={"n_clusters": 3, "n_init": 10},
            ),
            Alternative(
                name="B",
                algorithm="kmeans",
                scaler="standard",
                parameters={"n_clusters": 3, "n_init": 10},
            ),
        ],
        "factor": "scaler",
        "metric": "Silhouette",
        "direction": "max",
        "seeds": [42, 43],
        "validation": "same_rows_initializations",
    }
    return Definition(**(defaults | changes))


def test_hypothesis_and_controlled_design():
    state = state_for()
    definition = design(state)
    assert definition.question and definition.hypothesis and definition.null_hypothesis
    assert not any(f.blocking for f in Science.validate(definition, state["eda"].df))
    bad = definition.model_copy(
        update={
            "alternatives": [definition.alternatives[0], Alternative(name="B", algorithm="birch")]
        }
    )
    assert any(
        f.finding_id == "confounding" and f.blocking for f in Science.validate(bad, state["eda"].df)
    )


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"metric": "Accuracy"}, "metric"),
        ({"direction": "min"}, "metric"),
        ({"validation": "time"}, "protocol"),
        ({"metadata": {"group_column": "entity"}}, "structured_data"),
        ({"paired_bootstrap": True}, "bootstrap_design"),
        (
            {
                "seeds": [1, 2, 3, 4, 5],
                "alternatives": [
                    Alternative(
                        name=str(i),
                        algorithm="kmeans",
                        scaler="none" if i == 0 else "standard",
                        parameters={"n_clusters": i + 2},
                    )
                    for i in range(3)
                ],
            },
            "budget",
        ),
    ],
)
def test_invalid_designs(changes, code):
    state = state_for()
    assert any(
        f.finding_id == code and f.blocking
        for f in Science.validate(design(state, **changes), state["eda"].df)
    )


def test_dataset_reference_and_immutable_source():
    state = state_for()
    definition = design(state)
    frame = state["eda"].df.iloc[:100]
    with pytest.raises(FrameworkError):
        Science.create(definition, frame)


def test_confirmation_execution_cancellation_and_resume(monkeypatch):
    state = state_for()
    before = state["eda"].df
    calls = []
    original = Clustering.train

    def train(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Clustering, "train", train)
    run = Lab.prepare(state, design(state))
    with pytest.raises(ValueError):
        Lab.execute(state, run["id"])
    with pytest.raises(ValueError):
        Lab.step(state, run["id"])
    assert not calls
    Lab.authorize(state, run["id"], confirm=True)
    Lab.step(state, run["id"])
    Lab.cancel(state, run["id"])
    assert run["status"] == "cancelled"
    Lab.execute(state, run["id"], confirm=True)
    assert run["status"] == "completed" and len(calls) == 4
    assert len(state["experiments"]) == 4
    assert len(Science.stability(run)) == 2
    assert len(Science.paired_differences(run)) == 2
    assert Science.summary(run)["count"].tolist() == [2, 2]
    assert "intervalo de confianza" in Science.interpretation(run)
    pd.testing.assert_frame_equal(before, state["eda"].df)
    with pytest.raises(ValueError):
        Lab.execute(state, run["id"], confirm=True)


def test_reproduction_export_import_and_versions():
    state = state_for()
    run = Lab.prepare(state, design(state, seeds=[42]))
    Lab.execute(state, run["id"], confirm=True)
    manifest = Science.manifest(run)
    assert manifest["environment"]["libraries"]["scikit-learn"]
    assert manifest["started_at"] and manifest["completed_at"]
    json.dumps(manifest, allow_nan=False)
    bundle = ExperimentService.export_investigation(run)
    with ZipFile(BytesIO(bundle)) as archive:
        assert {"manifest.json", "dataset.parquet", "report.html", "results.csv"} <= set(
            archive.namelist()
        )
    imported = ExperimentService.import_investigation(bundle)
    assert imported["parent_id"] == run["id"] and imported["status"] == "ready"
    assert not imported["evaluations"]
    pd.testing.assert_frame_equal(imported["dataset"], run["dataset"])
    reproduced = Lab.reproduce(state, run["id"])
    Lab.execute(state, reproduced["id"], confirm=True)
    np.testing.assert_allclose(Science.summary(run)["mean"], Science.summary(reproduced)["mean"])
    state["eda"].apply(state["eda"].df.iloc[:120], "subset", confirm=True)
    original = Lab.reproduce(state, run["id"])
    current = Lab.reproduce(state, run["id"], current_dataset=True)
    assert len(original["dataset"]) == 178 and len(current["dataset"]) == 120
    assert current["definition"].dataset.content_hash != run["definition"].dataset.content_hash


def test_real_supervised_training_matches_holdouts_and_cv():
    state = state_for(load_iris(as_frame=True).frame)
    definition = design(
        state,
        task="classification",
        target="target",
        predictors=list(state["eda"].df.columns[:4]),
        template="algorithms",
        factor="algorithm",
        metric="Accuracy",
        validation="random_holdout_cv",
        alternatives=[
            Alternative(name="A", algorithm="logistic"),
            Alternative(name="B", algorithm="tree", parameters={"max_depth": 3}),
        ],
    )
    run = Lab.prepare(state, definition)
    Lab.execute(state, run["id"], confirm=True)
    assert run["status"] == "completed"
    for seed in definition.seeds:
        a, b = (run["artifacts"][f"{i}:{seed}"] for i in range(2))
        assert a.details["y_test"].equals(b.details["y_test"])
        assert len(a.details["cv_scores"]) == definition.cv
        assert a.model.named_steps["preprocess"]
    assert "interval" not in Science.charts(run)


def test_leakage_invalid_cv_and_bootstrap_design():
    state = state_for(load_iris(as_frame=True).frame)
    definition = design(
        state,
        task="classification",
        target="target",
        predictors=["target"],
        template="algorithms",
        factor="algorithm",
        metric="Accuracy",
        validation="random_holdout_cv",
        metadata={"resampling_fit_scope": "full_dataset"},
        alternatives=[
            Alternative(name="A", algorithm="logistic"),
            Alternative(name="B", algorithm="tree"),
        ],
    )
    codes = {f.finding_id for f in Science.validate(definition, state["eda"].df) if f.blocking}
    assert {"target_leakage", "resampling_leakage"} <= codes
    tiny = state_for(pd.DataFrame({"x": range(12), "target": [0] * 10 + [1] * 2}))
    bad = definition.model_copy(
        update={"dataset": tiny["eda"].dataset_version, "predictors": ["x"], "metadata": {}}
    )
    assert any(f.finding_id == "invalid_validation" for f in Science.validate(bad, tiny["eda"].df))


def test_paired_bootstrap_known_matched_errors():
    state = state_for(load_iris(as_frame=True).frame)
    definition = design(
        state,
        task="classification",
        target="target",
        predictors=list(state["eda"].df.columns[:4]),
        template="algorithms",
        factor="algorithm",
        metric="Accuracy",
        validation="random_holdout_cv",
        seeds=[42],
        paired_bootstrap=True,
        independent_observations_confirmed=True,
        alternatives=[
            Alternative(name="A", algorithm="logistic"),
            Alternative(name="B", algorithm="tree"),
        ],
    )
    run = Science.create(definition, state["eda"].df)
    y = pd.Series([0] * 20)
    a = SimpleNamespace(predictions=pd.DataFrame({"prediccion": [1] * 20}), details={"y_test": y})
    b = SimpleNamespace(predictions=pd.DataFrame({"prediccion": [0] * 20}), details={"y_test": y})
    run.update(status="completed", artifacts={"0:42": a, "1:42": b})
    interval = Science.bootstrap(run)
    assert interval["low"] == interval["estimate"] == interval["high"] == 1
    assert interval["samples"] == 1000
    assert "no demuestra certeza" in interval["scope"]
    b.predictions.index = range(1, 21)
    with pytest.raises(FrameworkError):
        Science.bootstrap(run)


def test_failure_recovery_and_budget(monkeypatch):
    state = state_for()
    run = Lab.prepare(state, design(state, seeds=[42]))
    original = Science.evaluate
    count = [0]

    def fail_once(*args):
        count[0] += 1
        if count[0] == 1:
            raise ValueError("Fallo recuperable")
        return original(*args)

    monkeypatch.setattr(Science, "evaluate", fail_once)
    Lab.execute(state, run["id"], confirm=True)
    assert run["status"] == "partial"
    Lab.execute(state, run["id"], confirm=True)
    assert run["status"] == "completed" and count[0] == 3
    assert len(run["attempts"]) == 1
    state["experiments"] = [None] * 30
    rerun = Lab.reproduce(state, run["id"])
    with pytest.raises(ValueError):
        Lab.execute(state, rerun["id"], confirm=True)


@pytest.mark.parametrize(
    "name", ["eda", "clustering", "classification", "regression", "comparison"]
)
def test_five_agents_recommend_and_interpret_without_training(name):
    state = state_for()
    context = AgentContext(state["eda"])
    context.application_state = state | {"navigation": "scientific_lab"}
    agent = AgentFactory.get_agent(name, context)
    assert agent.recommend_experiment()["requires_confirmation"]
    assert agent.on_event(SimpleNamespace(id="lab", type="LAB_DESIGN_CHANGED")).message
    assert not context.experiments


def test_reports_and_interactive_charts():
    state = state_for()
    run = Lab.prepare(state, design(state, seeds=[42]))
    Lab.execute(state, run["id"], confirm=True)
    html = ReportService.lab_report(run, "html")
    assert b"Hip" in html and run["id"].encode() in html
    pdf = ReportService.lab_report(run, "pdf")
    assert pdf.startswith(b"%PDF")
    from pypdf import PdfReader

    assert len(PdfReader(BytesIO(pdf)).pages) > 0
    charts = Science.charts(run)
    assert {"performance", "cost", "differences"} <= set(charts)
    assert "interval" not in charts


def test_real_regression_bootstrap_and_sensitivity():
    from sklearn.datasets import load_diabetes

    state = state_for(load_diabetes(as_frame=True).frame.iloc[:100])
    definition = design(
        state,
        task="regression",
        target="target",
        template="sensitivity",
        factor="parameters",
        metric="MAE",
        direction="min",
        seeds=[42],
        validation="random_holdout_cv",
        paired_bootstrap=True,
        independent_observations_confirmed=True,
        alternatives=[
            Alternative(name="A", algorithm="ridge", parameters={"alpha": 0.1}),
            Alternative(name="B", algorithm="ridge", parameters={"alpha": 10.0}),
        ],
    )
    run = Lab.prepare(state, definition)
    Lab.execute(state, run["id"], confirm=True)
    assert run["status"] == "completed" and run["uncertainty"] is not None
    interval = run["uncertainty"]
    assert interval["low"] <= interval["high"]
    assert interval["estimate"] == pytest.approx(
        Science.paired_differences(run).improvement.iloc[0]
    )
    assert "interval" in Science.charts(run)
    assert "sensitivity" in Science.charts(run)
    json.dumps(Science.manifest(run), allow_nan=False)


def test_previous_investigations_and_tampered_consent():
    state = state_for()
    original = Lab.prepare(state, design(state, seeds=[42]))
    Lab.execute(state, original["id"], confirm=True)
    reproduction = Lab.reproduce(state, original["id"])
    Lab.execute(state, reproduction["id"], confirm=True)
    assert Science.compare_investigations([original, reproduction], rank=True).attrs["comparable"]
    reproduction["definition"] = reproduction["definition"].model_copy(update={"seeds": [99]})
    assert not Science.compare_investigations([original, reproduction]).attrs["comparable"]
    with pytest.raises(FrameworkError):
        Science.compare_investigations([original, reproduction], rank=True)
    proposal = Lab.reproduce(state, original["id"])
    Lab.authorize(state, proposal["id"], confirm=True)
    proposal["definition"].alternatives[0].parameters["n_clusters"] = 4
    with pytest.raises(ValueError):
        Lab.step(state, proposal["id"])
    assert original["definition"].alternatives[0].parameters["n_clusters"] == 3


def test_numerical_warnings_are_recorded_without_duplicate_training(monkeypatch):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    state = state_for()
    definition = design(state, seeds=[42])
    calls = []

    def delegate(*args):
        calls.append(args)
        warnings.warn("Convergencia pendiente", ConvergenceWarning, stacklevel=2)
        return SimpleNamespace(details={}), object()

    monkeypatch.setattr(Science, "_evaluate", delegate)
    result, _ = Science.evaluate(definition, state["eda"].df, definition.alternatives[0], 42)
    assert len(calls) == 1
    assert result.details["lab_warnings"] == ["Convergencia pendiente"]
