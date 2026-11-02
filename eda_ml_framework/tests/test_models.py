"""English scientific API and academic compatibility execute the same implementations."""

import pandas as pd
from sklearn.datasets import load_diabetes

from atlas.models import (
    Classification,
    Clustering,
    ExploratoryDataAnalysis,
    Regression,
    SupervisedLearning,
    UnsupervisedLearning,
)
from framework import AnalisisDatosExploratorio, Clasificacion, Regresion


def test_english_models_and_academic_aliases_share_real_inheritance():
    assert AnalisisDatosExploratorio is ExploratoryDataAnalysis
    assert Clasificacion is Classification and Regresion is Regression
    assert issubclass(Classification, SupervisedLearning)
    assert issubclass(Regression, SupervisedLearning)
    assert issubclass(Clustering, UnsupervisedLearning)
    assert issubclass(SupervisedLearning, ExploratoryDataAnalysis)
    model = ExploratoryDataAnalysis(pd.DataFrame({"x": [1.0, 2.0, None, 4.0]}))
    assert model.profile()["missing_cells"] == 1
    assert model.mean("x") == model.media("x")
    assert model.source_columns == ["x"]


def test_configuration_is_english_but_supports_historical_lookups():
    frame = load_diabetes(as_frame=True).frame
    result = Regression(frame).train("ridge", "target", list(frame.columns[:-1]), cv=2)
    assert "target" in dict(result.config) and "objetivo" not in dict(result.config)
    assert result.config["objetivo"] == result.config["target"]
    assert result.config["predictores"] == result.config["predictors"]


def test_live_legacy_session_upgrade_preserves_data_and_invalidates_tickets():
    from atlas.core.compatibility import upgrade_session_types
    from atlas.core.session import AgentContext

    frame = pd.DataFrame({"x": [1.0, 2.0, None]})
    model = ExploratoryDataAnalysis(frame)
    legacy_type = type("AnalisisDatosExploratorio", (), {"__module__": "framework.eda"})
    legacy = legacy_type()
    legacy.__dict__.update(model.__dict__)
    context = AgentContext(legacy)
    context.pending["stale-ticket"] = object()
    state = {"eda": legacy, "agent_context": context, "experiments": [], "models": {}}
    upgrade_session_types(state)
    assert type(state["eda"]) is ExploratoryDataAnalysis
    pd.testing.assert_frame_equal(state["eda"].df, frame)
    assert context.eda is state["eda"]
    assert not context.pending
