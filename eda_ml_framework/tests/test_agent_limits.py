"""Límites de recursos y aprobación aplicada a propuestas costosas."""

import pandas as pd
import pytest

from agents import AgentContext, EDAAgent
from contracts import ToolCall
from framework import AnalisisDatosExploratorio, Regresion
from framework.exceptions import FrameworkError


def test_costly_preview_confirmation_and_caps(mixed):
    context = AgentContext(AnalisisDatosExploratorio(mixed))
    agent = EDAAgent(context)
    pending = agent.run(
        agent.request(
            "Imputar KNN",
            [ToolCall(tool="eda.preview", parameters={"operation": "knn", "columns": ["numero"]})],
        )
    )
    assert pending.status == "pending_confirmation"
    assert not context.proposals
    assert agent.confirm(pending.confirmation_id).status == "success"
    assert context.eda.df.numero.isna().sum() == 1
    assert len(context.proposals) == 1
    for _ in range(4):
        context.propose(mixed, "test")
    with pytest.raises(ValueError, match="cinco"):
        context.propose(mixed, "test")
    for _ in range(100):
        context.artifact(object())
    with pytest.raises(ValueError, match="100 artefactos"):
        context.artifact(object())


def test_data_and_search_guards(regression):
    high_cardinality = pd.DataFrame({"id": [str(i) for i in range(4000)]})
    with pytest.raises(FrameworkError, match="10 millones"):
        AnalisisDatosExploratorio(high_cardinality).transformar("onehot", ["id"])
    large = pd.DataFrame({"value": range(5001)})
    with pytest.raises(FrameworkError, match="5000"):
        AnalisisDatosExploratorio(large).transformar("knn", ["value"])
    with pytest.raises(FrameworkError, match="candidatos"):
        Regresion(regression).entrenar(
            "forest", "target", ["a"], cv=2, busqueda={"n_estimators": [1000000]}
        )
    with pytest.raises(FrameworkError, match="MLP"):
        Regresion(regression).entrenar(
            "mlp", "target", ["a"], cv=2, parametros={"hidden_layer_sizes": [1000000]}
        )


def test_schema_hash_and_preview_metadata(mixed):
    eda = AnalisisDatosExploratorio(mixed)
    original_version = eda.dataset_version
    eda.transformar("median", ["numero"])
    assert eda.dataset_version.operation == "carga"
    assert eda.dataset_version.content_hash == original_version.content_hash
    renamed = mixed.rename(columns={"numero": "otro"})
    assert (
        AnalisisDatosExploratorio(renamed).dataset_version.dataset_id != original_version.dataset_id
    )


def test_ovo_scores_are_not_ovr(iris):
    from framework import Clasificacion

    result = Clasificacion(iris).entrenar(
        "svm_rbf",
        "target",
        list(iris.columns[:-1]),
        cv=2,
        parametros={"decision_function_shape": "ovo"},
    )
    assert "ROC_AUC_macro_ovr" not in result.metrics
    assert result.details["confusion"].shape == (3, 3)


def test_shared_columns_in_plot(mixed):
    eda = AnalisisDatosExploratorio(mixed)
    assert eda.grafico("scatter", "numero", "constante", color="numero") is not None


def test_ordinal_high_cardinality_is_available():
    from framework import NoSupervisado

    frame = pd.DataFrame({"category": [f"c{i}" for i in range(1001)]})
    assert NoSupervisado(frame).preparar(["category"], encoding="ordinal").shape == (1001, 1)
    with pytest.raises(FrameworkError, match="categorías"):
        NoSupervisado(frame).preparar(["category"], encoding="onehot")


def test_nullable_statistics_are_serializable():
    frame = pd.DataFrame({"empty": pd.Series([pd.NA] * 3, dtype="Float64")})
    agent = EDAAgent(AgentContext(AnalisisDatosExploratorio(frame)))
    response = agent.run(
        agent.request(
            "Estadísticas", [ToolCall(tool="eda.statistics", parameters={"columns": ["empty"]})]
        )
    )
    assert response.status == "success"
    assert response.tools[0].result.metrics["empty.media"] is None
    assert "null" in response.model_dump_json()


def test_comparison_defaults_follow_problem(iris):
    from agents import ClassificationAgent
    from frontend.components.agent_panel import defaults_for

    context = AgentContext(AnalisisDatosExploratorio(iris))
    agent = ClassificationAgent(context)
    for algorithm in ["logistic", "tree"]:
        pending = agent.run(
            agent.request(
                "Entrenar",
                [
                    ToolCall(
                        tool="classification.train",
                        parameters={
                            "algorithm": algorithm,
                            "target": "target",
                            "predictors": list(iris.columns[:-1]),
                            "cv": 2,
                        },
                    )
                ],
            )
        )
        assert agent.confirm(pending.confirmation_id).status == "success"
    defaults = defaults_for("comparison.compare", context, {"clasificacion"})
    assert defaults["metric"] == "Accuracy" and defaults["direction"] == "max"
    assert len(defaults["experiment_ids"]) == 2
