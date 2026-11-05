"""Executable end-to-end evidence; no mocked analytical results or network services."""

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, make_blobs

from atlas.controllers.agent_controller import AgentController
from atlas.controllers.automatic_eda_controller import AutomaticEDAController, EDALimits
from atlas.controllers.dataset_controller import DatasetController
from atlas.controllers.workflow_controller import WorkflowController
from atlas.core.schemas import ToolCall
from atlas.services.dataset_service import DatasetService


def import_csv(frame, name):
    data = BytesIO(frame.to_csv(index=False).encode())
    data.name = name
    loaded = DatasetController.load(data).df
    state = {"navigation": "📂 Datos", "eda_limits": EDALimits(max_charts=0)}
    DatasetService.activate(state, loaded, name)
    WorkflowController.synchronize(state)
    execution = state["automatic_eda"][AutomaticEDAController.key(state["eda"])]
    assert execution.profile["rows"] == len(frame)
    return state


def authorized_training(state, agent_name, tool, parameters):
    context = AgentController.context_for(state)
    agent = AgentController(context, agent_name, state=state, offline=True)
    agent.open()
    before = len(state["experiments"])
    request = agent.request(
        "Entrenar alternativa solicitada", [ToolCall(tool=tool, parameters=parameters)]
    )
    pending = agent.run(request)
    assert pending.status == "pending_confirmation" and len(state["experiments"]) == before
    result = agent.confirm(pending.confirmation_id)
    assert result.status == "success", result.message
    agent.finish(result)
    WorkflowController.synchronize(state)
    assert len(state["experiments"]) == before + 1
    return state["experiments"][-1]


def main():
    evidence = {}
    iris = load_iris(as_frame=True).frame
    state = import_csv(iris, "iris.csv")
    evidence["csv_import"] = {
        "rows": len(iris),
        "eda_operations": len(next(iter(state["automatic_eda"].values())).operations),
    }
    state.update(
        navigation="🎯 Clasificación",
        analysis_selection={"target": "target", "predictors": list(iris.columns[:-1])},
    )
    params = {"target": "target", "predictors": list(iris.columns[:-1]), "cv": 3}
    first = authorized_training(
        state, "ClassificationAgent", "classification.train", {**params, "algorithm": "logistic"}
    )
    second = authorized_training(
        state, "ClassificationAgent", "classification.train", {**params, "algorithm": "tree"}
    )
    evidence["classification"] = {
        "first": first.metrics_for_context(),
        "second": second.metrics_for_context(),
    }
    evidence["comparison"] = state["intelligence_feedback"]["ModelComparisonAgent"].model_dump(
        mode="json"
    )
    assert "comparación" in evidence["comparison"]["title"].lower()
    values, _ = make_blobs(n_samples=90, centers=3, random_state=42)
    clusters = pd.DataFrame(values, columns=["x", "y"])
    state = import_csv(clusters, "clusters.csv")
    state.update(navigation="🧩 Clustering", analysis_selection={"predictors": ["x", "y"]})
    result = authorized_training(
        state,
        "ClusteringAgent",
        "cluster.train",
        {"algorithm": "kmeans", "columns": ["x", "y"], "parameters": {"n_clusters": 3}},
    )
    evidence["clustering"] = result.metrics_for_context()
    assert evidence["clustering"]["clusters"] == 3
    diabetes = load_diabetes(as_frame=True).frame
    state = import_csv(diabetes, "diabetes.csv")
    state.update(
        navigation="📈 Regresión",
        analysis_selection={"target": "target", "predictors": list(diabetes.columns[:-1])},
    )
    result = authorized_training(
        state,
        "RegressionAgent",
        "regression.train",
        {
            "algorithm": "ridge",
            "target": "target",
            "predictors": list(diabetes.columns[:-1]),
            "cv": 3,
        },
    )
    evidence["regression"] = result.metrics_for_context()
    evidence["regression_feedback"] = state["intelligence_feedback"]["RegressionAgent"].model_dump(
        mode="json"
    )
    output = Path(__file__).resolve().parents[1] / "docs" / "event_demo_results.json"
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Five real workflows verified; evidence saved to {output.name}")


if __name__ == "__main__":
    main()
