"""Demonstrate all five agents using real public-data calculations, not fabricated LLM output."""

import argparse
import json
from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris

from atlas.agents.agent import AgentFactory, NoProvider
from atlas.core.schemas import ToolCall
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Explicitly authorize the small public-data training demonstration.",
    )
    args = parser.parse_args()
    iris = load_iris(as_frame=True).frame
    features = list(iris.columns[:-1])
    shared = AgentContext(ExploratoryDataAnalysis(iris))
    responses = []

    def execute(kind, context, tool, parameters=None):
        agent = AgentFactory.get_agent(kind, context, NoProvider())
        response = agent.run(
            agent.request(
                "Demostración explícita con datos públicos",
                [ToolCall(tool=tool, parameters=parameters or {})],
            )
        )
        if response.status == "pending_confirmation" and args.confirm_training:
            response = agent.confirm(response.confirmation_id)
        responses.append(response)
        if response.status not in {"success", "pending_confirmation"}:
            raise RuntimeError(response.model_dump_json())

    execute("eda", shared, "eda.profile")
    execute(
        "clustering",
        shared,
        "cluster.train",
        {"algorithm": "kmeans", "columns": features, "parameters": {"n_clusters": 3}},
    )
    for algorithm in ["logistic", "tree"]:
        execute(
            "classification",
            shared,
            "classification.train",
            {"algorithm": algorithm, "target": "target", "predictors": features, "cv": 3},
        )
    diabetes = load_diabetes(as_frame=True).frame
    regression = AgentContext(ExploratoryDataAnalysis(diabetes))
    execute(
        "regression",
        regression,
        "regression.train",
        {
            "algorithm": "ridge",
            "target": "target",
            "predictors": list(diabetes.columns[:-1]),
            "cv": 3,
        },
    )
    classified = [
        experiment.id for experiment in shared.experiments if experiment.problem == "clasificacion"
    ]
    if len(classified) == 2:
        execute(
            "comparison",
            shared,
            "comparison.compare",
            {"experiment_ids": classified, "metric": "F1_macro", "direction": "max"},
        )
    output = Path(__file__).resolve().parents[1] / "artifacts" / "mvc-demo.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "mode": "offline_framework_tools_not_llm_inference",
                "responses": [response.model_dump(mode="json") for response in responses],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for response in responses:
        print(f"{response.agent}: {response.status} · {len(response.tools)} tool results")
    print(output)


if __name__ == "__main__":
    main()
