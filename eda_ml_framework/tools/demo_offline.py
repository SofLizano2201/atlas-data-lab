"""Exercise finite goal selection and real scientific execution for all five agents."""

import json
from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris

from atlas.agents.agent import AgentFactory, NoProvider
from atlas.core.schemas import ToolCall
from atlas.core.session import AgentContext
from atlas.models import ExploratoryDataAnalysis


def main():
    iris = load_iris(as_frame=True).frame
    shared = AgentContext(ExploratoryDataAnalysis(iris))
    shared.selections = {
        "analysis_selection": {"target": "target", "predictors": list(iris.columns[:-1]), "cv": 3}
    }
    outputs = []

    def execute(name, context, goal):
        agent = AgentFactory.get_agent(name, context, NoProvider())
        response = agent.run(agent.request(f"guided:{goal}"))
        if response.confirmation_id:
            # Explicitly running this public-data demonstration authorizes its small training jobs.
            response = agent.confirm(response.confirmation_id)
        if response.status != "success":
            raise RuntimeError(response.model_dump_json())
        outputs.append(response)

    execute("eda", shared, "diagnose")
    execute("clustering", shared, "train")
    execute("classification", shared, "classes")
    execute("classification", shared, "train")
    alternative = AgentFactory.get_agent("classification", shared, NoProvider())
    pending = alternative.run(
        alternative.request(
            "Alternativa explícita",
            [
                ToolCall(
                    tool="classification.train",
                    parameters={
                        "algorithm": "tree",
                        "target": "target",
                        "predictors": list(iris.columns[:-1]),
                        "cv": 3,
                    },
                )
            ],
        )
    )
    result = alternative.confirm(pending.confirmation_id)
    assert result.status == "success"
    outputs.append(result)
    diabetes = load_diabetes(as_frame=True).frame
    regression = AgentContext(ExploratoryDataAnalysis(diabetes))
    regression.selections = {
        "analysis_selection": {
            "target": "target",
            "predictors": list(diabetes.columns[:-1]),
            "cv": 3,
        }
    }
    execute("regression", regression, "train")
    execute("comparison", shared, "compare")
    output = Path(__file__).resolve().parents[1] / "artifacts/offline-decisions.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "mode": "deterministic_local_decision_support",
                "responses": [response.model_dump(mode="json") for response in outputs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for response in outputs:
        print(f"{response.agent}: {response.status}")


if __name__ == "__main__":
    main()
