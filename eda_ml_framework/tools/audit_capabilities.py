"""Inventory actual model methods and registered tools without guessing from documentation."""

import ast
import inspect
import json
from pathlib import Path

from atlas.controllers.capabilities import SPECS
from atlas.models import (
    Classification,
    Clustering,
    ExploratoryDataAnalysis,
    Regression,
    SupervisedLearning,
    UnsupervisedLearning,
    metrics,
)
from atlas.models.preprocessing import SCALERS


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tested = set()
    aliases = ast.literal_eval(
        next(
            node.value
            for node in ast.parse((root / "framework/_compat.py").read_text()).body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "METHOD_ALIASES"
                for target in node.targets
            )
        )
    )
    for path in (root / "tests").glob("test_*.py"):
        tested.update(
            aliases.get(node.attr, node.attr)
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig")))
            if isinstance(node, ast.Attribute)
        )
    exposed = {}
    for name, spec in SPECS.items():
        functions = (
            spec.backend_method
            if isinstance(spec.backend_method, tuple)
            else (spec.backend_method,)
        )
        for function in functions:
            exposed.setdefault(function.__qualname__, []).append(name)
    rows = []
    for cls in [
        ExploratoryDataAnalysis,
        UnsupervisedLearning,
        SupervisedLearning,
        Clustering,
        Classification,
        Regression,
    ]:
        for name, descriptor in cls.__dict__.items():
            if name.startswith("_"):
                continue
            function = (
                descriptor.fget
                if isinstance(descriptor, property)
                else descriptor.__func__
                if isinstance(descriptor, (classmethod, staticmethod))
                else descriptor
            )
            if not inspect.isfunction(function):
                continue
            rows.append(
                {
                    "class": cls.__name__,
                    "method": name,
                    "parameters": str(inspect.signature(function)),
                    "expected_behavior": inspect.getdoc(function),
                    "implemented_in": function.__module__,
                    "agent_tools": exposed.get(function.__qualname__, []),
                    "test_reference_exists": name in tested,
                    "verification_note": "A test reference is not a guarantee of coverage for every input. Optional dependencies and abstract hooks are documented separately.",
                }
            )
    tools = [
        {
            "name": name,
            "module": spec.module,
            "read_only": spec.read_only,
            "requires_confirmation": spec.confirmation,
            "input_schema": spec.arguments.model_json_schema(),
            "output_schema": spec.output_schema.model_json_schema(),
        }
        for name, spec in SPECS.items()
    ]
    output = root / "docs" / "capabilities_mvc.json"
    metric_tree = ast.parse(inspect.getsource(metrics))
    metric_labels = sorted(
        {
            key.value
            for node in ast.walk(metric_tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    )
    catalogs = {
        "classification": Classification.algorithms(),
        "regression": Regression.algorithms(),
        "clustering": Clustering.algorithms(),
        "scalers": [*SCALERS, "none"],
        "metric_dictionary_labels": metric_labels,
        "metric_note": "Labels discovered from dictionary literals, not an exhaustive metric availability guarantee; algorithm-specific metrics remain conditional.",
        "optional_dependencies": ["umap-learn", "scikit-learn-extra", "xgboost"],
    }
    output.write_text(
        json.dumps(
            {"methods": rows, "tools": tools, "catalogs": catalogs}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(
        f"Inventoried {len(rows)} owned methods/properties and {len(tools)} actual registered capabilities."
    )


if __name__ == "__main__":
    main()
