"""Enforce dependency boundaries and prevent duplicate agent/scientific implementations."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "atlas"


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] + [
        item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names
    ]


def test_models_do_not_depend_on_views_agents_or_streamlit():
    for path in (ROOT / "models").glob("*.py"):
        assert not any(
            "streamlit" in name or "agents" in name or "views" in name for name in imports(path)
        ), path


def test_modular_agents_have_one_definition_and_no_scientific_dependencies():
    definitions = {}
    for path in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef) and node.name in {
                "BaseAgent",
                "EDAAgent",
                "ClusteringAgent",
                "ClassificationAgent",
                "RegressionAgent",
                "ModelComparisonAgent",
            }:
                assert node.name not in definitions, f"Duplicate definition: {node.name}"
                definitions[node.name] = path
    assert len(definitions) == 6
    expected = {
        "BaseAgent": "base.py",
        "EDAAgent": "eda_agent.py",
        "ClusteringAgent": "clustering_agent.py",
        "ClassificationAgent": "classification_agent.py",
        "RegressionAgent": "regression_agent.py",
        "ModelComparisonAgent": "comparison_agent.py",
    }
    assert {name: path.name for name, path in definitions.items()} == expected
    for path in (ROOT / "agents").glob("*.py"):
        assert not any(
            name.startswith(("sklearn", "scipy", "numpy", "pandas", "streamlit")) or "views" in name
            for name in imports(path)
        ), path


def test_legacy_agent_imports_share_the_canonical_classes():
    from atlas.agents import agent
    from atlas.agents.factory import AgentFactory

    for cls in AgentFactory.registry.values():
        assert getattr(agent, cls.__name__) is cls
        assert cls.__module__ != "atlas.agents.agent"


def test_views_use_controllers_and_do_not_fit_or_compute_statistics():
    forbidden_calls = {
        "fit",
        "fit_predict",
        "fit_transform",
        "train_test_split",
        "corr",
        "mean",
        "std",
        "median",
        "sum",
        "idxmin",
        "idxmax",
    }
    for path in (ROOT / "views").rglob("*.py"):
        assert not any(
            name.startswith(("atlas.models", "sklearn", "scipy", "framework"))
            for name in imports(path)
        ), path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_calls
            for node in ast.walk(tree)
        ), path


def test_controllers_do_not_include_llm_transports_or_prompts():
    for path in (ROOT / "controllers").glob("*.py"):
        assert not any(
            name.startswith(("urllib", "httpx", "openai", "requests")) for name in imports(path)
        ), path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "Respond in Spanish" in node.value
            for node in ast.walk(tree)
        ), path
