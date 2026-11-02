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


def test_single_agent_file_and_no_scientific_libraries_in_agent_layer():
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
                definitions[node.name] = path
    assert len(definitions) == 6
    assert set(definitions.values()) == {ROOT / "agents" / "agent.py"}
    assert not any(
        name.startswith(("sklearn", "scipy", "numpy", "pandas", "streamlit")) or "views" in name
        for name in imports(ROOT / "agents" / "agent.py")
    )
    assert sorted(path.name for path in (ROOT / "agents").glob("*.py")) == [
        "__init__.py",
        "agent.py",
    ]


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
