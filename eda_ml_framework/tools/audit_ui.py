"""Inventory every statically visible control and its surrounding controller calls."""

import ast
import json
from pathlib import Path

CONTROLS = {
    "button",
    "download_button",
    "file_uploader",
    "selectbox",
    "multiselect",
    "radio",
    "checkbox",
    "slider",
    "number_input",
    "text_input",
    "text_area",
    "plotly_chart",
}


def main():
    root = Path(__file__).resolve().parents[1]
    rows = []
    for path in (root / "src/atlas/views").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
            handlers = sorted(
                {
                    ast.unparse(node.func)
                    for node in calls
                    if isinstance(node.func, ast.Attribute) and node.func.attr not in CONTROLS
                }
            )
            for call in calls:
                if not isinstance(call.func, ast.Attribute) or call.func.attr not in CONTROLS:
                    continue
                label = ast.unparse(call.args[0]) if call.args else "dynamic"
                rows.append(
                    {
                        "view": str(path.relative_to(root)).replace("\\", "/"),
                        "function": function.name,
                        "line": call.lineno,
                        "control": call.func.attr,
                        "label_expression": label,
                        "context_calls": handlers,
                        "verification": "Source wiring inventoried; runtime evidence is recorded in audit_2_0.md and tests. Not exhaustive parameter-combination coverage.",
                    }
                )
    output = root / "docs/ui_controls_2_0.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Inventoried {len(rows)} control declarations, including dynamic labels and shared components."
    )


if __name__ == "__main__":
    main()
