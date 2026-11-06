"""Upgrade presentation in long-lived local Streamlit processes without losing session data."""

import importlib
import sys

REVISION = "atlas-modular-agents-16"
_runtime_revision = globals().get("_runtime_revision")


def refresh_presentation_runtime():
    """Reload an explicit presentation dependency list once after a local code upgrade."""
    global _runtime_revision
    if _runtime_revision == REVISION:
        return
    for name in (
        "atlas.agents.agent",
        "atlas.agents",
        "atlas.models.experiment",
        "atlas.models.visualization",
        "atlas.controllers.charts",
        "atlas.controllers.automatic_eda_controller",
        "atlas.controllers.autonomous_controller",
        "atlas.controllers.workflow_controller",
        "atlas.views.components.tables",
        "atlas.views.utils.state",
        "atlas.views.components.ui",
        "atlas.views.components.agent_panel",
        "atlas.views.components.layout",
        "atlas.views.components.intelligence_panel",
        "atlas.views.components.autonomous_panel",
        "atlas.views.pages.workspace",
        "atlas.views.pages.modeling",
        "atlas.views.pages.experiments",
        "atlas.views.pages.reports",
    ):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    _runtime_revision = REVISION


def upgrade_session_presentation(state):
    """Replace only visual artifacts and reports; reuse all trained experiments and datasets."""
    if "eda" not in state or state.get("presentation_revision") == REVISION:
        return
    import plotly.graph_objects as go

    from atlas.controllers.automatic_eda_controller import AutomaticEDAController
    from atlas.controllers.autonomous_controller import AutonomousController
    from atlas.models.visualization import style

    execution = AutomaticEDAController.run(state, refresh=True)
    charts = {
        name: item["value"]
        for name, item in execution.operations.items()
        if name.startswith("chart:") and item["status"] == "completed"
    }
    for run in state.get("autonomous_runs", {}).values():
        # Historical dataset runs are not mixed with the active dataset.
        if not run.key.startswith(AutomaticEDAController.key(state["eda"])):
            continue
        run.figures = {
            name: style(go.Figure(graph))
            for name, graph in run.figures.items()
            if not name.startswith("chart:") and isinstance(graph, go.Figure)
        }
        run.figures = {**charts, **run.figures}
        run.report = AutonomousController.report(run, state)
    for name, graph in list(state.get("graphs", {}).items()):
        if isinstance(graph, go.Figure):
            state["graphs"][name] = style(go.Figure(graph))
    state["presentation_revision"] = REVISION

