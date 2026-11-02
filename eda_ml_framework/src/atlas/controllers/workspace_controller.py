"""Compose real session context and workflow state without scientific calculations."""

from atlas.controllers.eda_controller import EDAController


class WorkspaceController:
    @staticmethod
    def context(state, workflow):
        if "eda" not in state:
            return None
        model = EDAController(state["eda"])
        return {
            **model.profile(),
            "name": state.get("dataset_name", "Dataset"),
            "version": model.dataset_version.version,
            "workflow": workflow,
            "experiments": len(state.get("experiments", [])),
        }

    @staticmethod
    def progress(state):
        """Do not equate navigation with completion of scientific work."""
        return {
            "Datos": "eda" in state,
            "Exploración": bool(state.get("eda_reviewed")),
            "Preparación": "eda" in state and state["eda"].dataset_version.version > 0,
            "Modelado": bool(state.get("experiments")),
            "Evaluación": bool(state.get("experiments")),
            "Exportación": "eda" in state
            and any(
                report.dataset_hash == state["eda"].dataset_version.content_hash
                for report in state.get("reports", [])
            ),
        }
