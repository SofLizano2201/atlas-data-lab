"""Isolated dataset lifecycle and public examples; no Streamlit dependency."""

from collections.abc import MutableMapping
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, make_blobs

from atlas.models import ExploratoryDataAnalysis


class DatasetService:
    @staticmethod
    def upgrade(state):
        from atlas.core.compatibility import upgrade_session_types

        upgrade_session_types(state)

    @staticmethod
    def example(name: str) -> pd.DataFrame:
        if name == "Clusters sintéticos":
            values, _ = make_blobs(n_samples=180, centers=3, n_features=4, random_state=42)
            return pd.DataFrame(values, columns=[f"variable_{i + 1}" for i in range(4)])
        loader = {
            "Iris": load_iris,
            "Diabetes": load_diabetes,
            "Cáncer de mama": load_breast_cancer,
        }[name]
        return loader(as_frame=True).frame

    @staticmethod
    def activate(state: MutableMapping[str, Any], frame: pd.DataFrame, name: str) -> None:
        model = ExploratoryDataAnalysis(frame)
        for key in list(state):
            if key in {
                "new_predictions",
                "representation",
                "agent_context",
                "eda_selection",
                "comparison_selection",
                "eda_chart_request",
                "active_assistant",
            } or key.startswith(
                (
                    "agent_last_",
                    "args_",
                    "history_",
                    "consent_",
                    "analysis_",
                    "new_predictions_",
                    "assistant_flow_",
                    "assistant_technical_",
                )
            ):
                state.pop(key, None)
        state.update(
            original=frame.copy(deep=True),
            eda=model,
            dataset_name=name,
            work=model.df,
            experiments=[],
            models={},
            metrics={},
            graphs={},
            chart_preferences={},
            chart_selections={},
            diagnostics={},
            pending=None,
            preprocessing={},
            selected_variables=[],
            problem=None,
            eda_reviewed=False,
        )
        from atlas.controllers.automatic_eda_controller import AutomaticEDAController
        from atlas.core.events import EventType, emit

        for key in [
            "workflow_events",
            "workflow_snapshot",
            "workflow_experiments",
            "intelligence_feedback",
            "intelligence_errors",
            "dismissed_intelligence",
            "workflow_dataset_key",
        ]:
            state.pop(key, None)
        emit(state, EventType.DATASET_UPLOADED, {"name": name})
        emit(state, EventType.DATASET_CONFIRMED, {"name": name})
        AutomaticEDAController.run(state, progress=state.get("eda_progress_callback"))
