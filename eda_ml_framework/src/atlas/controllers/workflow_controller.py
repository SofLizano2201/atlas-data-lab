"""Emit workflow facts and dispatch events through the common agent contract."""

import json

from atlas.controllers.agent_controller import AgentController
from atlas.controllers.automatic_eda_controller import AutomaticEDAController
from atlas.core.dependencies import build_offline_agent
from atlas.core.events import EventType, emit

AGENT_BY_ROUTE = {
    "🧩 Clustering": "ClusteringAgent",
    "🎯 Clasificación": "ClassificationAgent",
    "📈 Regresión": "RegressionAgent",
    "⚖️ Experimentos": "ModelComparisonAgent",
    "experiment_history": "ModelComparisonAgent",
    "scientific_lab": "ModelComparisonAgent",
}


class WorkflowController:
    @staticmethod
    def training_started(state, problem, algorithm):
        state["training_attempt"] = state.get("training_attempt", 0) + 1
        return emit(
            state,
            EventType.TRAINING_STARTED,
            {
                "problem": problem,
                "algorithm": algorithm,
                "attempt": state.get("training_attempt", 0),
            },
        )

    @staticmethod
    def analysis_failed(state, problem, message):
        return emit(
            state,
            EventType.ANALYSIS_FAILED,
            {
                "problem": problem,
                "message": str(message),
                "attempt": state.get("training_attempt", 0),
            },
        )

    @staticmethod
    def synchronize(state):
        """Observe transitions once; never train or mutate data as a side effect."""
        if "eda" not in state:
            return
        key = AutomaticEDAController.key(state["eda"])
        previous = state.get("workflow_dataset_key")
        if previous != key:
            if previous:
                emit(state, EventType.DATASET_CHANGED)
                emit(state, EventType.DATA_CLEANED)
            state["workflow_dataset_key"] = key
            state["dismissed_intelligence"] = []
            state["intelligence_feedback"] = {}
            AutomaticEDAController.run(state)
        context = AgentController.context_for(state)
        selection = context.selections.get("analysis_selection") or {}
        from atlas.controllers.methodological_controller import MethodologicalController

        MethodologicalController.synchronize(state, selection)
        route = state.get("navigation", "🏠 Inicio")
        snapshot = state.setdefault("workflow_snapshot", {})
        facts = {
            "route": (EventType.WORKSPACE_ENTERED, route),
            "features": (EventType.FEATURES_CHANGED, selection.get("predictors", [])),
            "target": (EventType.TARGET_SELECTED, selection.get("target")),
            "model": (
                EventType.MODEL_SELECTED,
                state.get(
                    "atlas_algorithms_" + str(selection.get("problem")),
                    state.get("atlas_cluster_algorithm"),
                ),
            ),
            "validation": (EventType.VALIDATION_CHANGED, {key: selection.get(key) for key in ("cv", "test_size", "scaler", "encoding")}),
        }
        for field, (kind, value) in facts.items():
            serialized = json.dumps(value, sort_keys=True, default=str)
            snapshot_key = f"{route}:{field}"
            if snapshot.get(snapshot_key) != serialized:
                snapshot[snapshot_key] = serialized
                state["workflow_transition"] = state.get("workflow_transition", 0) + 1
                emit(
                    state,
                    kind,
                    {"route": route, field: value, "transition": state["workflow_transition"]},
                )
        seen = state.setdefault("workflow_experiments", [])
        for experiment in state.get("experiments", []):
            if experiment.id in seen:
                continue
            seen.append(experiment.id)
            payload = {"experiment_id": experiment.id, "problem": experiment.problem}
            emit(
                state,
                EventType.CLUSTERING_COMPLETED
                if experiment.problem == "clustering"
                else EventType.TRAINING_COMPLETED,
                payload,
            )
            emit(state, EventType.MODEL_EVALUATED, payload)
            emit(state, EventType.EXPERIMENT_SAVED, payload)
        agents = {}
        for event in list(state.get("workflow_events", {}).values()):
            if event.status != "pending":
                continue
            if (event.dataset_id, event.dataset_version, event.content_hash) != (
                context.dataset.dataset_id,
                context.dataset.version,
                context.dataset.content_hash,
            ):
                event.status = "outdated"
                continue
            responsible = AGENT_BY_ROUTE.get(route, "EDAAgent")
            targets = {responsible}
            if event.type in {
                EventType.DATASET_CONFIRMED,
                EventType.EDA_COMPLETED,
                EventType.DATA_CLEANED,
            }:
                targets.add("EDAAgent")
            if event.payload.get("problem"):
                targets.add(
                    {
                        "clasificacion": "ClassificationAgent",
                        "regresion": "RegressionAgent",
                        "clustering": "ClusteringAgent",
                        "classification": "ClassificationAgent",
                        "regression": "RegressionAgent",
                    }.get(event.payload["problem"], responsible)
                )
            if event.type in {EventType.EXPERIMENT_SAVED, EventType.COMPARISON_UPDATED}:
                targets.add("ModelComparisonAgent")
            for name in targets:
                try:
                    if name not in agents:
                        agents[name] = build_offline_agent(name, context)[0]
                    feedback = agents[name].on_event(event)
                    state.setdefault("intelligence_feedback", {})[name] = feedback
                except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                    state.setdefault("intelligence_errors", {})[name] = str(exc)
            event.status = "processed"

    @staticmethod
    def refresh_feedback(state):
        """Read current facts on demand without duplicating scientific work."""
        WorkflowController.synchronize(state)
        context = AgentController.context_for(state)
        name = AGENT_BY_ROUTE.get(state.get("navigation"), "EDAAgent")
        if name not in state.setdefault("intelligence_feedback", {}):
            event = emit(
                state,
                EventType.WORKSPACE_ENTERED,
                {"route": state.get("navigation"), "context": context.key},
            )
            state["intelligence_feedback"][name] = build_offline_agent(name, context)[0].on_event(
                event
            )
        return name, state["intelligence_feedback"][name]
