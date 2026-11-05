"""One session-owned experiment store shared by manual and assisted workflows."""

from collections.abc import MutableMapping
from typing import Any


class ExperimentService:
    @staticmethod
    def record(experiments, models, result, engine):
        result.model_version = len(experiments) + 1
        result.config["model_version"] = result.model_version
        experiments.append(result)
        models[result.id] = engine

    @staticmethod
    def remember(state: MutableMapping[str, Any], result: Any, engine: Any) -> None:
        ExperimentService.record(state["experiments"], state["models"], result, engine)
        state["metrics"][result.id] = result.metrics
        from atlas.core.events import EventType, emit

        payload = {"experiment_id": result.id, "problem": result.problem}
        emit(
            state,
            EventType.CLUSTERING_COMPLETED
            if result.problem == "clustering"
            else EventType.TRAINING_COMPLETED,
            payload,
        )
        emit(state, EventType.MODEL_EVALUATED, payload)
        emit(state, EventType.EXPERIMENT_SAVED, payload)
