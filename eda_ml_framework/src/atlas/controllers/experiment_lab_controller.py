"""Coordinate consent, one bounded evaluation per step, persistence and events."""

from hashlib import sha256

import numpy as np

from atlas.core.events import EventType, emit
from atlas.models.methodological_validation import MethodologicalValidation
from atlas.models.provenance import fingerprint
from atlas.models.scientific_experiment import (
    METRICS,
    TASK_MODELS,
    ExperimentalAlternative,
    ScientificExperiment,
    ScientificExperimentDefinition,
    now,
    parameter_options,
)
from atlas.services.experiment_service import ExperimentService

__all__ = [
    "METRICS",
    "ExperimentLabController",
    "ExperimentalAlternative",
    "ScientificExperimentDefinition",
    "parameter_options",
]


class ExperimentLabController:
    validate = staticmethod(ScientificExperiment.validate)
    enforce = staticmethod(MethodologicalValidation.enforce)
    summary = staticmethod(ScientificExperiment.summary)
    charts = staticmethod(ScientificExperiment.charts)
    interpretation = staticmethod(ScientificExperiment.interpretation)
    compare_investigations = staticmethod(ScientificExperiment.compare_investigations)
    manifest = staticmethod(ScientificExperiment.manifest)

    @staticmethod
    def algorithms(task):
        return TASK_MODELS[task].algorithms()

    @staticmethod
    def prepare(state, definition, frame=None, parent_id=None):
        source = state["eda"].df if frame is None else frame
        if fingerprint(source) != definition.dataset.content_hash:
            raise ValueError("La versión de datos no coincide con el diseño.")
        if parent_id is None:
            design_hash = sha256(definition.model_dump_json().encode()).hexdigest()
            for existing in state.get("scientific_investigations", {}).values():
                if existing["design_hash"] == design_hash and existing["status"] == "ready":
                    state["active_investigation"] = existing["id"]
                    return existing
        run = ScientificExperiment.create(definition, source, parent_id)
        ExperimentService.remember_investigation(state, run)
        emit(
            state,
            EventType.LAB_DESIGN_CHANGED,
            {"investigation_id": run["id"], "problem": definition.task},
        )
        return run

    @staticmethod
    def authorize(state, run_id, *, confirm=False, retry=False):
        if not confirm:
            raise ValueError("Confirma explícitamente el entrenamiento y su presupuesto.")
        run = state["scientific_investigations"][run_id]
        if run["status"] == "completed":
            raise ValueError(
                "Esta investigación ya terminó. Crea una reproducción con un identificador nuevo."
            )
        MethodologicalValidation.enforce(
            ScientificExperiment.validate(run["definition"], run["dataset"])
        )
        if sha256(run["definition"].model_dump_json().encode()).hexdigest() != run["design_hash"]:
            raise ValueError("El diseño cambió; crea una propuesta nueva antes de confirmar.")
        pending = run["definition"].evaluations - len(run["artifacts"])
        if len(state.get("experiments", [])) + pending > 30:
            raise ValueError("La ejecución superaría el límite de 30 modelos por sesión.")
        if retry:
            for key, record in list(run["evaluations"].items()):
                if record["status"] == "failed":
                    run["attempts"].append(record)
                    del run["evaluations"][key]
        run["approved_hash"] = run["design_hash"]
        state.pop("lab_exports", None)
        run["status"] = "running"
        run.setdefault("started_at", now())
        run["completed_at"] = None
        emit(
            state,
            EventType.LAB_STARTED,
            {"investigation_id": run_id, "problem": run["definition"].task},
        )

    @staticmethod
    def cancel(state, run_id):
        run = state["scientific_investigations"][run_id]
        if run["status"] == "running":
            state.pop("lab_exports", None)
            run["status"] = "cancelled"
            run["completed_at"] = now()
            run.pop("approved_hash", None)
            emit(state, EventType.LAB_UPDATED, {"investigation_id": run_id, "status": "cancelled"})

    @staticmethod
    def step(state, run_id):
        run = state["scientific_investigations"][run_id]
        definition = run["definition"]
        if (
            run["status"] != "running"
            or run.get("approved_hash") != sha256(definition.model_dump_json().encode()).hexdigest()
        ):
            raise ValueError("Esta evaluación no tiene una confirmación válida.")
        MethodologicalValidation.enforce(ScientificExperiment.validate(definition, run["dataset"]))
        for index, alternative in enumerate(definition.alternatives):
            for seed in definition.seeds:
                key = f"{index}:{seed}"
                if key in run["evaluations"]:
                    continue
                record = {
                    "alternative": alternative.name,
                    "seed": seed,
                    "started_at": now(),
                    "status": "running",
                }
                run["current_operation"] = f"{alternative.name} · semilla {seed}"
                try:
                    if len(state.setdefault("experiments", [])) >= 30:
                        raise ValueError("Se alcanzó el límite de 30 modelos por sesión.")
                    result, engine = ScientificExperiment.evaluate(
                        definition, run["dataset"], alternative, seed
                    )
                    value = result.metrics.get(definition.metric)
                    if value is None or not np.isfinite(value):
                        raise ValueError(
                            "La métrica principal no está definida para esta solución; no se sustituye por cero."
                        )
                    # Enforce actual matched evaluation rows, not only equal seed labels.
                    if definition.task != "clustering":
                        for previous_key, previous in run["artifacts"].items():
                            if run["evaluations"][previous_key][
                                "seed"
                            ] == seed and not previous.details["y_test"].equals(
                                result.details["y_test"]
                            ):
                                raise ValueError(
                                    "Las alternativas no utilizan las mismas etiquetas y filas de evaluación."
                                )
                    state.setdefault("models", {})
                    state.setdefault("metrics", {})
                    ExperimentService.remember(state, result, engine)
                    run["artifacts"][key] = result
                    if result.details.get("lab_warnings"):
                        warning = f"{alternative.name}: el ajuste emitió advertencias numéricas o de convergencia; revisa los detalles antes de concluir."
                        if warning not in run["warnings"]:
                            run["warnings"].append(warning)
                    for finding in result.details.get("methodological_findings", []):
                        warning = finding.title + ". " + finding.explanation
                        if warning not in run["warnings"]:
                            run["warnings"].append(warning)
                    record.update(
                        status="completed",
                        value=float(value),
                        seconds=result.seconds,
                        experiment_id=result.id,
                        metrics=result.metrics_for_context(),
                        configuration=dict(result.config),
                        warnings=result.details.get("lab_warnings", []),
                    )
                    import json

                    estimator = (
                        result.model.named_steps["model"]
                        if definition.task != "clustering"
                        else result.model
                    )
                    record["estimator_parameters"] = json.loads(
                        json.dumps(estimator.get_params(deep=False), default=str)
                    )
                except (ValueError, TypeError, ImportError, RuntimeError) as exc:
                    record.update(status="failed", error=str(exc))
                record["completed_at"] = now()
                run["evaluations"][key] = record
                emit(
                    state,
                    EventType.LAB_UPDATED,
                    {
                        "investigation_id": run_id,
                        "completed": len(run["artifacts"]),
                        "attempted": len(run["evaluations"]),
                        "problem": definition.task,
                    },
                )
                if len(run["evaluations"]) == definition.evaluations:
                    ExperimentLabController.finish(state, run)
                return record
        ExperimentLabController.finish(state, run)
        return None

    @staticmethod
    def finish(state, run):
        state.pop("lab_exports", None)
        run["status"] = (
            "completed" if len(run["artifacts"]) == run["definition"].evaluations else "partial"
        )
        run["completed_at"] = now()
        run.pop("approved_hash", None)
        try:
            run["uncertainty"] = ScientificExperiment.bootstrap(run)
        except ValueError as exc:
            run["warnings"].append(str(exc))
        emit(
            state,
            EventType.LAB_COMPLETED,
            {
                "investigation_id": run["id"],
                "status": run["status"],
                "problem": run["definition"].task,
            },
        )

    @staticmethod
    def execute(state, run_id, *, confirm=False):
        ExperimentLabController.authorize(state, run_id, confirm=confirm, retry=True)
        run = state["scientific_investigations"][run_id]
        while run["status"] == "running":
            ExperimentLabController.step(state, run_id)
        return run

    @staticmethod
    def reproduce(state, run_id, *, current_dataset=False):
        source = state["scientific_investigations"][run_id]
        definition = source["definition"]
        frame = source["dataset"]
        if current_dataset:
            frame = state["eda"].df
            definition = definition.model_copy(update={"dataset": state["eda"].dataset_version})
        return ExperimentLabController.prepare(state, definition, frame, parent_id=run_id)
