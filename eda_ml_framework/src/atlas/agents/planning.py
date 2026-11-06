"""Planning for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.schemas import (
    ToolCall,
)

from .defaults import default_arguments
from .policies import COMMON_POLICIES, SPECIALIST_POLICIES, DecisionPolicy


class AgentPlanningMixin:
    """Shared orchestration behavior; calculations remain in the Model layer."""

    def decision_policies(self) -> tuple[DecisionPolicy, ...]:
        common = COMMON_POLICIES if "eda.profile" in self.get_allowed_tools() else ()
        return common + SPECIALIST_POLICIES[self.name]

    def local_calls(self, goal: str) -> list[ToolCall]:
        """Evaluate explicit prerequisites using model-supplied metadata, without metrics."""
        if self.name == "EDAAgent" and goal.startswith("prepare_"):
            if self.context.selections.get("workspace_page") != "preparation":
                raise ValueError("Esta propuesta pertenece a Preparación.")
            option = next(
                (
                    item
                    for item in self.preparation_options(self.context.metadata())
                    if item[0] == goal
                ),
                None,
            )
            if not option:
                raise ValueError("La propuesta ya no corresponde al dataset actual.")
            return [ToolCall(tool="eda.preview", parameters=option[1], reason=option[4])]
        if self.name == "ClusteringAgent" and goal == "compare_groups":
            metadata = self.context.metadata()
            if not 10 <= metadata["rows"] <= 5000:
                raise ValueError("La búsqueda guiada admite entre 10 y 5000 registros.")
            parameters = default_arguments("cluster.train", self.context, set(self.problems))
            selection = metadata.get("active_selection", {}).get("analysis_selection") or {}
            if selection.get("predictors"):
                parameters["columns"] = selection["predictors"]
            unsafe = self.context.metadata().get("feature_roles", {})
            flagged = set(
                unsafe.get("identifiers", [])
                + unsafe.get("dates", [])
                + unsafe.get("constants", [])
            )
            if flagged.intersection(parameters["columns"]):
                raise ValueError(
                    "Revisa identificadores, fechas o constantes en las variables elegidas antes de comparar grupos."
                )
            parameters["scaler"] = selection.get("scaler", "standard")
            return [
                ToolCall(
                    tool="cluster.train",
                    parameters={
                        **parameters,
                        "algorithm": "kmeans",
                        "parameters": {"n_clusters": k},
                    },
                    reason="Comparar cantidades de grupos con las mismas variables y preprocesamiento.",
                )
                for k in range(2, 5)
            ]
        policy = next((item for item in self.decision_policies() if item.goal == goal), None)
        if policy is None:
            raise ValueError(
                "El modo local admite objetivos guiados, no lenguaje libre. Selecciona una acción del panel."
            )
        metadata = self.context.metadata()
        selection = metadata.get("active_selection", {}).get("analysis_selection") or {}
        columns = metadata["columns"]
        numeric = [
            c["name"]
            for c in columns
            if any(t in c["dtype"].lower() for t in ("int", "float", "double"))
        ]
        target = selection.get("target")
        experiments = [
            e
            for e in metadata["experiments"]
            if e["problem"] in self.problems
            and e["dataset_id"]
            in {self.context.dataset.dataset_id, self.context.dataset.content_hash}
        ]
        if self.name in {"ClassificationAgent", "RegressionAgent"} and target:
            experiments = [
                item for item in experiments if item["validation"].get("target") == target
            ]
        comparison_ids = selection.get("experiment_ids") or (
            default_arguments("comparison.compare", self.context, set(self.problems))[
                "experiment_ids"
            ]
            if policy.tool == "comparison.compare"
            else []
        )
        checks = {
            "dataset": (metadata["rows"] > 0, "Confirma un dataset con registros."),
            "numeric": (bool(numeric), "Selecciona variables numéricas para este diagnóstico."),
            "numeric_pair": (len(numeric) >= 2, "Se necesitan al menos dos variables numéricas."),
            "features": (metadata["column_count"] > 0, "Selecciona variables predictoras."),
            "sample": (
                metadata["rows"] >= 10 and bool(numeric),
                "Esta política inicial requiere diez registros y variables numéricas.",
            ),
            "target": (
                target is not None,
                "Selecciona primero la variable objetivo en el formulario y vuelve a solicitar asistencia.",
            ),
            "numeric_target": (
                target in numeric,
                "Selecciona un objetivo numérico en el formulario de regresión.",
            ),
            "experiment": (
                bool(experiments),
                "Entrena primero un modelo en este módulo y dataset.",
            ),
            "comparison": (
                len(comparison_ids) >= 2,
                "Selecciona al menos dos modelos entrenados con los mismos datos y separación de prueba antes de comparar.",
            ),
        }
        allowed, reason = checks[policy.prerequisite]
        if not allowed:
            raise ValueError(reason)
        parameters = default_arguments(policy.tool, self.context, set(self.problems))
        if self.name == "ClusteringAgent" and policy.tool.startswith("cluster."):
            if metadata["rows"] > 5000:
                raise ValueError(
                    "El agrupamiento interactivo admite hasta 5.000 filas. Selecciona una muestra en el formulario y ejecuta allí; el agente no utiliza esa muestra automáticamente."
                )
            if "predictors" in selection:
                if not selection["predictors"]:
                    raise ValueError(
                        "Selecciona al menos una variable de agrupamiento antes de continuar."
                    )
                parameters["columns"] = selection["predictors"]
            roles = metadata.get("feature_roles", {})
            flagged = set(
                roles.get("identifiers", []) + roles.get("dates", []) + roles.get("constants", [])
            )
            if flagged.intersection(parameters["columns"]):
                raise ValueError(
                    "Quita identificadores, fechas sin transformar y constantes antes de usar la búsqueda guiada. Puedes revisar la configuración primero."
                )
            if "scaler" in self.gateway.tools[policy.tool].arguments.model_fields:
                parameters["scaler"] = selection.get("scaler", "standard")
        if policy.tool == "comparison.compare" and selection.get("experiment_ids"):
            parameters = {
                key: selection[key]
                for key in ("experiment_ids", "metric", "direction")
                if key in selection
            }
        if policy.tool in {"model.metrics", "model.plot"} and experiments:
            parameters["experiment_id"] = experiments[-1]["id"]
        if policy.tool == "model.plot" and self.name == "RegressionAgent":
            parameters["kind"] = "residuals"
        if policy.tool == "analysis.configure":
            parameters.update(
                scaler="standard",
                algorithm={
                    "ClusteringAgent": "kmeans",
                    "ClassificationAgent": "logistic",
                    "RegressionAgent": "ridge",
                }[self.name],
            )
        if policy.tool.endswith(".train"):
            for key in ("target", "predictors", "test_size", "cv", "random_state", "scaler"):
                if (
                    key in selection
                    and key in self.gateway.tools[policy.tool].arguments.model_fields
                ):
                    parameters[key] = selection[key]
        if policy.tool == "cluster.elbow":
            parameters["max_k"] = min(8, metadata["rows"] - 1)
        return [
            ToolCall(
                tool=policy.tool,
                parameters=parameters,
                reason=f"Política local explícita: {policy.label}. Configuración inicial, no óptimo demostrado.",
            )
        ]

    def autonomous_plan(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Select bounded existing capabilities from model-produced evidence, without metrics."""
        columns = evidence.get("columns", [])
        eligible = [
            item
            for item in columns
            if item["name"] in evidence.get("suggested", [])
            and item["unique"] > 1
            and item["missing_percentage"] < 80
        ]
        if self.name == "EDAAgent":
            preferred = {
                "target",
                "label",
                "class",
                "clase",
                "objetivo",
                "price",
                "precio",
                "churn",
                "outcome",
                "sales",
                "ventas",
                "diagnosis",
            }
            candidates = [item for item in eligible if item["name"].lower() in preferred]
            chosen = evidence.get("target")
            if chosen is None:
                chosen = (candidates or eligible)[-1]["name"] if eligible else None
            return {
                "target": chosen,
                "provisional": evidence.get("target") is None,
                "reason": (
                    f"Usé «{chosen}» porque tú la elegiste como objetivo."
                    if evidence.get("target") is not None
                    else f"Propuse «{chosen}» porque su nombre sugiere una respuesta a predecir. Es una hipótesis; confirma que coincide con tu intención."
                    if candidates
                    else f"No encontré un nombre que identifique un objetivo claro. Usé provisionalmente «{chosen}», la última columna compatible; no hay evidencia de que sea lo que necesitas predecir. Cambia el objetivo si no tiene sentido."
                ),
            }
        if self.name == "ClusteringAgent":
            features = [
                item["name"]
                for item in eligible
                if item["name"] != evidence.get("target")
                and (item["unique"] <= 30 or item["name"] in evidence.get("numeric", []))
            ][:12]
            return {
                "features": features,
                "trials": [("kmeans", 2), ("kmeans", 3), ("kmeans", 4), ("agglomerative", 3)],
                "metric": "Silhouette",
                "direction": "max",
                "reason": "Comparo grupos compactos y una estructura por niveles, con 2–4 grupos y variables equilibradas. Excluyo códigos, fechas sin preparar, constantes y categorías demasiado numerosas. La silueta orienta la separación, no demuestra que los grupos sean útiles.",
            }
        if self.name in {"ClassificationAgent", "RegressionAgent"}:
            features = [
                item["name"]
                for item in eligible
                if item["name"] != evidence.get("target")
                and (item["unique"] <= 30 or item["name"] in evidence.get("numeric", []))
            ][:12]
            classification = self.name == "ClassificationAgent"
            return {
                "features": features,
                "algorithms": ["logistic", "tree", "forest"]
                if classification
                else ["ridge", "tree", "forest"],
                "metric": "CV_media",
                "direction": "max",
                "cv": 3,
                "reason": "Pruebo una alternativa sencilla, un árbol y un conjunto de árboles con las mismas variables y partición. Elijo según validación cruzada de entrenamiento; en clasificación equilibra el rendimiento entre categorías y en regresión considera los errores grandes. La prueba final queda para describir el rendimiento.",
            }
        return {
            "reason": "Comparo únicamente resultados con el mismo dataset, objetivo, variables y partición. La recomendación se limita a las alternativas y al criterio evaluados."
        }
