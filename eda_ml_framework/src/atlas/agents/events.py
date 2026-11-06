"""Events for the modular ATLAS assistant system."""

from atlas.core.schemas import (
    IntelligenceFeedback,
    ToolCall,
)


class AgentEventMixin:
    """Shared orchestration behavior; calculations remain in the Model layer."""

    def on_event(self, event) -> IntelligenceFeedback:
        """Interpret stored evidence through specialist policies; never train from an event."""
        metadata = self.context.metadata()
        advice = self.advice()
        evidence = [
            f"{metadata['rows']:,} registros · {metadata['column_count']} variables",
            f"{metadata['missing_cells']:,} celdas vacías · {metadata['additional_duplicates']:,} filas repetidas adicionales",
        ]
        warnings = []
        candidates = []
        title, message, priority = advice.title, advice.next_step, "information"
        state = self.context.application_state or {}
        if event.type == "ANALYSIS_FAILED":
            return IntelligenceFeedback(
                agent=self.name,
                event_id=event.id,
                title="La operación necesita una revisión",
                message="No se pudo completar la última acción. Revisa el requisito indicado antes de intentar otra alternativa.",
                evidence=evidence,
                warnings=[str(event.payload.get("message", "Revisa la selección actual."))],
                priority="critical",
            )
        if self.name == "EDAAgent":
            execution = state.get("automatic_eda", {}).get(self.context.key)
            if execution and execution.status != "running":
                title = "Tu exploración inicial está disponible"
                message = "Revisé el perfil, la calidad y los análisis compatibles de tus datos. Puedes consultar los resultados y qué operaciones se omitieron."
                if execution.status in {"partial", "failed"}:
                    warnings.append(
                        "Algunas operaciones se omitieron por límites o no pudieron completarse. Consulta el registro del EDA."
                    )
                if execution.status == "failed":
                    title, message, priority = (
                        "No se completó la exploración inicial",
                        "Tus datos se conservan. Abre EDA para revisar el error e intentar actualizar el análisis.",
                        "critical",
                    )
                excluded = set(
                    execution.roles.get("identifiers", [])
                    + execution.roles.get("dates", [])
                    + execution.roles.get("constants", [])
                )
                for column in execution.columns:
                    name = column["name"]
                    if (
                        name in excluded
                        or column["missing_percentage"] > 20
                        or column["unique"] < 2
                    ):
                        continue
                    named_target = any(
                        token in name.lower()
                        for token in (
                            "target",
                            "price",
                            "precio",
                            "sales",
                            "ventas",
                            "label",
                            "outcome",
                            "churn",
                            "diagnosis",
                        )
                    )
                    if not named_target:
                        continue
                    numeric = any(
                        kind in column["dtype"].lower() for kind in ("int", "float", "double")
                    )
                    objective = (
                        "predecir una cantidad"
                        if numeric and column["unique"] > 20
                        else "distinguir categorías"
                        if column["unique"] <= 20
                        else None
                    )
                    if objective:
                        candidates.append(
                            {
                                "column": name,
                                "objective": objective,
                                "limitation": "Candidato por nombre, tipo y variedad de valores; tú decides si corresponde al objetivo. Revisa posibles fugas de información.",
                            }
                        )
                    if len(candidates) >= 3:
                        break
            if metadata["missing_cells"] or metadata["additional_duplicates"]:
                title = "Hay datos que conviene revisar primero"
                message = advice.reason + " " + advice.next_step
                priority = "warning"
        else:
            selection = metadata.get("active_selection", {}).get("analysis_selection") or {}
            relevant = [
                item
                for item in metadata["experiments"]
                if item["problem"] in self.problems
                and item["dataset_id"]
                in {self.context.dataset.dataset_id, self.context.dataset.content_hash}
                and (
                    self.name not in {"ClassificationAgent", "RegressionAgent"}
                    or not selection.get("target")
                    or item["validation"].get("target") == selection["target"]
                )
            ]
            if self.name == "ClassificationAgent" and selection.get("target") and not relevant:
                try:
                    call = ToolCall(
                        tool="classification.class_distribution",
                        parameters={"column": selection["target"]},
                    )
                    output = self.read_event_evidence(call)
                    rows = output.result.tables.get("frecuencias", []) if output.result else []
                    rare = next((row for row in rows if row.get("absoluta", 0) < 2), None)
                    dominant = next((row for row in rows if row.get("relativa", 0) > 0.8), None)
                    if rare:
                        title, message, priority = (
                            "Faltan ejemplos de alguna categoría",
                            "Hay una categoría con un solo registro. Cambiar de modelo no lo resuelve. Revisa si elegiste un ID o una cantidad; para cantidades utiliza Regresión.",
                            "critical",
                        )
                    elif dominant:
                        title, message, priority = (
                            "Las categorías tienen tamaños desiguales",
                            "Una categoría reúne más del 80% de los ejemplos. Revisa la detección por categoría y el equilibrio entre aciertos; el porcentaje global puede ocultar errores.",
                            "warning",
                        )
                        evidence.append(
                            f"Categoría más frecuente: {dominant['relativa']:.1%} de los registros."
                        )
                except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                    warnings.append(f"No se pudo comprobar el objetivo: {str(exc)[:240]}")
            if self.name == "ModelComparisonAgent":
                try:
                    call = self.local_calls("compare")[0]
                    output = self.read_event_evidence(call)
                    interpretation = self.interpret_result(output)
                    title = "Comparación disponible con resultados guardados"
                    if self.context.application_state is not None:
                        from atlas.core.events import EventType, emit

                        emit(
                            self.context.application_state,
                            EventType.COMPARISON_UPDATED,
                            call.parameters,
                        )
                    message = interpretation["recommendations"][0]["text"]
                    evidence.append(
                        "Se compararon pruebas con la misma cohorte y según una métrica explícita; no se entrenaron modelos."
                    )
                except (ValueError, TypeError, KeyError, RuntimeError):
                    title, message = advice.title, advice.next_step
            elif relevant:
                try:
                    call = ToolCall(
                        tool="model.metrics", parameters={"experiment_id": relevant[-1]["id"]}
                    )
                    output = self.read_event_evidence(call)
                    interpretation = self.interpret_result(output)
                    title = {
                        "ClusteringAgent": "Tus grupos ya tienen resultados",
                        "ClassificationAgent": "Tu modelo de categorías ya fue evaluado",
                        "RegressionAgent": "Tus predicciones ya fueron evaluadas",
                    }[self.name]
                    message = interpretation["recommendations"][0]["text"]
                    for name, value in relevant[-1]["metrics"].items():
                        if (
                            name
                            in {
                                "MAE",
                                "RMSE",
                                "Accuracy",
                                "F1_macro",
                                "Silhouette",
                                "ruido",
                                "clusters",
                            }
                            and value is not None
                        ):
                            evidence.append(f"{name}: {value:,.3f}")
                except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                    warnings.append(f"No se pudo consultar el resultado guardado: {str(exc)[:200]}")
            if self.name == "ClusteringAgent" and metadata["rows"] > 5000:
                title, message, priority = (
                    "El agrupamiento necesita una muestra",
                    "El límite es 5.000 filas. Selecciona una muestra en el formulario manual; el agente no la usa automáticamente. No se entrenó ningún modelo al abrir esta pantalla.",
                    "warning",
                )
        return IntelligenceFeedback(
            agent=self.name,
            event_id=event.id,
            title=title,
            message=message,
            evidence=evidence,
            warnings=warnings,
            candidates=candidates,
            priority=priority,
        )
