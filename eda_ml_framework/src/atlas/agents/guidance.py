"""Guidance for the modular ATLAS assistant system."""

from typing import Any

from atlas.core.schemas import (
    AssistantAdvice,
    AssistantIdea,
    AssistantOverview,
)

from .policies import DOMAIN_IDEA_COPY, IDEA_COPY


class AssistantGuidanceMixin:
    """Shared orchestration behavior; calculations remain in the Model layer."""

    def suggestions(self) -> AssistantOverview:
        """Read bounded model metadata; never train, mutate or execute tools on opening."""
        advice = self.advice()
        metadata = self.context.metadata()
        if (
            self.name == "EDAAgent"
            and self.context.selections.get("workspace_page") == "preparation"
        ):
            return self.preparation_overview(metadata)
        ideas: list[AssistantIdea] = []
        domain_priority = {
            "EDAAgent": ["explore", "relationships", "outliers", "diagnose"],
            "ClusteringAgent": [
                "compare",
                "compare_groups",
                "elbow",
                "project",
                "configure",
                "evaluate",
            ],
            "ClassificationAgent": ["classes", "configure", "errors", "train", "evaluate"],
            "RegressionAgent": ["errors", "configure", "train", "evaluate"],
            "ModelComparisonAgent": ["compare", "evaluate"],
        }
        priority = [advice.goal, *domain_priority[self.name], *self.guided_actions()]
        seen = set()
        for goal in priority:
            if goal in seen:
                continue
            seen.add(goal)
            try:
                calls = self.local_calls(goal)
                self._validate(self.request(f"guided:{goal}", calls))
            except (ValueError, TypeError, KeyError):
                continue
            title, description, why, expected = (
                (
                    "Buscar cuántos grupos funcionan mejor",
                    "Prueba de 2 a 4 grupos con las mismas variables.",
                    "El número de grupos debe evaluarse con resultados, no adivinarse.",
                    "Guardaré las alternativas para comparar su separación. Una puntuación alta no garantiza utilidad práctica.",
                )
                if goal == "compare_groups"
                else IDEA_COPY[goal]
            )
            title, description = DOMAIN_IDEA_COPY.get(self.name, {}).get(goal, (title, description))
            call = calls[0]
            effect = {
                "analysis.configure": "Cambiará las opciones del formulario solo si decides continuar. No entrenará un modelo.",
                "cluster.train": "Calculará un agrupamiento y guardará el resultado. Te pediré permiso antes de empezar.",
                "classification.train": "Entrenará un modelo y guardará sus resultados. Te pediré permiso antes de empezar.",
                "regression.train": "Entrenará un modelo y guardará sus resultados. Te pediré permiso antes de empezar.",
            }.get(call.tool, "No modificará tus datos ni volverá a entrenar modelos.")
            if self.requires_confirmation(call) and call.tool not in {
                "analysis.configure",
                "cluster.train",
                "classification.train",
                "regression.train",
            }:
                effect = (
                    "Este cálculo puede tardar. Solo empezará cuando confirmes; no eliminará datos."
                )
            if goal == "compare_groups":
                effect = "Entrenará tres alternativas (2, 3 y 4 grupos) con las mismas variables. Solo empezará si confirmas."
            ideas.append(
                AssistantIdea(
                    id=goal,
                    title=title,
                    description=description,
                    why=why,
                    expected=expected,
                    effect=effect,
                    goal=goal,
                    destination=advice.destination
                    if goal == "diagnose" and goal == advice.goal
                    else None,
                )
            )
            if len(ideas) == 3:
                break
        destinations = []
        if advice.destination:
            destinations.append(
                (advice.destination, advice.destination_label or "Continuar", advice.next_step)
            )
        if self.name == "ModelComparisonAgent" and not ideas:
            destinations = [
                (
                    "classification",
                    "Predecir categorías",
                    "Crea un modelo para distinguir categorías.",
                ),
                (
                    "regression",
                    "Predecir una cantidad",
                    "Crea un modelo para estimar valores numéricos.",
                ),
            ]
        if self.name in {"ClassificationAgent", "RegressionAgent"}:
            destination = "classification" if self.name == "ClassificationAgent" else "regression"
            destinations.append(
                (
                    destination,
                    "Revisar qué quiero predecir",
                    "Comprueba la variable elegida en el formulario.",
                )
            )
        destinations.extend(
            [
                (
                    "exploration",
                    "Seguir explorando",
                    "Continúa revisando tus variables en la pantalla principal.",
                ),
                (
                    "data",
                    "Revisar los datos activos",
                    "Consulta la vista previa y el tipo de cada columna.",
                ),
            ]
        )
        for destination, title, description in destinations:
            if len(ideas) >= 2:
                break
            if any(idea.destination == destination for idea in ideas):
                continue
            ideas.append(
                AssistantIdea(
                    id=f"navigate_{destination}",
                    title=title,
                    description=description,
                    why=advice.reason,
                    expected=description,
                    effect="Solo abrirá la pantalla correspondiente; no modificará datos.",
                    destination=destination,
                )
            )
        selection = metadata.get("active_selection", {}).get("analysis_selection") or {}
        relevant = [
            item
            for item in metadata["experiments"]
            if item["problem"] in self.problems
            and item["dataset_id"]
            in {self.context.dataset.dataset_id, self.context.dataset.content_hash}
        ]
        if self.name == "ModelComparisonAgent":
            welcome = (
                "Podemos contrastar las alternativas guardadas y revisar si son comparables."
                if len(relevant) >= 2
                else "Antes de comparar necesitamos dos alternativas obtenidas con los mismos datos. Estas opciones te ayudan a comenzar."
            )
        elif self.name != "EDAAgent" and relevant:
            welcome = "Ya tienes resultados en este análisis. Podemos entenderlos mejor o preparar otra prueba sin repetir el trabajo al abrir la ayuda."
        elif self.name == "ClusteringAgent":
            welcome = (
                f"Tus {metadata['rows']:,} registros superan el límite de 5.000 del agrupamiento interactivo. En el formulario selecciona una muestra (por ejemplo 2.000 filas) y ejecuta allí el análisis. La búsqueda guiada del agente no usa esa muestra automáticamente."
                if metadata["rows"] > 5000
                else f"Podemos buscar casos parecidos entre tus {metadata['rows']:,} registros. Te propongo estas formas de empezar."
            )
        elif self.name == "ClassificationAgent" and selection.get("target"):
            welcome = f"Podemos revisar las categorías de {selection['target']} y preparar una primera prueba para reconocerlas."
        elif self.name == "RegressionAgent" and selection.get("target"):
            welcome = f"Podemos preparar una primera prueba para predecir {selection['target']} y después revisar cuánto se equivoca."
        elif metadata["missing_cells"]:
            welcome = f"Encontré {metadata['missing_cells']:,} celdas sin información. Conviene revisarlas antes de interpretar los resultados."
        elif self.context.experiments:
            welcome = "Ya tienes resultados guardados. Podemos revisarlos o elegir otra forma de continuar sin repetir el trabajo."
        elif metadata["additional_duplicates"]:
            welcome = f"Encontré filas repetidas adicionales: {metadata['additional_duplicates']:,}. Podemos revisarlas y explorar el resto de tus datos."
        else:
            welcome = f"Revisé la información disponible de tus {metadata['rows']:,} registros. Estas ideas pueden ayudarte a comenzar."
        return AssistantOverview(welcome=welcome, ideas=ideas)

    def preparation_overview(self, metadata: dict[str, Any]) -> AssistantOverview:
        """Select reversible preparation proposals from existing model diagnostics."""
        ideas = []
        for goal, parameters, title, description, why in self.preparation_options(metadata):
            ideas.append(
                AssistantIdea(
                    id=goal,
                    goal=goal,
                    title=title,
                    description=description,
                    why=why,
                    expected="Verás una comparación antes y después antes de decidir si aplicas el cambio.",
                    effect="Solo prepara una vista previa. Aplicar el cambio requiere tu confirmación.",
                )
            )
        if len(ideas) < 3:
            has_numeric = any(
                any(kind in column["dtype"].lower() for kind in ("int", "float", "double"))
                for column in metadata["columns"]
            )
            ideas.append(
                AssistantIdea(
                    id="outliers" if has_numeric else "diagnose",
                    goal="outliers" if has_numeric else "diagnose",
                    title="Revisar valores poco habituales"
                    if has_numeric
                    else "Revisar la calidad general",
                    description="Comprueba si son errores o casos válidos antes de borrarlos.",
                    why="Un valor extremo puede ser información importante; detectarlo no justifica eliminarlo.",
                    expected="Una lista de casos para revisar.",
                    effect="No modificará datos.",
                )
            )
        if len(ideas) < 2:
            ideas.append(
                AssistantIdea(
                    id="navigate_clustering",
                    destination="clustering",
                    title="Preparar el siguiente análisis",
                    description="Elige qué variables usar para buscar grupos.",
                    why="No hay nulos ni duplicados detectados que resolver primero.",
                    expected="Abrirás Clustering para seleccionar variables y preprocesamiento.",
                    effect="Solo cambia de pantalla.",
                )
            )
        return AssistantOverview(
            welcome=(
                "Estamos preparando tus datos. Te propongo cambios concretos y reversibles; revisa su efecto antes de aplicarlos."
                if metadata["missing_cells"]
                or metadata["additional_duplicates"]
                or metadata["constant_columns"]
                else "No detecté nulos, duplicados ni columnas constantes. No hace falta borrar datos por rutina; revisa los valores poco habituales o continúa al modelado."
            ),
            ideas=ideas[:3],
        )

    def preparation_options(self, metadata: dict[str, Any]) -> list[tuple]:
        """Explicit policies; counts and percentages are supplied by the EDA model."""
        options = []
        missing = metadata.get("missing_by_column", {})
        roles = metadata.get("feature_roles", {})
        excluded = set(roles.get("identifiers", []) + roles.get("dates", []))
        types = {column["name"]: column["dtype"] for column in metadata["columns"]}
        for numeric, operation, label in [
            (True, "median", "mediana"),
            (False, "mode", "categoría más frecuente"),
        ]:
            columns = [
                column
                for column, values in missing.items()
                if column in types
                and column not in excluded
                and 0 < values["porcentaje"] < 50
                and any(kind in types[column].lower() for kind in ("int", "float", "double"))
                == numeric
            ][:30]
            if columns:
                names = ", ".join(columns[:4]) + ("…" if len(columns) > 4 else "")
                options.append(
                    (
                        f"prepare_{operation}",
                        {"operation": operation, "columns": columns},
                        f"Completar {names}",
                        f"Previsualiza los datos faltantes con la {label}.",
                        f"Estas columnas tienen menos del 50% de datos faltantes. La {label} es una primera alternativa {'resistente a valores extremos' if numeric else 'que conserva categorías existentes'}; puede reducir la variabilidad y debe revisarse según el significado de los datos.",
                    )
                )
        if metadata["additional_duplicates"] and metadata["column_count"] <= 30:
            options.append(
                (
                    "prepare_duplicates",
                    {
                        "operation": "duplicates",
                        "columns": [c["name"] for c in metadata["columns"]][:30],
                    },
                    f"Revisar {metadata['additional_duplicates']:,} filas repetidas",
                    "Previsualiza qué cambia al quitar registros idénticos.",
                    "Elimina solo si son copias por error. Las observaciones legítimamente repetidas deben conservarse.",
                )
            )
        columns = [column for column, values in missing.items() if values["porcentaje"] >= 50][:30]
        if columns and len(columns) < metadata["column_count"]:
            options.insert(
                0,
                (
                    "prepare_sparse",
                    {"operation": "drop_columns", "columns": columns},
                    "Revisar columnas con mucha información faltante",
                    f"Posible exclusión: {', '.join(columns[:4])}.",
                    "Falta al menos la mitad de los valores. Antes de eliminarlas, considera su importancia y si puedes recuperar información; no conviene rellenarlas automáticamente.",
                ),
            )
        constants = metadata.get("constant_names", [])
        if constants and len(constants) < metadata["column_count"]:
            options.append(
                (
                    "prepare_constants",
                    {"operation": "drop_columns", "columns": constants},
                    "Revisar columnas sin variación",
                    f"Posible exclusión: {', '.join(constants[:4])}.",
                    "No distinguen registros en estos datos. Conserva las que tengan valor documental; excluirlas del modelado puede simplificar el análisis.",
                )
            )
        return options[:3]

    def provider_status(self) -> dict[str, Any]:
        """Query the injected provider without importing transport logic into the View."""
        if not self.provider.available:
            return {
                "available": False,
                "message": "Motor offline disponible; modelo de lenguaje no configurado.",
            }
        if hasattr(self.provider, "probe"):
            return self.provider.probe()
        return {
            "available": False,
            "message": "El proveedor no implementa una comprobación local de disponibilidad.",
        }

    def guided_actions(self) -> dict[str, str]:
        """Expose finite commands; never claim arbitrary natural-language understanding."""
        actions = {policy.goal: policy.label for policy in self.decision_policies()}
        metadata = self.context.metadata()
        selection = metadata.get("active_selection", {}).get("analysis_selection") or {}
        relevant = [
            item
            for item in metadata["experiments"]
            if item["problem"] in self.problems
            and item["dataset_id"]
            in {self.context.dataset.dataset_id, self.context.dataset.content_hash}
        ]
        previous = self.context.conversation.get(self.name, [])
        preferences = {
            "EDAAgent": [(metadata["missing_cells"] > 0, "diagnose"), (bool(previous), "explore")],
            "ClusteringAgent": [
                (bool(relevant), "evaluate"),
                (metadata["rows"] >= 10, "configure"),
            ],
            "ClassificationAgent": [
                (bool(relevant), "evaluate"),
                (bool(selection.get("target")), "classes"),
            ],
            "RegressionAgent": [
                (bool(relevant), "evaluate"),
                (bool(selection.get("target")), "configure"),
            ],
            "ModelComparisonAgent": [(len(relevant) >= 2, "compare"), (bool(relevant), "evaluate")],
        }
        preferred = next(
            (goal for eligible, goal in preferences[self.name] if eligible), next(iter(actions))
        )
        return {preferred: actions[preferred], **actions}

    def advice(self) -> AssistantAdvice:
        """Explain one next step; all counts and metrics originate in the model layer."""
        metadata = self.context.metadata()
        selection = metadata.get("active_selection", {}).get("analysis_selection") or {}
        page = self.context.selections.get("workspace_page")
        relevant = [
            item
            for item in metadata["experiments"]
            if item["problem"] in self.problems
            and item["dataset_id"]
            in {self.context.dataset.dataset_id, self.context.dataset.content_hash}
        ]
        if self.name in {"ClassificationAgent", "RegressionAgent"} and selection.get("target"):
            relevant = [
                item for item in relevant if item["validation"].get("target") == selection["target"]
            ]
        if self.name == "EDAAgent":
            if metadata["missing_cells"]:
                return AssistantAdvice(
                    title="Primero revisa los datos que faltan",
                    reason=f"Celdas sin información detectadas: {metadata['missing_cells']:,}.",
                    next_step="Revisa qué columnas están afectadas. Después elige cómo completar esos valores y previsualiza el cambio antes de aplicarlo.",
                    goal="diagnose",
                    action_label="Revisar mis datos",
                    destination="preparation" if page != "preparation" else None,
                    destination_label="Ir a preparación",
                )
            if metadata["additional_duplicates"]:
                return AssistantAdvice(
                    title="Comprueba las filas repetidas",
                    reason=f"Filas adicionales idénticas detectadas: {metadata['additional_duplicates']:,}.",
                    next_step="Comprueba si son registros repetidos por error. Si decides quitarlos, previsualiza la eliminación antes de confirmar.",
                    goal="diagnose",
                    action_label="Revisar mis datos",
                    destination="preparation" if page != "preparation" else None,
                    destination_label="Revisar en preparación",
                )
            return AssistantAdvice(
                title="Empieza por conocer tus variables",
                reason="No se detectan celdas vacías ni filas idénticas adicionales. Esto no descarta otros problemas.",
                next_step="Elige una variable importante y observa su distribución para identificar valores poco habituales.",
                goal="explore",
                action_label="Mostrar una distribución",
                destination="exploration" if page != "exploration" else None,
                destination_label="Ir a exploración",
            )
        if self.name == "ModelComparisonAgent":
            if len(relevant) < 2:
                return AssistantAdvice(
                    title="Necesitas dos modelos para comparar",
                    reason=f"Resultados guardados para estos datos: {len(relevant)}.",
                    next_step="Entrena dos alternativas con los mismos datos, variables y separación de prueba. Luego compara la medida que te importe.",
                    goal="evaluate" if relevant else "compare",
                    action_label="Revisar experimento" if relevant else "Primero crea dos modelos",
                    runnable=bool(relevant),
                    destination="history" if relevant else "home",
                    destination_label="Ver experimentos" if relevant else "Elegir un análisis",
                )
            candidates = [
                item
                for item in relevant
                if (item["problem"], item["cohort"])
                in {
                    (other["problem"], other["cohort"])
                    for other in relevant
                    if other["id"] != item["id"]
                }
            ]
            compatible = [
                item
                for item in candidates
                if (item["problem"], item["cohort"])
                == (candidates[-1]["problem"], candidates[-1]["cohort"])
            ]
            if len(compatible) < 2:
                return AssistantAdvice(
                    title="Estos resultados no se pueden comparar directamente",
                    reason="Los experimentos disponibles usan distintos datos o separaciones de prueba.",
                    next_step="Entrena otra alternativa con la misma configuración de datos. Comparar solo sus cifras podría llevarte a una conclusión incorrecta.",
                    goal="evaluate",
                    action_label="Revisar el último resultado",
                    destination="history",
                    destination_label="Ver configuraciones",
                )
            return AssistantAdvice(
                title="Elige qué resultado es más importante para ti",
                reason="Hay al menos dos resultados obtenidos con los mismos datos y separación de prueba.",
                next_step="En el comparador selecciona una medida y si buscas un valor mayor o menor. Considera también el tiempo de entrenamiento.",
                goal="compare",
                action_label="Comparar resultados",
                destination="experiments",
                destination_label="Elegir medida de comparación",
            )
        if relevant:
            return AssistantAdvice(
                title="Revisa dónde falla antes de entrenar otra vez",
                reason="Ya tienes un resultado guardado para estos datos; puedes revisarlo sin repetir el entrenamiento.",
                next_step={
                    "ClusteringAgent": "Comprueba si los grupos son útiles y si quedaron muchos registros sin grupo.",
                    "ClassificationAgent": "Observa qué categorías confunde y presta atención a las que tienen pocos casos.",
                    "RegressionAgent": "Observa cuánto se alejan las predicciones de los valores reales y si los errores muestran un patrón.",
                }[self.name],
                goal="evaluate" if self.name == "ClusteringAgent" else "errors",
                action_label="Revisar mis grupos"
                if self.name == "ClusteringAgent"
                else "Ver errores del modelo",
            )
        if self.name == "ClusteringAgent":
            if metadata["rows"] > 5000:
                return AssistantAdvice(
                    title="Selecciona una muestra antes de agrupar",
                    reason=f"Hay {metadata['rows']:,} filas y el límite interactivo es 5.000.",
                    next_step="En el formulario de Clustering elige 2.000 filas y activa el uso de la muestra. Los resultados describirán solo esos registros.",
                    goal="configure",
                    action_label="Revisar variables primero",
                )
            return AssistantAdvice(
                title="Prepara las variables antes de buscar grupos",
                reason="El agrupamiento depende de las variables elegidas y de sus escalas.",
                next_step="Elige características que describan los casos. Evita códigos de cliente o documentos. Te propongo una configuración inicial que podrás revisar.",
                goal="configure",
                action_label="Preparar una configuración",
            )
        target = selection.get("target")
        if target in metadata.get("feature_roles", {}).get("identifiers", []):
            return AssistantAdvice(
                title="Revisa qué quieres predecir",
                reason=f"{target} parece un identificador por su nombre, no un resultado que predecir.",
                next_step="Selecciona una categoría o cantidad con significado para tu objetivo. No conviene entrenar para adivinar códigos de registro.",
                goal="diagnose",
                action_label="Revisar los datos",
            )
        if not target:
            return AssistantAdvice(
                title="Elige qué quieres predecir",
                reason="Todavía no hay una variable objetivo seleccionada en este módulo.",
                next_step="Selecciona esa columna en el formulario: una categoría para clasificación o una cantidad para regresión.",
                goal="diagnose",
                action_label="Revisar los datos primero",
            )
        if self.name == "ClassificationAgent":
            return AssistantAdvice(
                title=f"Revisa las categorías de {target}",
                reason="Antes de entrenar conviene saber si alguna categoría tiene muy pocos ejemplos.",
                next_step="Revisa sus cantidades. Después prepara un primer modelo; no te guíes únicamente por el porcentaje total de aciertos.",
                goal="classes",
                action_label="Ver cómo se reparten las categorías",
            )
        numeric_target = any(
            column["name"] == target
            and any(kind in column["dtype"].lower() for kind in ("int", "float", "double"))
            for column in metadata["columns"]
        )
        if not numeric_target:
            return AssistantAdvice(
                title="Selecciona una cantidad para predecir",
                reason=f"La columna {target} no se ha identificado como numérica.",
                next_step="Si quieres predecir una categoría, usa Clasificación. Para regresión, selecciona un valor numérico, como precio o duración.",
                goal="diagnose",
                action_label="Revisar mis datos",
            )
        return AssistantAdvice(
            title=f"Prepara un primer modelo para {target}",
            reason="Ya elegiste una cantidad numérica como objetivo.",
            next_step="Empieza con una configuración sencilla. Reserva datos para comprobar las predicciones y revisa los errores antes de probar alternativas.",
            goal="configure",
            action_label="Preparar el primer modelo",
        )
