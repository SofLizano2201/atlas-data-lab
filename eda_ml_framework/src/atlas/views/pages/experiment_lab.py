"""Spanish research workflow; scientific work is delegated to Model/controllers."""

import json
from zipfile import BadZipFile

import streamlit as st

from atlas.controllers.agent_controller import AgentController
from atlas.controllers.experiment_lab_controller import (
    METRICS,
    ExperimentalAlternative,
    ExperimentLabController,
    ScientificExperimentDefinition,
)
from atlas.core.dependencies import build_offline_agent
from atlas.services.experiment_service import ExperimentService
from atlas.services.report_service import ReportService
from atlas.views.components.layout import page_header
from atlas.views.components.tables import data_table
from atlas.views.components.ui import figure

TEMPLATES = {
    "Comparar algoritmos": ("algorithms", "algorithm"),
    "Evaluar una transformación": ("preprocessing", "scaler"),
    "Comparar variables": ("features", "features"),
    "Comprobar estabilidad del modelo": ("model_stability", "seed"),
    "Comprobar estabilidad de clusters": ("cluster_stability", "seed"),
    "Analizar incertidumbre": ("algorithms", "algorithm"),
    "Sensibilidad a un hiperparámetro": ("sensitivity", "parameters"),
    "Diseñar una comparación personalizada": ("custom", "custom"),
}
TASKS = {"Agrupamiento": "clustering", "Clasificación": "classification", "Regresión": "regression"}
STATUS = {
    "ready": "Diseño validado",
    "running": "En ejecución",
    "completed": "Completado",
    "partial": "Resultados parciales",
    "cancelled": "Cancelado entre evaluaciones",
}
ALGORITHM_LABELS = {
    "kmeans": "K-Means",
    "minibatch": "MiniBatch K-Means",
    "dbscan": "DBSCAN",
    "agglomerative": "Agrupamiento aglomerativo",
    "gaussian": "Mezcla gaussiana",
    "birch": "Birch",
    "optics": "OPTICS",
    "spectral": "Espectral",
    "logistic": "Regresión logística",
    "forest": "Bosque aleatorio",
    "tree": "Árbol de decisión",
    "linear_multiple": "Regresión lineal",
    "ridge": "Regresión Ridge",
    "lasso": "Regresión Lasso",
    "knn": "Vecinos más cercanos",
    "naive_bayes": "Bayes gaussiano",
    "extra_trees": "Árboles extra",
    "linear_simple": "Regresión lineal simple",
    "elasticnet": "Regresión Elastic Net",
    "svm_linear": "Máquina de soporte vectorial lineal",
    "svm_rbf": "Máquina de soporte vectorial radial",
    "svm_poly": "Máquina de soporte vectorial polinómica",
    "svr_linear": "Regresión de soporte vectorial lineal",
    "svr_rbf": "Regresión de soporte vectorial radial",
    "svr_poly": "Regresión de soporte vectorial polinómica",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Boosting de gradiente",
    "hist_gradient_boosting": "Boosting de gradiente con histogramas",
    "mlp": "Perceptrón multicapa",
}
PARAMETER_LABELS = {
    "alpha": "Intensidad de regularización",
    "C": "Regularización C",
    "n_estimators": "Cantidad de estimadores",
    "max_depth": "Profundidad máxima",
    "n_neighbors": "Cantidad de vecinos",
    "eps": "Radio de vecindad",
    "min_samples": "Mínimo de observaciones",
    "n_clusters": "Cantidad de grupos",
    "n_components": "Cantidad de componentes",
}
SCALERS = {
    "Sin escalar": "none",
    "Estandarización": "standard",
    "Escala robusta": "robust",
    "Rango mínimo–máximo": "minmax",
}


def experiment_lab_page():
    page_header("lab")
    design_tab, results_tab, reproduce_tab = st.tabs(
        ["1. Pregunta y diseño", "2. Ejecución y resultados", "3. Reproducibilidad"]
    )
    with design_tab:
        render_designer()
    run = st.session_state.get("scientific_investigations", {}).get(
        st.session_state.get("active_investigation")
    )
    with results_tab:
        if run:
            render_execution(run)
        else:
            st.info("Define y valida una investigación antes de ejecutar.")
    with reproduce_tab:
        render_history()


def render_designer():
    st.markdown("### ¿Qué quieres investigar?")
    template_label = st.selectbox("Pregunta de investigación", list(TEMPLATES), key="lab_template")
    template, factor = TEMPLATES[template_label]
    task_options = (
        ["Agrupamiento"]
        if template == "cluster_stability"
        else (
            ["Clasificación", "Regresión"]
            if template_label == "Analizar incertidumbre"
            else list(TASKS)
        )
    )
    task = TASKS[st.selectbox("Tipo de análisis", task_options, key=f"lab_task_{template_label}")]
    source = st.session_state.eda
    frame = source.df
    available = list(frame.columns)
    target = None
    if task != "clustering":
        target = st.selectbox(
            "Variable que quieres predecir",
            available,
            index=len(available) - 1,
            key=f"lab_target_{task}",
        )
    choices = [c for c in available if c != target]
    features = st.multiselect(
        "Variables utilizadas",
        choices,
        default=choices[: min(4, len(choices))],
        key=f"lab_features_{task}",
    )
    question = st.text_input(
        "Pregunta",
        value="¿Qué alternativa mejora el desempeño bajo las mismas condiciones?",
        max_chars=600,
    )
    hypothesis = st.text_area(
        "Hipótesis",
        value="La alternativa B mejora la métrica seleccionada respecto a la alternativa A bajo este diseño.",
        max_chars=1000,
    )
    hypothesis_type = st.selectbox(
        "Tipo de hipótesis",
        ["exploratory", "directional", "comparative", "stability"],
        format_func={
            "exploratory": "Exploratoria",
            "directional": "Direccional",
            "comparative": "Comparativa",
            "stability": "Estabilidad",
        }.get,
    )
    algorithms = [
        a
        for a in ExperimentLabController.algorithms(task)
        if a not in {"hierarchical", "kmedoids", "xgboost", "ridge_cv", "lasso_cv", "elasticnet_cv"}
    ]
    cols = st.columns(2)
    with cols[0].container(border=True, key="lab_alternative_a"):
        st.markdown("#### Alternativa A · Referencia")
        algorithm_a = st.selectbox(
            "Algoritmo de referencia",
            algorithms,
            format_func=lambda a: ALGORITHM_LABELS.get(a, a),
            key=f"lab_algorithm_a_{task}",
        )
        scaler_a = SCALERS[
            st.selectbox(
                "Preparación de referencia", list(SCALERS), index=0 if factor == "scaler" else 1
            )
        ]
    with cols[1].container(border=True, key="lab_alternative_b"):
        st.markdown("#### Alternativa B · Comparación")
        if factor in {"algorithm", "custom"}:
            algorithm_b = st.selectbox(
                "Algoritmo alternativo",
                algorithms,
                index=min(1, len(algorithms) - 1),
                format_func=lambda a: ALGORITHM_LABELS.get(a, a),
                key=f"lab_algorithm_b_{task}",
            )
        else:
            algorithm_b = algorithm_a
            st.caption("Se conserva el algoritmo de referencia.")
        if factor in {"scaler", "custom"}:
            scaler_b = SCALERS[st.selectbox("Preparación alternativa", list(SCALERS), index=1)]
        else:
            scaler_b = scaler_a
        features_b = (
            st.multiselect(
                "Variables alternativas",
                features,
                default=features[:-1] or features,
                key=f"lab_features_b_{task}",
            )
            if factor in {"features", "custom"}
            else features
        )
        st.caption("En estabilidad se repite una sola alternativa con semillas coordinadas.")
    params_a, params_b = {}, {}
    with st.expander("Ajustes avanzados y condiciones controladas"):
        null_hypothesis = st.text_input(
            "Hipótesis nula", "Sin mejora en la cantidad de desempeño seleccionada."
        )
        alternative_hypothesis = st.text_input(
            "Hipótesis alternativa", "Mejora en la dirección especificada bajo este diseño."
        )
        if task == "clustering" and algorithm_a in {
            "kmeans",
            "minibatch",
            "agglomerative",
            "birch",
            "spectral",
            "gaussian",
        }:
            k = st.number_input("Cantidad de grupos de referencia", 2, 30, 3)
            params_a["n_components" if algorithm_a == "gaussian" else "n_clusters"] = int(k)
        if factor == "parameters":
            from atlas.controllers.experiment_lab_controller import parameter_options

            options = parameter_options(task, algorithm_a)
            if options:
                parameter = st.selectbox(
                    "Parámetro que cambia",
                    options,
                    format_func=lambda p: PARAMETER_LABELS.get(p, p),
                )
                pa = st.number_input(
                    "Valor de referencia",
                    min_value=0.001,
                    max_value=1000.0,
                    value=3.0 if parameter in {"n_clusters", "n_components"} else 1.0,
                )
                pb = st.number_input(
                    "Valor alternativo", min_value=0.001, max_value=1000.0, value=10.0
                )
                integer = parameter in {
                    "n_clusters",
                    "n_components",
                    "n_estimators",
                    "max_depth",
                    "n_neighbors",
                    "min_samples",
                }
                params_a[parameter] = int(pa) if integer else float(pa)
                params_b = dict(params_a) | {parameter: int(pb) if integer else float(pb)}
            else:
                st.info(
                    "Este algoritmo no ofrece un parámetro de sensibilidad en el formulario. Selecciona otro algoritmo."
                )
        else:
            params_b = dict(params_a) if algorithm_b == algorithm_a else {}
        extra_a = st.text_area(
            "Parámetros adicionales de A (JSON, opcional)", "{}", max_chars=3000, key="lab_extra_a"
        )
        extra_b = st.text_area(
            "Parámetros adicionales de B (JSON, opcional)",
            "{}",
            max_chars=3000,
            key="lab_extra_b",
            disabled=factor not in {"algorithm", "custom"},
        )
        st.caption(
            "Opcional para usuarios avanzados: se aceptan únicamente parámetros de los estimadores registrados; nunca código Python."
        )
        metrics = list(METRICS[task])
        metric = st.selectbox(
            "Métrica principal",
            metrics,
            index=metrics.index("MAE") if task == "regression" else 0,
            key=f"lab_metric_{task}",
        )
        count = st.number_input(
            "Repeticiones coordinadas",
            1,
            5,
            1 if template_label == "Analizar incertidumbre" else 3,
            key=f"lab_repetitions_{template_label}",
        )
        seed = st.number_input("Primera semilla", 0, 99995, 42)
        cv = st.number_input(
            "Folds dentro del entrenamiento", 2, 5, 3, disabled=task == "clustering"
        )
        test_size = st.slider(
            "Proporción para prueba", 0.05, 0.5, 0.2, disabled=task == "clustering"
        )
        protocol = st.selectbox(
            "Protocolo",
            ["automatic", "group", "time", "nested"],
            format_func={
                "automatic": "Inicializaciones sobre las mismas filas"
                if task == "clustering"
                else "Prueba reservada y CV dentro del entrenamiento",
                "group": "Por grupos · requiere soporte adicional",
                "time": "Temporal · requiere soporte adicional",
                "nested": "CV anidada · requiere soporte adicional",
            }.get,
        )
        structured = st.checkbox(
            "Los datos contienen entidades repetidas o requieren respetar el tiempo"
        )
        practical = st.number_input("Mejora mínima de interés práctico", min_value=0.0, value=0.0)
        bootstrap = st.checkbox(
            "Calcular intervalo bootstrap pareado", value=template_label == "Analizar incertidumbre"
        )
        independent = st.checkbox(
            "Confirmo que las observaciones de prueba pueden tratarse como independientes",
            disabled=not bootstrap,
        )
        st.caption(
            "Bootstrap: solo dos alternativas, una repetición y Accuracy o MAE. El intervalo es condicional a los modelos entrenados; no mide toda la incertidumbre de entrenamiento."
        )
        reason = st.text_input(
            "Justificación de cambios simultáneos", disabled=factor != "custom", max_chars=1000
        )
    st.info(
        "Se conserva el dataset, el objetivo, las semillas y el protocolo. Cada alternativa cambia únicamente el factor declarado, salvo diseños personalizados documentados. No se infiere causalidad."
    )
    if template == "cluster_stability" and task != "clustering":
        st.warning("Selecciona Agrupamiento para estudiar estabilidad de clusters.")
        return
    if st.button("Validar y guardar diseño", type="primary", key="lab_prepare"):
        try:
            additional_a = json.loads(extra_a)
            additional_b = json.loads(extra_b) if factor in {"algorithm", "custom"} else {}
            if not isinstance(additional_a, dict) or not isinstance(additional_b, dict):
                raise TypeError("Los parámetros deben ser objetos JSON.")
            params_a.update(additional_a)
            if factor in {"algorithm", "custom"}:
                params_b.update(additional_b)
            elif factor == "parameters":
                params_b = dict(params_a) | params_b
            else:
                params_b = dict(params_a)
            a = ExperimentalAlternative(
                name="A · Referencia", algorithm=algorithm_a, scaler=scaler_a, parameters=params_a
            )
            alternatives = [a]
            if factor != "seed":
                alternatives.append(
                    ExperimentalAlternative(
                        name="B · Comparación",
                        algorithm=algorithm_b,
                        scaler=scaler_b,
                        predictors=features_b,
                        parameters=params_b,
                    )
                )
            definition = ScientificExperimentDefinition(
                name=template_label,
                question=question,
                hypothesis=hypothesis,
                hypothesis_type=hypothesis_type,
                template=template,
                task=task,
                dataset=source.dataset_version,
                predictors=features,
                target=target,
                null_hypothesis=null_hypothesis,
                alternative_hypothesis=alternative_hypothesis,
                alternatives=alternatives,
                factor=factor,
                metric=metric,
                direction=METRICS[task][metric],
                seeds=list(range(int(seed), int(seed) + int(count))),
                cv=int(cv),
                test_size=float(test_size),
                validation=(
                    "same_rows_initializations" if task == "clustering" else "random_holdout_cv"
                )
                if protocol == "automatic"
                else protocol,
                metadata={"group_column": "declared"} if structured else {},
                custom_justification=reason,
                paired_bootstrap=bootstrap,
                independent_observations_confirmed=independent,
                practical_difference=practical,
            )
            findings = ExperimentLabController.validate(definition, frame)
            for finding in findings:
                (st.error if finding.blocking else st.warning)(
                    finding.title + ". " + finding.explanation
                )
            ExperimentLabController.enforce(findings)
            run = ExperimentLabController.prepare(st.session_state, definition)
            st.success(
                "Diseño guardado. Abre Ejecución y resultados para revisar el presupuesto y confirmar."
            )
            st.session_state["lab_design_feedback"] = run["id"]
        except (ValueError, TypeError):
            st.error(
                "El diseño no se pudo guardar. Revisa los requisitos indicados y las selecciones."
            )
    with st.expander("Orientación de los especialistas"):
        context = AgentController.context_for(st.session_state)
        specialist = {
            "classification": "ClassificationAgent",
            "regression": "RegressionAgent",
            "clustering": "ClusteringAgent",
        }[task]
        for name in ["EDAAgent", specialist, "ModelComparisonAgent"]:
            st.write(build_offline_agent(name, context)[0].recommend_experiment()["explanation"])


def render_execution(run):
    definition = run["definition"]
    st.markdown("### " + definition.question)
    st.write(definition.hypothesis)
    st.caption(
        f"Datos guardados: {len(run['dataset'])} filas · versión {definition.dataset.version} · huella {definition.dataset.content_hash[:12]}"
    )
    if st.session_state.eda.dataset_version.content_hash != definition.dataset.content_hash:
        st.warning(
            "Esta investigación utiliza su versión guardada, que es diferente del dataset activo."
        )
    st.caption(STATUS[run["status"]])
    algorithms = ", ".join(
        dict.fromkeys(
            ALGORITHM_LABELS.get(a.algorithm, a.algorithm) for a in definition.alternatives
        )
    )
    st.info(
        f"{len(definition.alternatives)} alternativas · {definition.evaluations} evaluaciones · hasta {definition.fits} ajustes, incluidos los folds. Algoritmos: {algorithms}. Carga {'moderada' if definition.fits > 12 else 'acotada'}. Límites: 12 evaluaciones, 72 ajustes, 5 repeticiones, 4 alternativas y 5000 filas."
    )
    st.caption(
        "Las semillas coordinan particiones entre alternativas. Cambiar la semilla en aprendizaje supervisado también cambia la partición: la variabilidad combina ambos efectos."
    )
    completed = len(run["evaluations"])
    st.progress(
        completed / definition.evaluations,
        text=f"{len(run['artifacts'])} evaluaciones válidas · {completed} intentadas de {definition.evaluations}",
    )
    if run.get("current_operation"):
        st.caption(run["current_operation"])
    if run["status"] in {"ready", "partial", "cancelled"}:
        confirm = st.checkbox(
            "Confirmo la ejecución de los entrenamientos y el presupuesto mostrado",
            key=f"lab_consent_{run['id']}",
        )
        if st.button(
            "Ejecutar investigación"
            if run["status"] == "ready"
            else "Reanudar evaluaciones pendientes",
            disabled=not confirm,
            key=f"lab_start_{run['id']}",
            type="primary",
        ):
            try:
                ExperimentLabController.authorize(
                    st.session_state, run["id"], confirm=confirm, retry=True
                )
            except ValueError as exc:
                st.error(str(exc))
    if run["status"] == "running" and st.button(
        "Cancelar después de la evaluación actual", key=f"lab_cancel_{run['id']}"
    ):
        ExperimentLabController.cancel(st.session_state, run["id"])
    if run["evaluations"]:
        st.markdown("### Resultados observados")
        summary = ExperimentLabController.summary(run)
        data_table(
            summary.rename(
                columns={
                    "alternative": "Alternativa",
                    "count": "Evaluaciones",
                    "mean": "Media",
                    "std": "Desviación estándar",
                    "median": "Mediana",
                    "q25": "Percentil 25",
                    "q75": "Percentil 75",
                }
            ),
            target=st,
        )
        st.caption(
            "La desviación estándar queda sin definir con una sola evaluación. Los percentiles describen estas repeticiones; no son intervalos de confianza."
        )
        charts = ExperimentLabController.charts(run)
        for offset in range(0, len(charts), 2):
            for column, (name, chart) in zip(
                st.columns(2), list(charts.items())[offset : offset + 2]
            ):
                with column:
                    figure(chart, f"lab_{run['id']}_{name}")
        if run.get("uncertainty"):
            st.caption(run["uncertainty"]["scope"])
        st.markdown("### Interpretación científica")
        st.write(ExperimentLabController.interpretation(run))
        for record in run["evaluations"].values():
            if record["status"] == "failed":
                st.warning(f"No se completó {record['alternative']} con semilla {record['seed']}.")
                with st.expander("Revisar el requisito de esta evaluación"):
                    st.write(record["error"])


def render_history():
    investigations = st.session_state.get("scientific_investigations", {})
    if investigations:
        selected = st.selectbox(
            "Investigación guardada",
            list(investigations),
            format_func=lambda i: (
                f"{investigations[i]['definition'].name} · {STATUS[investigations[i]['status']]} · {i[:8]}"
            ),
        )
        run = investigations[selected]
        if st.button("Abrir investigación"):
            st.session_state.active_investigation = selected
            st.rerun()
        st.caption(
            f"Identificador: {run['id']} · Versión del dataset: {run['definition'].dataset.version}"
        )
        with st.expander("Consultar condiciones guardadas"):
            st.write(run["definition"].question)
            st.write(run["definition"].hypothesis)
            for alternative in run["definition"].alternatives:
                st.write(
                    f"{alternative.name}: {ALGORITHM_LABELS.get(alternative.algorithm, alternative.algorithm)} · escala {alternative.scaler}"
                )
            st.caption("Semillas: " + str(run["definition"].seeds))
        current = st.radio(
            "Datos para una nueva ejecución",
            ["Versión original guardada", "Dataset actual"],
            horizontal=True,
        )
        if st.button("Preparar reproducción con un identificador nuevo"):
            try:
                ExperimentLabController.reproduce(
                    st.session_state, selected, current_dataset=current == "Dataset actual"
                )
                st.success("Nueva propuesta creada. Requiere confirmación antes de entrenar.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if st.button("Preparar archivos de esta investigación"):
            st.session_state["lab_exports"] = {
                "id": selected,
                "zip": ExperimentService.export_investigation(run),
                "html": ReportService.remember_lab_report(st.session_state, run, "html").data,
                "pdf": ReportService.remember_lab_report(st.session_state, run, "pdf").data,
                "csv": ReportService.lab_results(run).to_csv(index=False).encode("utf-8-sig"),
                "json": json.dumps(
                    ExperimentLabController.manifest(run),
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                ).encode("utf-8"),
            }
        exports = st.session_state.get("lab_exports", {})
        if exports.get("id") == selected:
            for extension, label, mime in [
                ("zip", "Paquete reproducible con dataset", "application/zip"),
                ("html", "Informe HTML", "text/html"),
                ("pdf", "Informe PDF", "application/pdf"),
                ("csv", "Resultados CSV", "text/csv"),
                ("json", "Manifiesto de configuración", "application/json"),
            ]:
                st.download_button(
                    label,
                    exports[extension],
                    f"atlas_lab_{selected[:8]}.{extension}",
                    mime,
                    key=f"lab_download_{extension}",
                )
        with st.expander("Comparar con investigaciones anteriores"):
            comparisons = st.multiselect(
                "Investigaciones",
                list(investigations),
                format_func=lambda i: investigations[i]["definition"].name + " · " + i[:8],
            )
            ranking = st.checkbox("Ordenar por el desempeño medio", key="lab_history_rank")
            if len(comparisons) >= 2:
                try:
                    table = ExperimentLabController.compare_investigations(
                        [investigations[i] for i in comparisons], rank=ranking
                    )
                    if not table.attrs["comparable"]:
                        st.warning(
                            "Consulta descriptiva: los diseños o versiones no permiten una clasificación directa."
                        )
                    data_table(
                        table.rename(
                            columns={
                                "investigation": "Investigación",
                                "name": "Nombre",
                                "alternative": "Alternativa",
                                "count": "Evaluaciones",
                                "mean": "Media",
                                "std": "Desviación estándar",
                                "median": "Mediana",
                                "q25": "Percentil 25",
                                "q75": "Percentil 75",
                            }
                        ),
                        target=st,
                    )
                except ValueError as exc:
                    st.warning(str(exc))
    else:
        st.info(
            "Los diseños y resultados se conservan en esta sesión. Exporta el paquete para guardarlos fuera de ATLAS."
        )
    with st.expander("Importar un paquete reproducible de ATLAS"):
        uploaded = st.file_uploader("Paquete ZIP", type=["zip"], key="lab_import")
        if uploaded is not None and st.button("Importar sin ejecutar"):
            try:
                run = ExperimentService.import_investigation(uploaded.getvalue())
                ExperimentService.remember_investigation(st.session_state, run)
                st.success(
                    "Configuración y dataset verificados. La reproducción requiere confirmación."
                )
            except (ValueError, KeyError, OSError, BadZipFile):
                st.error("El paquete no contiene una configuración y un dataset compatibles.")
