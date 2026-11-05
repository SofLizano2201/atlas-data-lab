"""Spanish Streamlit views: rendering and input collection only."""

import json
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from atlas.controllers.charts import PALETTE, diagnostic_bars, style
from atlas.controllers.classification_controller import ClassificationController
from atlas.controllers.clustering_controller import ClusteringController
from atlas.controllers.comparison_controller import ComparisonController, compare, record_experiment
from atlas.controllers.dataset_controller import DatasetController
from atlas.controllers.regression_controller import RegressionController
from atlas.controllers.report_controller import export_frame, html_report
from atlas.controllers.workflow_controller import WorkflowController
from atlas.core.exceptions import FrameworkError
from atlas.views.components.layout import algorithm_cards, metric_cards, page_header
from atlas.views.components.ui import figure, format_number


def remember(result: object, engine: object) -> None:
    record_experiment(st.session_state, result, engine)


def parameters(algorithm: str, problem: str) -> dict:
    """Provide parameters using validated inputs."""
    params = {}
    if algorithm in {"tree", "forest", "extra_trees", "gradient_boosting"}:
        params["max_depth"] = st.slider("Profundidad máxima", 1, 20, 4)
    if algorithm in {"forest", "extra_trees", "gradient_boosting", "adaboost"}:
        params["n_estimators"] = st.slider("Estimadores", 10, 300, 100)
    if algorithm == "knn":
        params["n_neighbors"] = st.slider("Vecinos", 1, 30, 5)
    if algorithm.startswith(("svm_", "svr_")):
        params["C"] = st.number_input("Regularización C", 0.01, 1000.0, 1.0)
    if algorithm in {"lasso", "ridge", "elasticnet"}:
        params["alpha"] = st.number_input("Regularización alpha", 0.0001, 100.0, 1.0)
    if problem == "clasificacion" and algorithm in {
        "logistic",
        "tree",
        "forest",
        "extra_trees",
        "svm_linear",
        "svm_rbf",
        "svm_poly",
    }:
        balanced = st.checkbox("Ponderar clases desbalanceadas", value=True)
        params["class_weight"] = "balanced" if balanced else None
    with st.expander("Parámetros avanzados"):
        st.caption(
            'Objeto JSON, por ejemplo {"min_samples_leaf": 3}. Sobrescribe los controles anteriores.'
        )
        text = st.text_area("Parámetros del estimador", "{}")
        extra = json.loads(text)
        if not isinstance(extra, dict):
            raise FrameworkError("Los parámetros deben ser un objeto JSON.")
        params.update(extra)
    return params


def supervised_page(problem: str) -> None:
    """Provide supervised page using validated inputs."""
    page_header("classification" if problem == "clasificacion" else "regression")
    st.session_state.problem = problem
    st.info(
        "Predice una categoría, como el tipo de cliente o si alguien compra."
        if problem == "clasificacion"
        else "Predice una cantidad, como el precio de una vivienda o las ventas."
    )
    with st.expander("Cambiar los datos que se usarán (opcional)"):
        source = st.radio(
            "Datos de modelado", ["Original (recomendado)", "Trabajo transformado"], horizontal=True
        )
        st.caption(
            "Usamos el original por defecto y aprendemos las transformaciones solo con los datos de entrenamiento."
        )
    df = st.session_state.eda.original if source.startswith("Original") else st.session_state.eda.df
    if source.startswith("Trabajo"):
        st.warning(
            "Transformaciones exploratorias ajustadas sobre todas las filas pueden sesgar la evaluación. Usa el original para estimar generalización."
        )
    cls = ClassificationController if problem == "clasificacion" else RegressionController
    with st.container(border=True, key="workflow_supervised_1"):
        st.subheader("1. Elige qué quieres predecir")
        objective_col, predictors_col = st.columns([1, 2])
        target = objective_col.selectbox(
            "¿Qué columna quieres predecir?",
            list(df.columns),
            index=len(df.columns) - 1,
            key=f"analysis_target_{problem}",
        )
        predictor_key = f"analysis_predictors_{problem}"
        if predictor_key in st.session_state:
            st.session_state[predictor_key] = [
                c for c in st.session_state[predictor_key] if c in df and c != target
            ]
        features = predictors_col.multiselect(
            "¿Qué información usará para aprender?",
            [c for c in df if c != target],
            default=[c for c in DatasetController(df).feature_roles()["suggested"] if c != target][
                :12
            ],
            key=predictor_key,
            help="Selecciona características conocidas antes de la predicción. Evita códigos de registro y columnas que revelen la respuesta.",
        )
        st.caption(
            "Por defecto reservamos el 20% de los registros para comprobar qué tan bien predice en datos que no vio al aprender."
        )
        roles = DatasetController(df).feature_roles()
        flagged = [
            column
            for column in features
            if column in roles["identifiers"] + roles["dates"] + roles["constants"]
        ]
        if flagged:
            st.warning(
                "Revisa estas columnas antes de aprender: "
                + ", ".join(flagged)
                + ". Los códigos, fechas sin transformar y valores constantes pueden producir un modelo poco útil."
            )
        if target in roles["identifiers"]:
            st.warning(
                "La columna que quieres predecir parece un identificador. Elige una cantidad o categoría que tenga sentido para tu objetivo."
            )
        with st.expander("Ajustar la evaluación (avanzado)"):
            test_col, folds_col, seed_col = st.columns(3)
            size = test_col.slider(
                "Parte reservada para comprobar resultados",
                0.05,
                0.5,
                0.2,
                0.05,
                key=f"analysis_size_{problem}",
                help="Estos datos no se usan para entrenar ni elegir parámetros.",
            )
            folds = folds_col.number_input(
                "Particiones de validación cruzada",
                2,
                10,
                5,
                key=f"analysis_cv_{problem}",
                help="Comprueba el modelo varias veces dentro del entrenamiento. En datasets pequeños puede ser necesario reducir este número.",
            )
            seed = seed_col.number_input(
                "Semilla para repetir el experimento", 0, 99999, 42, key=f"analysis_seed_{problem}"
            )
        st.session_state.analysis_selection = {
            "problem": problem,
            "target": target,
            "predictors": features,
            "test_size": size,
            "cv": folds,
            "random_state": seed,
        }
        st.session_state.selected_variables = features
        if problem == "clasificacion":
            distribution = cls(df).frequencies(target)
            if (distribution["absoluta"] < 2).any():
                st.warning(
                    f"La columna «{target}» contiene categorías que aparecen una sola vez. Una categoría es cada respuesta posible, como «sí» o «no». Necesitamos varios ejemplos de cada respuesta para aprender y comprobar el modelo. Cambiar de algoritmo no lo soluciona: si quieres predecir un precio o cantidad, ve a Regresión; si es un código, elige otra columna. Para categorías reales, consigue más ejemplos sin duplicar filas artificialmente."
                )
            with st.expander("Distribución de clases y definición de métricas"):
                st.dataframe(distribution, use_container_width=True)
                st.caption(
                    "Precision, Recall y F1 usan promedio macro; se añaden métricas binarias cuando corresponde."
                )
        with st.expander("Cómo se preparan los datos (avanzado)"):
            a, b, c = st.columns(3)
            scaler = a.selectbox(
                "Escalamiento",
                ["standard", "minmax", "robust", "normalizer", "none"],
                key=f"analysis_scaler_{problem}",
            )
            encoding = b.selectbox("Categorías", ["onehot", "ordinal"])
            imputation = c.selectbox(
                "Imputación numérica", ["median", "mean", "most_frequent", "constant"]
            )
        st.session_state.preprocessing = {
            "scaler": scaler,
            "encoding": encoding,
            "imputation": imputation,
        }
    with st.container(border=True, key="workflow_supervised_2"):
        st.subheader("2. Elige cómo quieres comenzar")
        st.caption(
            "Puedes empezar con el modelo sencillo seleccionado. Las alternativas permiten comparar después; ninguna es mejor para todos los datos."
        )
        default = "logistic" if problem == "clasificacion" else "linear_multiple"
        algorithm_key = f"atlas_algorithms_{problem}"
        st.session_state.setdefault(algorithm_key, [default])
        with st.expander("Elegir otros modelos y ajustar sus opciones"):
            algorithm_cards(
                [
                    (
                        default,
                        "Regresión logística" if problem == "clasificacion" else "Lineal múltiple",
                        "Modelo lineal interpretable.",
                    ),
                    (
                        "knn" if problem == "clasificacion" else "ridge",
                        "KNN" if problem == "clasificacion" else "Ridge",
                        "Vecinos cercanos." if problem == "clasificacion" else "Regularización L2.",
                    ),
                    ("forest", "Random Forest", "Ensamble de árboles de decisión."),
                    (
                        "svm_rbf" if problem == "clasificacion" else "svr_rbf",
                        "SVM" if problem == "clasificacion" else "SVR",
                        "Máquinas de soporte con kernel RBF.",
                    ),
                    ("gradient_boosting", "Gradient Boosting", "Ensamble secuencial de árboles."),
                ],
                algorithm_key,
            )
            algorithms = st.multiselect(
                "Algoritmos a entrenar", cls.algorithms(), key=algorithm_key
            )
            with st.expander("Parámetros del modelo"):
                params = parameters(algorithms[0], problem) if len(algorithms) == 1 else {}
        search = None
        kind = "grid"
        with st.expander("Búsqueda de hiperparámetros"):
            enabled = st.checkbox("Activar búsqueda dentro de entrenamiento")
            kind = st.selectbox("Estrategia", ["grid", "random"])
            key = st.selectbox(
                "Parámetro", ["alpha", "C", "max_depth", "n_estimators", "n_neighbors"]
            )
            values = st.text_input("Valores separados por coma", "1, 3, 5")
            if enabled:
                candidates = [float(v.strip()) for v in values.split(",")]
                search = {
                    key: [int(v) for v in candidates]
                    if key in {"max_depth", "n_estimators", "n_neighbors"}
                    else candidates
                }
                st.caption("Las opciones deben ser compatibles con cada algoritmo seleccionado.")
    with st.container(border=True, key="workflow_supervised_3"):
        st.subheader("3. Aprende con mis datos y comprueba las predicciones")
        st.write(
            f"Se entrenarán {len(algorithms)} modelos para predecir **{target}** con {len(features)} variables. Los resultados quedarán guardados en Experimentos."
        )
        if st.button(
            "Crear modelo y revisar resultados",
            key=f"train_{problem}",
            type="primary",
            disabled=not algorithms or not features,
        ):
            progress = st.progress(0, text="Preparando experimentos…")
            completed = 0
            for i, algorithm in enumerate(algorithms):
                engine = cls(df, random_state=int(seed))
                WorkflowController.training_started(st.session_state, problem, algorithm)
                try:
                    with st.spinner(f"Entrenando {algorithm}…"):
                        result = engine.train(
                            algorithm,
                            target,
                            features,
                            test_size=size,
                            cv=int(folds),
                            scaler=scaler,
                            encoding=encoding,
                            imputation=imputation,
                            parameters=params,
                            search=search,
                            search_type=kind,
                        )
                        remember(result, engine)
                        completed += 1
                except (ValueError, TypeError, ImportError, RuntimeError) as exc:
                    WorkflowController.analysis_failed(st.session_state, problem, exc)
                    st.error(f"{algorithm}: {exc}")
                progress.progress(
                    (i + 1) / len(algorithms), text=f"{i + 1}/{len(algorithms)} completados"
                )
            if completed:
                st.success(
                    f"Se crearon y guardaron {completed} de {len(algorithms)} modelos. Revisa sus resultados abajo."
                )
        results = [e for e in st.session_state.experiments if e.problem == problem]
        if not results:
            st.info(
                "Todavía no hay resultados. Elige qué predecir y pulsa «Crear modelo y revisar resultados»."
            )
            return
    with st.container(border=True, key="workflow_supervised_4"):
        st.subheader("4. Revisa tus resultados")
        model_names = {
            "ridge": "Regresión con regularización (Ridge)",
            "linear_multiple": "Regresión lineal",
            "linear_simple": "Regresión lineal simple",
            "logistic": "Clasificación logística",
            "forest": "Bosque de árboles (Random Forest)",
            "tree": "Árbol de decisión",
            "knn": "Vecinos cercanos",
            "lasso": "Regresión Lasso",
            "gradient_boosting": "Árboles con aprendizaje secuencial",
            "svr_rbf": "Regresión SVR",
            "svm_rbf": "Clasificación SVM",
        }
        result_order = {item.id: index for index, item in enumerate(results, 1)}
        result = st.selectbox(
            "¿Qué modelo guardado quieres revisar?",
            results,
            index=len(results) - 1,
            format_func=lambda e: (
                f"Prueba {result_order[e.id]} · {model_names.get(e.name, e.name)} · {e.config.get('target', '')}"
            ),
        )
        st.caption(
            "Este resultado conserva los datos y opciones con los que se entrenó. Cambiar el formulario no vuelve a entrenarlo."
        )
        engine = st.session_state.models[result.id]
        st.html(
            '<div class="atlas-result-context"><div><span>QUÉ SE PREDICE</span>'
            f"<strong>{escape(str(result.config.get('target', 'Sin especificar')))}</strong></div>"
            f"<div><span>MODELO GUARDADO</span><strong>{escape(model_names.get(result.name, result.name))}</strong></div>"
            f"<div><span>TIEMPO DE ENTRENAMIENTO</span><strong>{format_number(result.seconds)} s</strong></div></div>"
        )
        featured = (
            ["Accuracy", "Precision", "Recall", "F1"]
            if problem == "clasificacion"
            else ["MAE", "RMSE", "R2", "MAPE"]
        )
        metric_explanations = {
            "Accuracy": ("Aciertos globales", "Proporción correcta: 0,90 equivale al 90%."),
            "Precision": (
                "Fiabilidad al señalar una categoría",
                "Mayor valor indica menos falsas asignaciones; promedio entre categorías.",
            ),
            "Recall": (
                "Casos que logra reconocer",
                "Mayor valor indica menos casos que pasan desapercibidos; promedio entre categorías.",
            ),
            "F1": (
                "Equilibrio entre aciertos y detección",
                "De 0 a 1; mayor es mejor según este criterio.",
            ),
            "MAE": (
                "Error medio",
                "En las unidades de lo que predices; menor es mejor.",
            ),
            "RMSE": (
                "Impacto de fallos grandes",
                "En las unidades del objetivo; menor es mejor.",
            ),
            "R2": (
                "Ajuste del modelo (R²)",
                "1 es ajuste perfecto; puede ser negativo si funciona mal.",
            ),
            "MAPE": (
                "Error porcentual medio",
                "Respecto al valor real; poco fiable si los valores están cerca de cero.",
            ),
        }
        cards = []
        for name in featured:
            value = result.metrics.get(name)
            if value is None or not np.isfinite(value):
                continue
            formatting = (
                ".1%"
                if name in {"Accuracy", "Precision", "Recall", "MAPE"}
                else ".3f"
                if name in {"F1", "R2"}
                else ",.2f"
            )
            cards.append(
                f'<article class="atlas-result-card"><span>{escape(metric_explanations[name][0])}</span><strong>{format_number(value, formatting)}</strong><p>{escape(metric_explanations[name][1])}</p></article>'
            )
        if cards:
            st.html('<div class="atlas-result-grid">' + "".join(cards) + "</div>")
        if problem == "regresion" and np.isfinite(result.metrics.get("MAE", np.nan)):
            st.html(
                '<section class="atlas-result-reading"><span>QUÉ SIGNIFICA PARA TI</span>'
                f"<h4>Una diferencia media de {format_number(result.metrics['MAE'])} unidades</h4>"
                f"<p>Al predecir <strong>{escape(str(result.config.get('target', 'la variable seleccionada')))}</strong>, esa fue la diferencia absoluta media entre las predicciones y los valores reales del conjunto de prueba. Cada caso puede tener un error mayor o menor.</p>"
                "<p>Para decidir si te sirve, considera las unidades y cuánto error puedes aceptar. Compara alternativas con los mismos datos y revisa el gráfico de errores de abajo.</p></section>"
            )
        elif problem == "clasificacion" and np.isfinite(result.metrics.get("Accuracy", np.nan)):
            st.info(
                f"El modelo acertó en el {format_number(result.metrics['Accuracy'], '.1%')} de los casos de prueba. Revisa también qué categorías confunde: un porcentaje global alto puede ocultar errores en categorías con pocos ejemplos."
            )
        first, second = st.columns(2)
        chart_types = (
            ["confusion", "roc"]
            if problem == "clasificacion"
            else ["actual_predicted", "residuals"]
        )
        for column, chart_type in zip([first, second], chart_types):
            with column.container(border=True):
                try:
                    summary_key = f"summary_{result.id}_{chart_type}"
                    if summary_key not in st.session_state.graphs:
                        st.session_state.graphs[summary_key] = engine.plot_result(
                            result, chart_type
                        )
                    diagnostic = st.session_state.graphs[summary_key]
                    figure(diagnostic, f"summary_{result.id}_{chart_type}")
                except (ValueError, TypeError) as exc:
                    st.info(f"Diagnóstico no disponible: {exc}")
        metrics, charts, predictions, advanced = st.tabs(
            [
                "Detalle de resultados",
                "Explorar gráficos",
                "Ver predicciones",
                "Entender el modelo (avanzado)",
            ]
        )
        with metrics:
            st.dataframe(
                pd.Series(result.metrics, name="valor").to_frame(), use_container_width=True
            )
            st.caption(
                "CV de regresión usa RMSE negativo (más alto es mejor). CV_seleccion es el valor usado para elegir parámetros, no una estimación independiente."
            )
            if problem == "clasificacion":
                st.dataframe(result.details["report"])
        with charts:
            kinds = (
                [
                    "confusion",
                    "confusion_normalized",
                    "roc",
                    "pr",
                    "probabilities",
                    "tree",
                    "correctness",
                ]
                if problem == "clasificacion"
                else ["actual_predicted", "residuals", "histogram", "qq"]
            )
            chart_names = {
                "confusion": "Qué categorías confunde",
                "confusion_normalized": "Errores por categoría en proporciones",
                "roc": "Detección frente a falsas alarmas (ROC)",
                "pr": "Aciertos y detección (Precision–Recall)",
                "probabilities": "Seguridad de las predicciones",
                "tree": "Árbol de decisiones",
                "correctness": "Predicciones correctas e incorrectas",
                "actual_predicted": "Valores reales frente a predicciones",
                "residuals": "Dónde y cuánto se equivoca",
                "histogram": "Cómo se distribuyen los errores",
                "qq": "Forma de los errores (Q–Q)",
            }
            chart = st.selectbox(
                "¿Qué quieres entender?", kinds, format_func=lambda value: chart_names[value]
            )
            if st.button("Ver este gráfico"):
                st.session_state.graphs[result.id] = engine.plot_result(result, chart)
            if result.id in st.session_state.graphs:
                figure(st.session_state.graphs[result.id], "model")
        with predictions:
            st.dataframe(result.predictions, use_container_width=True)
            st.download_button(
                "Predicciones CSV", export_frame(result.predictions), f"{result.id}.csv"
            )
            new_file = st.file_uploader(
                "Predecir nuevos datos (CSV UTF-8)", type=["csv"], key="predict_upload"
            )
            if new_file and st.button("Generar nuevas predicciones"):
                data = DatasetController.load(new_file).df
                st.session_state[f"new_predictions_{result.id}"] = engine.predict(result, data)
            if f"new_predictions_{result.id}" in st.session_state:
                st.dataframe(st.session_state[f"new_predictions_{result.id}"])
                st.download_button(
                    "Descargar nuevas predicciones",
                    export_frame(st.session_state[f"new_predictions_{result.id}"]),
                    "nuevas_predicciones.csv",
                )
        with advanced:
            st.caption(
                "La importancia sobre prueba sirve para interpretar; no selecciones variables con ella."
            )
            diagnostics = st.session_state.setdefault("diagnostics", {}).setdefault(result.id, {})
            if st.button("Calcular importancia por permutación"):
                diagnostics["importance"] = engine.feature_importance(result)
            if "importance" in diagnostics:
                importance = diagnostics["importance"]
                st.dataframe(importance)
                limit = (
                    st.slider(
                        "Variables a mostrar",
                        1,
                        min(50, len(importance)),
                        min(15, len(importance)),
                        key=f"importance_limit_{result.id}",
                    )
                    if len(importance) > 1
                    else 1
                )
                figure(diagnostic_bars(importance, "importance", limit), f"importance_{result.id}")
            if st.button("Consultar coeficientes"):
                diagnostics["coefficients"] = engine.coefficients(result)
            if "coefficients" in diagnostics:
                st.dataframe(diagnostics["coefficients"])
                limit = st.number_input(
                    "Máximo de coeficientes visibles",
                    1,
                    50,
                    15,
                    key=f"coefficient_limit_{result.id}",
                )
                figure(
                    diagnostic_bars(diagnostics["coefficients"], "coefficients", int(limit)),
                    f"coefficients_{result.id}",
                )
            if st.button("Calcular curva de aprendizaje"):
                diagnostics["learning"] = engine.learning_curve(result)
            if "learning" in diagnostics:
                table = diagnostics["learning"]
                figure(
                    style(
                        px.line(
                            table,
                            x="filas",
                            y=["entrenamiento", "validacion"],
                            color_discrete_sequence=PALETTE,
                            labels={
                                "filas": "Registros de entrenamiento",
                                "value": "Puntuación",
                                "variable": "Conjunto",
                            },
                        ),
                        "Curva de aprendizaje",
                    ),
                    "learning",
                )
        experiment_downloads(result)


def clustering_page() -> None:
    """Provide clustering page using validated inputs."""
    page_header("clustering")
    df = st.session_state.eda.df
    with st.container(border=True, key="workflow_cluster_features"):
        st.subheader("1. Elige qué quieres comparar")
        st.caption(
            "Selecciona características que describan tus registros. Evita identificadores como el número de cliente o de vivienda."
        )
        features = st.multiselect(
            "Características para encontrar grupos",
            list(df),
            default=st.session_state.eda.feature_roles()["suggested"][:4],
            key="analysis_cluster_columns",
        )
        roles = st.session_state.eda.feature_roles()
        flagged = [
            column
            for column in features
            if column in roles["identifiers"] + roles["dates"] + roles["constants"]
        ]
        if flagged:
            st.warning(
                "Revisa estas variables: "
                + ", ".join(flagged)
                + ". Los identificadores no describen similitud; las fechas necesitan preparación y las constantes no separan grupos."
            )
        with st.expander("Ajustar la preparación de las variables"):
            scale, categories = st.columns(2)
            scaler = scale.selectbox(
                "Cómo equilibrar las escalas",
                ["standard", "minmax", "robust", "none"],
                format_func=lambda value: {
                    "standard": "Equilibrar las variables (recomendado)",
                    "minmax": "Llevar los valores a un rango común",
                    "robust": "Reducir el efecto de valores extremos",
                    "none": "Mantener las escalas originales",
                }[value],
                key="analysis_cluster_scaler",
            )
            encoding = categories.selectbox(
                "Cómo representar categorías",
                ["onehot", "ordinal"],
                format_func=lambda value: {
                    "onehot": "Separar categorías sin darles un orden",
                    "ordinal": "Asignar un número a cada categoría",
                }[value],
            )
            st.caption(
                "La segunda opción introduce un orden numérico: úsala solo si tiene sentido para tus categorías."
            )
            seed = st.number_input(
                "Número para repetir la misma selección", 0, 99999, 42, key="analysis_cluster_seed"
            )
    with st.container(border=True, key="workflow_cluster_sample"):
        st.subheader("2. Define cuántos registros analizar")
        if len(df) > 5000:
            summary, settings = st.columns([1, 1.5])
            with summary:
                st.metric("Registros disponibles", format_number(len(df)))
                st.caption(
                    "Trabajaremos con una selección aleatoria. Tus datos originales se conservan."
                )
            with settings:
                sample_size = st.number_input(
                    "Registros que se incluirán",
                    min_value=3,
                    max_value=5000,
                    value=2000,
                    step=100,
                    key="cluster_sample_size",
                )
                use_sample = st.checkbox(
                    "Usar esta muestra para el agrupamiento",
                    key="cluster_use_sample",
                    help="Autoriza analizar únicamente la cantidad indicada; no elimina filas del dataset.",
                )
            st.caption(
                "Los grupos describirán solo la muestra. Jerárquico, Spectral y K-Medoids admiten hasta 2.000 registros."
            )
            if use_sample:
                df = ClusteringController.sample_for_analysis(df, int(sample_size), int(seed))
                st.caption(
                    f"Selección lista: {len(df):,} registros. Podrás ejecutar el análisis en el paso 3."
                )
            else:
                st.info(
                    "Marca la casilla de la muestra para continuar. No se ejecutará ningún análisis todavía."
                )
        else:
            use_sample = True
            st.caption(
                f"Se analizarán los {len(df):,} registros del dataset activo. No es necesario seleccionar una muestra."
            )
    if not use_sample:
        return
    st.session_state.selected_variables = features
    st.session_state.analysis_selection = {
        "problem": "clustering",
        "predictors": features,
        "scaler": scaler,
        "random_state": seed,
    }
    engine = ClusteringController(df, random_state=int(seed))
    train, representation = st.tabs(["Agrupamiento", "Representación avanzada"])
    with representation:
        method = st.selectbox("Reducción de dimensionalidad", ["pca", "tsne", "umap"])
        dimensions = st.radio("Dimensiones", [2, 3], horizontal=True)
        st.caption(
            "t-SNE y UMAP son exploratorios; las distancias visuales no equivalen a las originales."
        )
        if st.button("Calcular representación"):
            with st.spinner("Calculando representación…"):
                reduced = engine.reduce_dimensions(
                    features, method, dimensions, scaler=scaler, encoding=encoding
                )
                st.session_state.graphs["representation"] = engine.plot_representation(
                    reduced["datos"]
                )
                st.session_state.representation = reduced
        if "representation" in st.session_state.graphs:
            figure(st.session_state.graphs["representation"], "representation")
            variance = st.session_state.representation["varianza"]
            if variance is not None:
                st.write("Varianza explicada:", variance)
                components = st.session_state.representation["modelo"].components_
                st.dataframe(
                    pd.DataFrame(
                        components.T, columns=[f"PCA {i + 1}" for i in range(len(variance))]
                    )
                )
        metric = st.selectbox("Distancia", ["euclidean", "cityblock", "cosine"])
        if st.button("Calcular matriz de distancias"):
            st.dataframe(engine.distances(features, metric))
    with train:
        with st.container(border=True, key="workflow_cluster_method"):
            st.subheader("3. Encuentra grupos")
            st.caption(
                "Elige una alternativa y ejecuta el análisis. Cambiar opciones no vuelve a entrenar por sí solo."
            )
            st.session_state.setdefault("atlas_cluster_algorithm", "kmeans")
            algorithm_cards(
                [
                    (
                        "kmeans",
                        "Grupos similares",
                        "K-Means: una primera opción para grupos compactos.",
                    ),
                    (
                        "dbscan",
                        "Grupos y casos aislados",
                        "DBSCAN: busca zonas densas y señala casos sin grupo.",
                    ),
                    (
                        "hierarchical",
                        "Grupos por niveles",
                        "Jerárquico: explora cómo se unen los registros.",
                    ),
                    (
                        "kmedoids",
                        "Grupos representativos",
                        "K-Medoids: requiere una dependencia opcional.",
                    ),
                ],
                "atlas_cluster_algorithm",
                multiple=False,
            )
            algorithm = st.selectbox(
                "Método de agrupamiento",
                engine.algorithms(),
                key="atlas_cluster_algorithm",
                help="Las tarjetas son accesos rápidos. Aquí también están disponibles los demás métodos.",
            )
            params = {}
            if algorithm not in {"dbscan", "optics"}:
                k = st.slider(
                    "¿Cuántos grupos quieres explorar?",
                    2,
                    min(20, max(2, len(df) - 1)),
                    min(3, max(2, len(df) - 1)),
                )
                params["n_components" if algorithm == "gaussian" else "n_clusters"] = k
            if algorithm == "dbscan":
                params["eps"] = st.number_input("Radio eps", 0.01, 20.0, 0.5)
            if algorithm in {"dbscan", "optics"}:
                params["min_samples"] = st.number_input("Mínimo de vecinos", 2, 100, 5)
            if algorithm in {"hierarchical", "agglomerative"}:
                params["method" if algorithm == "hierarchical" else "linkage"] = st.selectbox(
                    "Enlace", ["ward", "complete", "average", "single"]
                )
            if st.button(
                "Encontrar grupos en mis datos",
                type="primary",
                disabled=len(features) == 0 or len(df) < 3,
                use_container_width=True,
            ):
                WorkflowController.training_started(st.session_state, "clustering", algorithm)
                with st.spinner("Calculando grupos…"):
                    result = engine.train(
                        algorithm, features, scaler=scaler, encoding=encoding, **params
                    )
                    remember(result, engine)
        with st.expander("Explorar el número de grupos y su estructura"):
            st.caption(
                "Estas acciones calculan análisis adicionales. El codo orienta qué cantidades probar; no garantiza un número óptimo."
            )
            diagnostic_scope = f"{st.session_state.eda.dataset_version.content_hash}:{features}:{scaler}:{encoding}:{seed}:{len(df)}"
            if st.button("Calcular método del codo"):
                table = engine.elbow(features, scaler=scaler, encoding=encoding)
                st.session_state.graphs[f"elbow:{diagnostic_scope}"] = style(
                    px.line(
                        table,
                        x="k",
                        y="inercia",
                        markers=True,
                        color_discrete_sequence=PALETTE,
                        labels={
                            "k": "Número de grupos (K)",
                            "inercia": "Inercia · suma de distancias cuadráticas",
                        },
                    ),
                    "¿Cómo cambia el agrupamiento al variar K?",
                )
            if f"elbow:{diagnostic_scope}" in st.session_state.graphs:
                st.caption(
                    "El cambio de pendiente puede orientar qué cantidades de grupos probar; no demuestra un K óptimo por sí solo."
                )
                figure(st.session_state.graphs[f"elbow:{diagnostic_scope}"], "elbow")
            if st.button("Crear dendrograma"):
                st.session_state.graphs[f"dendrogram:{diagnostic_scope}"] = engine.dendrogram(
                    features, scaler=scaler, encoding=encoding
                )
            if f"dendrogram:{diagnostic_scope}" in st.session_state.graphs:
                figure(st.session_state.graphs[f"dendrogram:{diagnostic_scope}"], "dendrogram")
    results = [e for e in st.session_state.experiments if e.problem == "clustering"]
    if not results:
        st.info("Ejecuta un algoritmo para explorar sus grupos y métricas.")
        return
    st.subheader("4. Comprende los grupos encontrados")
    st.caption("Revisa un análisis guardado; tus resultados anteriores siguen disponibles.")
    result = st.selectbox(
        "¿Qué agrupamiento quieres revisar?",
        results,
        format_func=lambda e: f"{e.name} · {len(e.predictions):,} registros",
    )
    saved = st.session_state.models[result.id]
    overview, details = st.tabs(["Entender mis grupos", "Evaluación técnica"])
    with details:
        metric_cards(
            [
                (name, f"{result.metrics[name]:.4g}", "Ruido excluido de los índices internos")
                for name in ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz", "ruido"]
                if name in result.metrics
            ]
        )
        st.dataframe(pd.Series(result.metrics, name="valor").to_frame())
        st.caption(
            "Ruido: cluster -1. Las métricas internas excluyen ruido; no se calculan para un solo grupo."
        )
        st.dataframe(saved.characterize(result))
    with overview:
        st.caption(
            "Comienza por el tamaño de cada grupo. Después selecciona uno para ver los registros que contiene."
        )
        figure(saved.cluster_sizes(result), f"overview_sizes_{result.id}")
        with st.expander("¿Qué caracteriza a cada grupo?"):
            try:
                figure(saved.profile_heatmap(result), f"profiles_{result.id}")
                st.caption(
                    "Medias en unidades originales: compara grupos dentro de cada variable. El grupo -1 representa ruido."
                )
            except ValueError as exc:
                st.info(str(exc))
        label = st.selectbox(
            "Inspeccionar grupo",
            sorted(set(result.details["labels"])),
            format_func=lambda value: "Ruido" if value == -1 else f"Grupo {value}",
        )
        st.dataframe(saved.inspect_cluster(result, int(label)), use_container_width=True)
        with st.expander("Representantes en espacio transformado"):
            st.dataframe(saved.representatives(result))
        dimensions = st.radio("Visualización de clusters", [2, 3], horizontal=True)
        if st.button("Ver mis grupos en un gráfico"):
            reduced = saved.reduce_dimensions(
                result.config["columns"],
                dimensions=dimensions,
                scaler=result.config["scaler"],
                encoding=result.config["encoding"],
            )
            st.session_state.graphs[f"clusters_{result.id}"] = saved.plot_representation(
                reduced["datos"], result.details["labels"]
            )
        if f"clusters_{result.id}" in st.session_state.graphs:
            figure(st.session_state.graphs[f"clusters_{result.id}"], "clusters")
            st.caption(
                "PCA es una proyección reducida; no conserva todas las distancias originales."
            )
        if st.button("Revisar qué tan separados están los grupos"):
            st.session_state.graphs[f"silhouette_{result.id}"] = saved.silhouette_plot(result)
        if f"silhouette_{result.id}" in st.session_state.graphs:
            figure(st.session_state.graphs[f"silhouette_{result.id}"], "silhouette")
        st.download_button(
            "Exportar etiquetas CSV", export_frame(result.predictions), "clusters.csv"
        )
    experiment_downloads(result)


def experiment_downloads(result: object) -> None:
    """Provide experiment downloads using validated inputs."""
    with st.container(border=True, key=f"workflow_export_{result.id}"):
        st.subheader("Guarda y comparte este resultado")
        st.download_button(
            "Descargar opciones del análisis",
            json.dumps(result.config, indent=2, default=str),
            f"{result.id}.json",
            "application/json",
        )
        st.download_button(
            "Descargar modelo para Python", result.model_bytes(), f"{result.id}.joblib"
        )
        st.caption(
            "Exportación para uso Python de confianza; la aplicación no carga archivos joblib."
        )
        report = html_report(
            f"Atlas · {result.name}",
            {
                "Métricas": pd.Series(result.metrics, name="valor").to_frame(),
                "Predicciones": result.predictions.head(100),
            },
            f"Tiempo total: {result.seconds:.3f} s. Configuración: {result.config}",
        )
        st.download_button(
            "Descargar informe de resultados", report, f"{result.id}.html", "text/html"
        )


def comparison_page() -> None:
    """Provide comparison page using validated inputs."""
    page_header("experiments")
    experiments = st.session_state.experiments
    if not experiments:
        st.info("Entrena modelos o crea agrupamientos para comenzar la comparación.")
        return
    cohorts = ComparisonController.groups(experiments)
    cohort = st.selectbox("Dataset, problema y partición compatibles", list(cohorts))
    st.caption(
        "Solo se ofrecen experimentos de la misma cohorte. Otras particiones permanecen en Experimentos."
    )
    selected = st.multiselect(
        "Experimentos", cohorts[cohort], format_func=lambda e: f"{e.problem} · {e.name} · {e.id}"
    )
    if not selected:
        return
    table = compare(selected)
    controls, criterion = st.columns([2, 1])
    metric = controls.selectbox("Métrica a comparar", list(table.select_dtypes(include=np.number)))
    lower_default = metric in {
        "MAE",
        "MSE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "ER",
        "Log_loss",
        "Davies_Bouldin",
        "Inercia",
        "segundos",
        "ruido",
    }
    direction = criterion.radio(
        "Criterio",
        ["Menor es mejor", "Mayor es mejor"],
        index=0 if lower_default else 1,
        horizontal=True,
    )
    st.session_state.comparison_selection = {
        "experiment_ids": [experiment.id for experiment in selected],
        "metric": metric,
        "direction": "min" if direction.startswith("Menor") else "max",
    }
    table, best, best_value = ComparisonController.rank(
        selected, metric, "min" if direction.startswith("Menor") else "max"
    )
    ComparisonController.record_evaluation(
        st.session_state, selected, metric, "min" if direction.startswith("Menor") else "max"
    )
    results_column, recommendation = st.columns([3, 1])
    chart_kind = st.selectbox(
        "Vista de comparación",
        ["ranking", "runtime", "cost", "cv", "roc", "pr"]
        + (["k_silhouette"] if all(result.problem == "clustering" for result in selected) else []),
        format_func={
            "ranking": "Rendimiento por modelo",
            "runtime": "Tiempo de entrenamiento",
            "cost": "Rendimiento frente a tiempo",
            "cv": "Distribución entre folds",
            "roc": "Superposición ROC (binaria)",
            "pr": "Superposición Precision-Recall (binaria)",
            "k_silhouette": "Separación según la cantidad de grupos probada",
        }.get,
    )
    try:
        figure(
            ComparisonController.chart(
                selected,
                metric,
                chart_kind,
                direction="min" if direction.startswith("Menor") else "max",
            ),
            "comparison_detail",
        )
    except ValueError as exc:
        st.info(str(exc))
    with results_column.container(border=True):
        st.subheader("Resultados de la comparación")
        st.dataframe(table, use_container_width=True)
    with recommendation.container(border=True):
        st.subheader("Recomendación según la métrica")
        st.markdown(f"**{table.loc[best, 'modelo']}**")
        st.metric(metric, f"{best_value:.4g}")
        st.caption(f"Experimento {best}")
        st.caption(
            "La conclusión depende de esta métrica, partición y contexto; no hay un ganador universal. Usar reiteradamente la prueba para seleccionar modelos sesga su evaluación final."
        )
    metric_cards(
        [
            ("Experimentos", len(selected), "Modelos seleccionados"),
            (
                "Tiempo promedio",
                f"{ComparisonController.summary(selected)['average_seconds']:.2f} s",
                "Por entrenamiento",
            ),
            ("Métrica", metric, direction),
        ]
    )
    with st.expander("Comparar varios indicadores lado a lado"):
        numeric_metrics = list(table.select_dtypes(include="number").columns)
        displayed_metrics = st.multiselect(
            "Indicadores a visualizar",
            numeric_metrics,
            default=[metric],
            key="comparison_displayed_metrics",
        )
        st.caption(
            "Cada panel conserva su escala. Las métricas con unidades distintas no se suman ni forman un ranking global."
        )
        if displayed_metrics:
            figure(
                ComparisonController.chart(selected, metric, "metrics", metrics=displayed_metrics),
                "comparison_metrics",
            )
    for result in selected:
        with st.expander(f"Parámetros · {result.name} · {result.id}"):
            st.json(result.config)
    st.download_button("Comparación CSV", export_frame(table.reset_index()), "comparacion.csv")
    st.download_button(
        "Comparación Excel", export_frame(table.reset_index(), "xlsx"), "comparacion.xlsx"
    )
    st.download_button(
        "Reporte comparativo HTML",
        html_report(
            "Atlas · Comparación",
            {"Resultados": table},
            f"Métrica: {metric}; criterio: {direction}",
        ),
        "comparacion.html",
        "text/html",
    )
